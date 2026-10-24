"""RH-MVECC core library (revision version).

Reliability-Weighted Heterogeneous Multi-View Ensemble Consensus Clustering with
View-Dominance Control.  This module is dataset-agnostic: it accepts any normalized
feature matrix X (n x p) and a dictionary of views {name: column indices}.

Every design parameter discussed by the reviewers is exposed in `Config`:
  alpha      -> quality-modulation weight in Eq. (4):  (1 - alpha) + alpha * SC_v
                alpha = 0.5 is the affine map of SC in [-1, 1] onto [0, 1].
  beta       -> optional exponent on the quality term (generalized Eq. 4).
  eps        -> reliability floor in Eq. (7): r_m = eps + (1 - eps) * q_m.
  use_T      -> include repeat-run stability T_v in Eq. (4).
  use_GH     -> include sqrt(G_v H_v) in Eq. (4).
  view_weight_mode -> 'reliability' (Eq. 4-5), 'equal', 'sc' (raw SC), 'adaptive' (entropy-weight)
  family_balance   -> each family receives 1/F of the total weight (Sec. 3.5).
  tau, comp_w      -> VDI threshold and composite weights of Eq. (11).

All random operations are seeded explicitly (see `Config.seed`, `Config.view_seeds`,
`Config.resample_seed`).
"""
from __future__ import annotations

import os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_k, "1")

from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.stats import t as student_t
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)


# ----------------------------------------------------------------------------- config
@dataclass
class Config:
    ks: tuple = tuple(range(2, 11))          # candidate K for Full/Weighted families and consensus
    view_k_max: int = 6                      # view-specific library uses K = 2..min(view_k_max, U_v)
    view_ref_k_max: int = 10                 # reference scan K = 2..min(view_ref_k_max, U_v)
    kmeans_n_init: int = 20
    seed: int = 42                           # seed of every K-Means / GMM fit in the library
    view_seeds: tuple = tuple(range(10))     # seeds used to estimate T_v
    view_n_init_T: int | None = None         # n_init used when estimating T_v (None -> kmeans_n_init)
    gmm_cov: str = "full"
    full_methods: tuple = ("KMeans", "GMM")
    families: tuple = ("Full", "Weighted", "ViewSpecific")
    alpha: float = 0.5
    beta: float = 1.0
    eps: float = 0.1
    use_T: bool = True
    use_GH: bool = True
    view_weight_mode: str = "reliability"    # reliability | equal | sc | adaptive
    family_balance: bool = True
    tau: float = 0.85
    comp_w: tuple = (0.30, 0.25, 0.15, 0.20, 0.10)   # SC, DBI(good), CHI, Stability, 1-VDI
    use_vdi_threshold: bool = True
    resample_frac: float = 0.8
    resample_seed: int = 123
    repeats: int = 50

    def as_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


# ----------------------------------------------------------------------------- helpers
def minmax(v):
    v = np.asarray(v, float)
    r = np.ptp(v)
    return (v - v.min()) / r if r > 0 else np.zeros_like(v)


def kmeans(x, k, seed=42, n_init=20):
    return KMeans(k, random_state=seed, n_init=n_init).fit_predict(x)


def internal_metrics(x, y, dist=None):
    if len(np.unique(y)) < 2:
        return dict(SC=np.nan, DBI=np.nan, CHI=np.nan)
    sc = silhouette_score(dist, y, metric="precomputed") if dist is not None else silhouette_score(x, y)
    return dict(SC=float(sc), DBI=float(davies_bouldin_score(x, y)), CHI=float(calinski_harabasz_score(x, y)))


def consensus_cut(labels, w, ks):
    """Weighted co-association (Eq. 8), D = 1 - C (Eq. 9), average linkage, cut at each K."""
    n = labels.shape[1]
    d = np.ones(n * (n - 1) // 2, dtype=np.float64)
    for y, a in zip(labels, w / w.sum()):
        d -= a * (pdist(y[:, None], metric="hamming") == 0)
    z = linkage(np.clip(d, 0, 1), method="average")
    return cut_tree(z, n_clusters=list(ks))


def entropy_bits(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_info_bits(a, b):
    from sklearn.metrics import mutual_info_score
    return float(mutual_info_score(a, b) / np.log(2))


def ci95(vals):
    vals = np.asarray(vals, float)
    m, s = vals.mean(), vals.std(ddof=1)
    h = student_t.ppf(0.975, len(vals) - 1) * s / np.sqrt(len(vals))
    return m, s, m - h, m + h


# ----------------------------------------------------------------------------- view stage
def view_stage(x, views, cfg: Config, dist_cache=None):
    """Granularity (Eq. 2), entropy (Eq. 3), reference scan, stability T_v and reliability (Eq. 4-5)."""
    n = len(x)
    rows, refs, vcache, scans = [], {}, {}, {}
    for v, cols in views.items():
        xv = x[:, cols]
        dv = squareform(pdist(xv)) if n <= 6000 else None
        _, counts = np.unique(np.round(xv, 10), axis=0, return_counts=True)
        u = len(counts)
        p = counts / n
        g = np.log(u) / np.log(n) if u > 1 else 0.0
        h = -(p * np.log(p)).sum() / np.log(u) if u > 1 else 0.0
        cand = []
        kmax = max(cfg.view_ref_k_max, cfg.view_k_max)
        for k in range(2, kmax + 1):
            if k > u:
                continue
            y = kmeans(xv, k, cfg.seed, cfg.kmeans_n_init)
            vcache[v, k] = y
            if k <= cfg.view_ref_k_max:
                cand.append({"K": k, **internal_metrics(xv, y, dv)})
        best = max(cand, key=lambda d: d["SC"])
        bk = best["K"]
        refs[v] = vcache[v, bk]
        n_init_T = cfg.view_n_init_T or cfg.kmeans_n_init
        T_runs = [adjusted_rand_score(refs[v], kmeans(xv, bk, s, n_init_T)) for s in cfg.view_seeds]
        T = float(np.mean(T_runs))
        rows.append(dict(view=v, U=u, Granularity=g, Entropy=h, T=T, T_min=float(np.min(T_runs)), **best))
        scans[v] = pd.DataFrame(cand)
    vt = pd.DataFrame(rows)
    vt["R"] = reliability(vt, cfg)
    vt["weight"] = view_weights(vt, cfg)
    return vt, refs, vcache, scans


def reliability(vt, cfg: Config):
    gh = np.sqrt(vt.Granularity * vt.Entropy) if cfg.use_GH else 1.0
    qual = ((1 - cfg.alpha) + cfg.alpha * vt.SC).clip(lower=0) ** cfg.beta
    T = vt["T"] if cfg.use_T else 1.0
    return gh * qual * T


def entropy_weight_method(M):
    """Classical entropy-weight method: columns of M (alternatives x criteria, >0) receive
    weights proportional to their dispersion (1 - normalized entropy)."""
    M = np.asarray(M, float)
    P = M / M.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = -np.nansum(np.where(P > 0, P * np.log(P), 0), axis=0) / np.log(M.shape[0])
    d = 1 - E
    return d / d.sum() if d.sum() > 0 else np.full(M.shape[1], 1 / M.shape[1])


def view_weights(vt, cfg: Config):
    V = len(vt)
    if cfg.view_weight_mode == "equal":
        w = np.full(V, 1.0 / V)
    elif cfg.view_weight_mode == "sc":
        w = vt.SC.clip(lower=1e-9).values
    elif cfg.view_weight_mode == "adaptive":
        # data-driven exponents for the three reliability factors via entropy-weight method
        comps = np.column_stack([np.sqrt(vt.Granularity * vt.Entropy), (1 + vt.SC) / 2, vt["T"]])
        comps = np.clip(comps, 1e-9, None)
        lam = entropy_weight_method(comps) * comps.shape[1]   # exponents average to 1
        w = np.prod(comps ** lam, axis=1)
    else:
        w = vt.R.values
    w = np.asarray(w, float)
    return w / w.sum()


# ----------------------------------------------------------------------------- library stage
def build_library(x, views, vt, vcache, cfg: Config, dist):
    wv = dict(zip(vt.view, vt.weight))
    parts, meta = [], []

    def add(y, family, method, k, v=""):
        parts.append(np.asarray(y))
        meta.append(dict(family=family, method=method, K=k, view=v, **internal_metrics(x, y, dist)))

    p = x.shape[1]
    if "Full" in cfg.families:
        for k in cfg.ks:
            if "KMeans" in cfg.full_methods:
                add(kmeans(x, k, cfg.seed, cfg.kmeans_n_init), "Full", "KMeans", k)
            if "GMM" in cfg.full_methods:
                cov = cfg.gmm_cov if p <= 60 else "diag"
                y = GaussianMixture(k, random_state=cfg.seed, n_init=1, covariance_type=cov,
                                    reg_covar=1e-6).fit_predict(x)
                add(y, "Full", "GMM", k)
    if "Weighted" in cfg.families:
        scales = np.zeros(p)
        for v, cols in views.items():
            scales[cols] = np.sqrt(wv[v] / len(cols))
        for k in cfg.ks:
            add(kmeans(x * scales, k, cfg.seed, cfg.kmeans_n_init), "Weighted", "KMeans", k)
    if "ViewSpecific" in cfg.families:
        for v in views:
            u = int(vt.loc[vt.view == v, "U"].iloc[0])
            for k in range(2, min(cfg.view_k_max, u) + 1):
                add(vcache[v, k], "ViewSpecific", "KMeans", k, v)
    labels = np.asarray(parts)
    m = pd.DataFrame(meta)
    m["weight"] = partition_weights(m, wv, cfg)
    return labels, m


def partition_weights(m, wv, cfg: Config):
    q = np.zeros(len(m))
    for fam, ix in m.groupby("family").groups.items():
        q[ix] = (minmax(m.loc[ix, "SC"]) + 1 - minmax(m.loc[ix, "DBI"]) + minmax(m.loc[ix, "CHI"])) / 3
    raw = cfg.eps + (1 - cfg.eps) * q
    for i, row in m.iterrows():
        if row.family == "ViewSpecific":
            raw[i] *= wv[row["view"]]
    m["quality"] = q
    m["raw_reliability"] = raw
    if cfg.family_balance:
        fams = m.family.unique()
        w = np.zeros(len(m))
        for fam, ix in m.groupby("family").groups.items():
            w[ix] = raw[ix] / raw[ix].sum() / len(fams)
    else:
        w = raw / raw.sum()
    return w


# ----------------------------------------------------------------------------- dominance
def vdi_profile(y, refs):
    return {v: float(normalized_mutual_info_score(y, r)) for v, r in refs.items()}


def vdi(y, refs):
    return max(vdi_profile(y, refs).values())


# ----------------------------------------------------------------------------- full pipeline
class RHMVECC:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()

    def fit(self, x, views, dist=None, verbose=False, view_result=None):
        cfg = self.cfg
        x = np.asarray(x, float)
        self.x, self.views = x, views
        self.dist = squareform(pdist(x)) if dist is None else dist
        if view_result is None:
            view_result = view_stage(x, views, cfg)
        self.vt, self.refs, self.vcache, self.scans = view_result
        # view weights may need recomputing if cfg changed but view_result was reused
        self.vt = self.vt.copy()
        self.vt["R"] = reliability(self.vt, cfg)
        self.vt["weight"] = view_weights(self.vt, cfg)
        self.labels, self.parts = build_library(x, views, self.vt, self.vcache, cfg, self.dist)
        self.consensus = consensus_cut(self.labels, self.parts.weight.values, cfg.ks)
        rows = []
        for j, k in enumerate(cfg.ks):
            y = self.consensus[:, j]
            prof = vdi_profile(y, self.refs)
            rows.append(dict(K=k, **internal_metrics(x, y, self.dist), VDI=max(prof.values()),
                             VDI_view=max(prof, key=prof.get),
                             **{f"NMI_{v}": s for v, s in prof.items()}))
        self.kscan = pd.DataFrame(rows)
        if verbose:
            print(self.vt.to_string(index=False), flush=True)
        return self

    def stability(self, repeats=None, seed=None, verbose=False):
        cfg = self.cfg
        repeats = cfg.repeats if repeats is None else repeats
        rng = np.random.default_rng(cfg.resample_seed if seed is None else seed)
        M = len(self.parts)
        w = self.parts.weight.values
        runs = []
        for rep in range(repeats):
            ix = rng.choice(M, int(cfg.resample_frac * M), replace=False)  # order kept as drawn (reproducibility)
            sub = consensus_cut(self.labels[ix], w[ix], cfg.ks)
            for j, k in enumerate(cfg.ks):
                runs.append(dict(repeat=rep + 1, K=k, n_partitions=len(ix),
                                 ARI=adjusted_rand_score(self.consensus[:, j], sub[:, j]),
                                 NMI=normalized_mutual_info_score(self.consensus[:, j], sub[:, j]),
                                 dropped=";".join(map(str, sorted(set(range(M)) - set(ix))))))
            if verbose and (rep + 1) % 10 == 0:
                print(f"  resampling {rep + 1}/{repeats}", flush=True)
        self.stab_runs = pd.DataFrame(runs)
        st = []
        for k, g in self.stab_runs.groupby("K"):
            row = {"K": k}
            for met in ["ARI", "NMI"]:
                m_, s_, lo, hi = ci95(g[met])
                row.update({f"{met}_mean": m_, f"{met}_sd": s_, f"{met}_ci_low": lo, f"{met}_ci_high": hi,
                            f"{met}_min": g[met].min()})
            st.append(row)
        self.stab = pd.DataFrame(st)
        self._rng = rng
        return self

    def select(self):
        cfg = self.cfg
        sel = self.kscan.merge(self.stab, on="K")
        comps = np.column_stack([minmax(sel.SC), 1 - minmax(sel.DBI), minmax(sel.CHI),
                                 minmax(sel.ARI_mean), 1 - sel.VDI])
        self.components = comps
        sel["Composite"] = comps @ np.asarray(cfg.comp_w)
        sel["Eligible"] = (sel.VDI <= cfg.tau) if cfg.use_vdi_threshold else True
        el = sel[sel.Eligible]
        self.selection = sel
        self.k_star = int(el.loc[el.Composite.idxmax(), "K"]) if len(el) else None
        self.labels_star = self.consensus[:, list(cfg.ks).index(self.k_star)] if self.k_star else None
        return self

    def run(self, x, views, dist=None, verbose=False, view_result=None, repeats=None):
        return self.fit(x, views, dist, verbose, view_result).stability(repeats, verbose=verbose).select()

    def labels_at(self, k):
        return self.consensus[:, list(self.cfg.ks).index(k)]


def canonical(y):
    """Renumber clusters 1..K by descending size (cluster numbers are not class labels)."""
    order = pd.Series(y).value_counts().index
    mp = {old: i + 1 for i, old in enumerate(order)}
    return np.array([mp[z] for z in y])
