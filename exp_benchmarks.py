"""Reviewer 2 comment 3, Reviewer 3 comments 2 and 5: public multi-view benchmarks with ground truth.

Datasets (benchmarks/): 3Sources, BBCSport, MSRC-v5, Handwritten (UCI mfeat), Caltech101-7,
and Palmer Penguins as a small *heterogeneous mixed-type* benchmark (binary/categorical + continuous views).
Metrics: ACC (Hungarian), NMI, ARI, Purity; mean +- SD over 5 seeds.

Preprocessing of high-dimensional views (documented in the manuscript):
  text views (3Sources, BBCSport): row L2 normalisation -> TruncatedSVD (50 comps)
  image/feature views: z-score -> PCA (<= 50 comps)
  each view block is then rescaled so that its average per-feature variance equals 1
Penguins follows the MSME pipeline exactly (min-max, no reduction).

python exp_benchmarks.py --data benchmarks --out results/benchmarks
"""
import argparse, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph

from rhmvecc.core import Config, RHMVECC, consensus_cut, kmeans, view_stage, vdi

TEXT = {"3Sources", "BBCSport"}


def acc(y, z):
    c = contingency_matrix(y, z)
    r, k = linear_sum_assignment(-c)
    return c[r, k].sum() / len(y)


def purity(y, z):
    c = contingency_matrix(y, z)
    return c.max(0).sum() / len(y)


def scores(y, z):
    return dict(ACC=acc(y, z), NMI=NMI(y, z), ARI=ARI(y, z), Purity=purity(y, z))


def load_mat(path):
    d = sio.loadmat(path)
    Xs = [np.asarray(v.toarray() if hasattr(v, "toarray") else v, float) for v in d["X"].ravel()]
    y = np.asarray(d["y"]).ravel().astype(int)
    # some files store views as d x n
    Xs = [X if X.shape[0] == len(y) else X.T for X in Xs]
    return Xs, y


def preprocess(Xs, text, seed=0, max_dim=50):
    blocks = []
    for X in Xs:
        X = X[:, X.std(0) > 0]
        if text:
            X = normalize(X)
            if X.shape[1] > max_dim:
                X = TruncatedSVD(max_dim, random_state=seed).fit_transform(X)
        else:
            X = StandardScaler().fit_transform(X)
            if X.shape[1] > max_dim:
                X = PCA(max_dim, random_state=seed).fit_transform(X)
        X = X - X.mean(0)
        X = X / np.sqrt(X.var(0).mean())
        blocks.append(X)
    views, start = {}, 0
    for i, B in enumerate(blocks):
        views[f"V{i + 1}"] = list(range(start, start + B.shape[1])); start += B.shape[1]
    return np.hstack(blocks), views


def load_penguins(path):
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    y = df.species.factorize()[0]
    isl = pd.get_dummies(df.island).astype(float).values
    sex = (df.sex == "MALE").astype(float).values[:, None]
    x = np.hstack([isl, sex, df[["bill_length_mm", "bill_depth_mm"]].values, df[["flipper_length_mm", "body_mass_g"]].values])
    x = (x - x.min(0)) / np.ptp(x, 0)
    views = {"V1": [0, 1, 2, 3], "V2": [4, 5], "V3": [6, 7]}
    return x, views, y


def mv_spectral(x, views, k, seed):
    """Multi-view spectral baseline: average of per-view symmetric kNN graphs (kernel addition)."""
    A = None
    for cols in views.values():
        G = kneighbors_graph(x[:, cols], n_neighbors=10, include_self=False).toarray()
        G = np.maximum(G, G.T)
        A = G if A is None else A + G
    A /= len(views)
    return SpectralClustering(k, affinity="precomputed", random_state=seed, assign_labels="cluster_qr").fit_predict(A)


def add_nuisance(x, views, seed):
    r = np.random.default_rng(seed)
    nz = r.integers(0, 2, size=(len(x), 2)).astype(float)
    nz = (nz - nz.mean(0)) / nz.std(0)
    v = dict(views); v["Nuisance"] = [x.shape[1], x.shape[1] + 1]
    return np.hstack([x, nz]), v


def run_dataset(name, x, views, y, seeds, repeats, rows, extra_rows, tag=""):
    C = len(np.unique(y))
    dx = squareform(pdist(x))
    for s in seeds:
        t0 = time.perf_counter()
        cfg = Config(ks=tuple(range(2, max(10, C) + 1)), view_k_max=max(6, C), seed=s, resample_seed=100 + s, repeats=repeats)
        vr = view_stage(x, views, cfg)
        vt = vr[0]
        res = {}
        # single views
        sv = []
        for v, cols in views.items():
            z = kmeans(x[:, cols], C, s, 20); sv.append((v, scores(y, z), float(vt.loc[vt.view == v, "SC"].iloc[0])))
        best_sc = max(sv, key=lambda t: t[2]); best_or = max(sv, key=lambda t: t[1]["ACC"])
        res["Best single view (by SC)"] = best_sc[1]
        res["Best single view (oracle ACC)"] = best_or[1]
        res["Concatenated K-Means"] = scores(y, kmeans(x, C, s, 20))
        try:
            res["Multi-view spectral (avg. kNN graph)"] = scores(y, mv_spectral(x, views, C, s))
        except Exception as e:  # noqa
            print("spectral failed", e)
        m = RHMVECC(cfg).fit(x, views, dist=dx, view_result=vr)
        M = len(m.parts)
        res["Unweighted co-association"] = scores(y, consensus_cut(m.labels, np.ones(M), [C])[:, 0])
        res["Quality-weighted co-association"] = scores(y, consensus_cut(m.labels, cfg.eps + (1 - cfg.eps) * m.parts.quality.values, [C])[:, 0])
        m_eq = RHMVECC(replace(cfg, view_weight_mode="equal")).fit(x, views, dist=dx, view_result=vr)
        res["RH-MVECC, equal view weights (K = C)"] = scores(y, m_eq.labels_at(C))
        res["RH-MVECC (K = C)"] = scores(y, m.labels_at(C))
        m.stability(repeats).select()
        ks = m.k_star
        if ks is not None:
            res["RH-MVECC (automatic K)"] = {**scores(y, m.labels_star), "K_selected": ks}
        for meth, sc in res.items():
            rows.append(dict(dataset=name + tag, n=len(y), views=len(views), classes=C, seed=s, method=meth, **sc))
        for _, r in vt.iterrows():
            extra_rows.append(dict(dataset=name + tag, seed=s, view=r.view, dims=len(views[r.view]), U=r.U, G=r.Granularity, H=r.Entropy,
                                   SC=r.SC, T=r["T"], R=r.R, weight=r.weight, VDI_RH_KC=vdi(m.labels_at(C), m.refs)))
        print(f"{name}{tag} seed {s}: {time.perf_counter() - t0:.1f}s  RH(K=C) ACC={res['RH-MVECC (K = C)']['ACC']:.3f} "
              f"concat={res['Concatenated K-Means']['ACC']:.3f} K*={ks}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="benchmarks"); ap.add_argument("--out", default="results/benchmarks")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--only", default="")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seeds))
    names = ["3Sources", "BBCSport", "MSRC-v5", "Penguins", "Caltech101-7", "Handwritten"]
    if a.only: names = a.only.split(",")
    for name in names:
        rows, extra = [], []
        if name == "Penguins":
            x, views, y = load_penguins(Path(a.data) / "penguins.csv")
        else:
            Xs, y = load_mat(Path(a.data) / f"{name}.mat")
            x, views = preprocess(Xs, name in TEXT)
        run_dataset(name, x, views, y, seeds, a.repeats, rows, extra)
        if name in ("Handwritten", "Caltech101-7", "Penguins"):
            if name == "Penguins":
                r = np.random.default_rng(99); nz = r.integers(0, 2, size=(len(x), 2)).astype(float)
                xn, vn = np.hstack([x, nz]), {**views, "Nuisance": [x.shape[1], x.shape[1] + 1]}
            else:
                xn, vn = add_nuisance(x, views, 99)
            run_dataset(name, xn, vn, y, seeds, a.repeats, rows, extra, tag=" + nuisance view")
        pd.DataFrame(rows).to_csv(out / f"bench_{name}.csv", index=False)
        pd.DataFrame(extra).to_csv(out / f"views_{name}.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
