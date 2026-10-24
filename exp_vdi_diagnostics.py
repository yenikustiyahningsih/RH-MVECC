"""Reviewer 2 comment 2 / Reviewer 3 comment 1: what does VDI below tau mean, and does VDI
detect single-view dominance?

Outputs (results/vdi):
  nmi_profile_by_K.csv            NMI of every consensus P(K) with every single-view reference
  information_decomposition.csv   share of H(P*) explained by V1, V1+V2, ... (bits)
  contingency_V1_by_cluster.csv   P* versus the four V1 patterns (NIB x Land)
  surrogate_tree.csv              how well each view subset can reproduce P* (5-fold CV)
  reference_partitions.csv        VDI of constructed reference partitions (pure V1, V1 x Vj, ...)
  nuisance_injection.csv          stress test: an artificial, perfectly separable binary view is added
"""
import argparse
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from rhmvecc.core import Config, RHMVECC, view_stage, vdi_profile, entropy_bits, mutual_info_bits, kmeans, internal_metrics
from rhmvecc import msme


def joint(*labs):
    return pd.Series(list(zip(*labs))).astype(str).factorize()[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out", default="results/vdi")
    ap.add_argument("--repeats", type=int, default=50)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ids, data, x, info = msme.load(a.input)
    dx = squareform(pdist(x))
    cfg = Config(repeats=a.repeats)
    m = RHMVECC(cfg).run(x, msme.VIEWS, dist=dx)
    refs = m.refs
    y = m.labels_star; K = m.k_star

    # (a) NMI profile
    prof = m.selection[["K", "VDI", "VDI_view"] + [f"NMI_{v}" for v in msme.VIEWS] + ["Eligible", "Composite"]].copy()
    prof["dominance_gap"] = prof[[f"NMI_{v}" for v in msme.VIEWS]].apply(lambda r: np.sort(r.values)[-1] - np.sort(r.values)[-2], axis=1)
    prof.to_csv(out / "nmi_profile_by_K.csv", index=False)

    # (b) information decomposition of P* (bits), with view patterns/reference partitions
    H = entropy_bits(y)
    rows = []
    order = ["V1", "V2", "V3", "V4"]
    for r in range(1, 5):
        for sub in combinations(order, r):
            j = joint(*[refs[v] for v in sub])
            rows.append(dict(views="+".join(sub), n_views=r, I_bits=mutual_info_bits(y, j), H_Pstar_bits=H,
                             share_of_H=mutual_info_bits(y, j) / H, NMI=normalized_mutual_info_score(y, j)))
    dec = pd.DataFrame(rows); dec.to_csv(out / "information_decomposition.csv", index=False)

    # (c) contingency of P* versus V1 pattern and V2 reference
    from rhmvecc.core import canonical
    yc = canonical(y)
    v1pat = pd.Series([f"NIB={int(a_)}, Land={int(b_)}" for a_, b_ in x[:, [0, 7]]])
    ct = pd.crosstab(pd.Series(yc, name="Cluster"), v1pat)
    ct2 = pd.crosstab(pd.Series(yc, name="Cluster"), pd.Series(np.where(x[:, 2] > np.median(x[:, 2]), "high capacity", "low capacity"), name="capacity"))
    pd.concat([ct, ct2], axis=1).to_csv(out / "contingency_V1_by_cluster.csv")

    # (d) surrogate decision trees: can a single view reproduce P*?
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    st = []
    for r in range(1, 5):
        for sub in combinations(order, r):
            cols = sum([msme.VIEWS[v] for v in sub], [])
            for depth in [3, 6]:
                acc = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=0), x[:, cols], yc, cv=skf, scoring="accuracy")
                bacc = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=0), x[:, cols], yc, cv=skf, scoring="balanced_accuracy")
                st.append(dict(views="+".join(sub), max_depth=depth, accuracy=acc.mean(), balanced_accuracy=bacc.mean()))
    pd.DataFrame(st).to_csv(out / "surrogate_tree.csv", index=False)

    # (e) VDI of constructed reference partitions
    rng = np.random.default_rng(7)
    refs_rows = []
    cand = {"Pure V1 reference (NIB x Land)": refs["V1"],
            "V1 x V2 reference": joint(refs["V1"], refs["V2"]),
            "V1 x V3 reference": joint(refs["V1"], refs["V3"]),
            "V1 x V4 reference": joint(refs["V1"], refs["V4"]),
            "V2 x V3 reference": joint(refs["V2"], refs["V3"]),
            "V1 x V2 x V3 x V4 reference": joint(*[refs[v] for v in order]),
            "Random partition, sizes of P*": rng.permutation(y),
            f"RH-MVECC P* (K={K})": y}
    for kk in [3, 4]:
        cand[f"RH-MVECC consensus K={kk}"] = m.labels_at(kk)
    for name, z in cand.items():
        p_ = vdi_profile(z, refs)
        refs_rows.append(dict(partition=name, K=len(np.unique(z)), VDI=max(p_.values()), **{f"NMI_{v}": s for v, s in p_.items()},
                              ARI_vs_Pstar=adjusted_rand_score(y, z)))
    pd.DataFrame(refs_rows).to_csv(out / "reference_partitions.csv", index=False)

    # (f) nuisance-view injection: add a fifth view of two independent Bernoulli(0.5) attributes.
    # The view is perfectly separable (4 patterns, SC = 1) but carries no information about the MSMEs.
    rows = []
    for rep, seed in enumerate([11, 12, 13]):
        r_ = np.random.default_rng(seed)
        nz = r_.integers(0, 2, size=(len(x), 2)).astype(float)
        xn = np.hstack([x, nz])
        views5 = dict(msme.VIEWS); views5["V5_nuisance"] = [8, 9]
        dxn = squareform(pdist(xn))
        for mode in ["reliability", "sc", "equal"]:
            c = Config(repeats=a.repeats, view_weight_mode=mode)
            mm = RHMVECC(c).run(xn, views5, dist=dxn)
            w5 = float(mm.vt.loc[mm.vt.view == "V5_nuisance", "weight"].iloc[0])
            # without VDI control (pure composite) for comparison
            sel = mm.selection
            k_noctrl = int(sel.loc[sel.Composite.idxmax(), "K"])
            ks = mm.k_star
            for label, kk in [("with VDI control", ks), ("without VDI control", k_noctrl)]:
                if kk is None:
                    rows.append(dict(seed=seed, view_weighting=mode, selection=label, K=None)); continue
                z = mm.labels_at(kk)
                pr = vdi_profile(z, mm.refs)
                rows.append(dict(seed=seed, view_weighting=mode, selection=label, K=kk, weight_V5=w5,
                                 VDI=max(pr.values()), VDI_view=max(pr, key=pr.get), NMI_nuisance=pr["V5_nuisance"],
                                 ARI_vs_original_Pstar=adjusted_rand_score(y, z), NMI_vs_original_Pstar=normalized_mutual_info_score(y, z)))
        zk = kmeans(xn, K, 42, 20)
        pr = vdi_profile(zk, {**refs, "V5_nuisance": joint(nz[:, 0], nz[:, 1])})
        rows.append(dict(seed=seed, view_weighting="concatenated K-Means", selection=f"fixed K={K}", K=K, VDI=max(pr.values()),
                         VDI_view=max(pr, key=pr.get), NMI_nuisance=pr["V5_nuisance"], ARI_vs_original_Pstar=adjusted_rand_score(y, zk),
                         NMI_vs_original_Pstar=normalized_mutual_info_score(y, zk)))
        print("nuisance seed", seed, "done", flush=True)
    pd.DataFrame(rows).to_csv(out / "nuisance_injection.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
