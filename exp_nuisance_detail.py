"""Detail of the nuisance-view stress test (seed 11, reliability weighting): full K scan with
VDI/SVS per view and agreement with the original P*, to show which tau would veto the nuisance view."""
import sys, numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
from rhmvecc.core import Config, RHMVECC, entropy_bits, mutual_info_bits
from rhmvecc import msme
_, _, x, _ = msme.load(sys.argv[1])
y0 = np.load("results/main/consensus_labels_allK.npy")[:, 4]
nz = np.random.default_rng(11).integers(0, 2, size=(len(x), 2)).astype(float)
xn = np.hstack([x, nz]); views = dict(msme.VIEWS); views["V5_nuisance"] = [8, 9]
m = RHMVECC(Config(repeats=50)).run(xn, views, dist=squareform(pdist(xn)))
rows = []
for k in m.cfg.ks:
    y = m.labels_at(k); H = entropy_bits(y)
    s = {v: mutual_info_bits(y, r) / H for v, r in m.refs.items()}
    r = m.selection[m.selection.K == k].iloc[0]
    rows.append(dict(K=k, SC=r.SC, DBI=r.DBI, ARI_stab=r.ARI_mean, VDI=r.VDI, VDI_view=r.VDI_view, SVS=max(s.values()),
                     SVS_view=max(s, key=s.get), S_nuisance=s["V5_nuisance"], S_V1=s["V1"], Composite=r.Composite,
                     ARI_vs_original_K6=adjusted_rand_score(y0, y)))
d = pd.DataFrame(rows); d.to_csv("results/vdi/nuisance_detail_seed11.csv", index=False)
for tau in [0.85, 0.80, 0.75, 0.70]:
    e = d[d.VDI <= tau]
    k = None if e.empty else int(e.loc[e.Composite.idxmax(), "K"])
    print(tau, k, None if k is None else round(float(d[d.K == k].ARI_vs_original_K6.iloc[0]), 4))
print(m.vt[["view", "U", "Granularity", "Entropy", "SC", "R", "weight"]].round(4).to_string())
print(d.round(4).to_string())
