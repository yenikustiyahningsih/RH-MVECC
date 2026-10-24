"""Complementary asymmetric dominance diagnostic (Reviewer 2, comment 2).
Single-view explained share  S_v(K) = I(P(K); P_v*) / H(P(K)),  SVS(K) = max_v S_v(K).
SVS = 1 means that P(K) is a function of (nested in) one view's reference partition,
which the symmetric NMI-based VDI can miss for coarse candidates (e.g. K = 2)."""
import sys, numpy as np, pandas as pd
from rhmvecc.core import Config, view_stage, entropy_bits, mutual_info_bits
from rhmvecc import msme
_, _, x, _ = msme.load(sys.argv[1])
cons = np.load("results/main/consensus_labels_allK.npy")
sel = pd.read_csv("results/main/selection.csv")
vt, refs, _, _ = view_stage(x, msme.VIEWS, Config())
rows = []
for j, k in enumerate(range(2, 11)):
    y = cons[:, j]; H = entropy_bits(y)
    s = {v: mutual_info_bits(y, r) / H for v, r in refs.items()}
    r = sel[sel.K == k].iloc[0]
    rows.append(dict(K=k, VDI=r.VDI, SVS=max(s.values()), SVS_view=max(s, key=s.get), **{f"S_{v}": s[v] for v in s},
                     Composite=r.Composite))
d = pd.DataFrame(rows)
for tau_s in [0.70, 0.80, 0.90]:
    e = d[(d.VDI <= 0.85) & (d.SVS <= tau_s)]
    print("VDI<=0.85 & SVS<=", tau_s, "->", None if e.empty else int(e.loc[e.Composite.idxmax(), "K"]))
d.to_csv("results/vdi/single_view_share_by_K.csv", index=False)
print(d.round(4).to_string())
