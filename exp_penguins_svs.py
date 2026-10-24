"""VDI vs SVS for the RH-MVECC K = 3 solution on Palmer Penguins (Section 5.14)."""
import numpy as np, pandas as pd
from sklearn.metrics import normalized_mutual_info_score as NMI
from exp_benchmarks import load_penguins
from rhmvecc.core import Config, RHMVECC, view_stage, vdi_profile, entropy_bits, mutual_info_bits
x, v, y = load_penguins("benchmarks/penguins.csv")
c = Config(); vr = view_stage(x, v, c)
m = RHMVECC(c).fit(x, v, view_result=vr)
z = m.labels_at(3); H = entropy_bits(z)
rows = [dict(view=k, NMI=vdi_profile(z, m.refs)[k], share=mutual_info_bits(z, r) / H, ref_K=int(vr[0].set_index("view").loc[k, "K"]),
             weight=float(m.vt.set_index("view").loc[k, "weight"])) for k, r in m.refs.items()]
d = pd.DataFrame(rows); d["NMI_with_island"] = NMI(z, np.argmax(x[:, :3], 1)); d["NMI_with_species"] = NMI(z, y)
d.to_csv("results/benchmarks/penguins_vdi_svs.csv", index=False); print(d.round(4))
