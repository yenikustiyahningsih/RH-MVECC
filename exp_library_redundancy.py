"""Reviewer 2 comment 3: redundancy among the 45 base partitions (pairwise ARI)."""
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score
L = np.load("results/main/library_labels.npy"); m = pd.read_csv("results/main/partitions.csv")
M = len(L); A = np.eye(M)
for i in range(M):
    for j in range(i + 1, M):
        A[i, j] = A[j, i] = adjusted_rand_score(L[i], L[j])
pd.DataFrame(A).to_csv("results/ablation/library_pairwise_ARI.csv", index=False)
fam = m.family.values; rows = []
for f1 in ["Full", "Weighted", "ViewSpecific"]:
    for f2 in ["Full", "Weighted", "ViewSpecific"]:
        mask = np.outer(fam == f1, fam == f2) & ~np.eye(M, dtype=bool)
        rows.append(dict(family_a=f1, family_b=f2, mean_ARI=A[mask].mean()))
iu = np.triu_indices(M, 1)
summary = dict(pairs=len(iu[0]), mean_ARI=A[iu].mean(), median_ARI=np.median(A[iu]), share_ARI_gt_0_9=(A[iu] > .9).mean(),
               n_pairs_ARI_gt_0_99=int((A[iu] > .99).sum()),
               partitions_with_near_duplicate=int(((A - np.eye(M)) > .99).any(1).sum()))
pd.DataFrame(rows).to_csv("results/ablation/library_redundancy_by_family.csv", index=False)
pd.DataFrame([summary]).to_csv("results/ablation/library_redundancy_summary.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string()); print(summary)
dup = [(i, j, round(A[i, j], 3)) for i, j in zip(*iu) if A[i, j] > .99]
print([(m.loc[i, 'family'] + m.loc[i, 'method'] + str(m.loc[i, 'K']) + str(m.loc[i, 'view']), m.loc[j, 'family'] + m.loc[j, 'method'] + str(m.loc[j, 'K']) + str(m.loc[j, 'view']), a) for i, j, a in dup])
