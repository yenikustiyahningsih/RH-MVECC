"""Reviewer 2 comment 5: detailed stability verification.

Three independent repeated-run procedures (all seeds explicit):
  A. Library resampling (as in the manuscript): 80% of the 45 base partitions, 50 repetitions,
     Generator(seed=123); every run is stored with the indices of dropped partitions
     (results/main/stability_runs.csv). Here additionally 60/70/90% fractions are tested.
  B. Seed-level re-runs: the complete pipeline is re-executed with master seeds 0..9
     (K-Means/GMM random_state = s, view-stability seeds 10s..10s+9, resampling seed 1000+s).
  C. Sample-level subsampling: 10 random 80% subsets of the MSMEs (Generator(seed=555)),
     complete pipeline re-executed on each subset; agreement with P* on the common MSMEs.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from rhmvecc.core import Config, RHMVECC, view_stage, ci95
from rhmvecc import msme


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out", default="results/stability")
    ap.add_argument("--repeats", type=int, default=50); ap.add_argument("--rerun-repeats", type=int, default=20)
    ap.add_argument("--n-seeds", type=int, default=10); ap.add_argument("--n-sub", type=int, default=10)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _, _, x, _ = msme.load(a.input)
    n = len(x); dx = squareform(pdist(x))
    ref = RHMVECC(Config(repeats=a.repeats)).run(x, msme.VIEWS, dist=dx)
    y = ref.labels_star; K = ref.k_star

    # A. library fraction sensitivity
    rows = []
    for frac in [0.6, 0.7, 0.8, 0.9]:
        ref.cfg.resample_frac = frac
        ref.stability(a.repeats)
        r = ref.stab[ref.stab.K == K].iloc[0]
        rows.append(dict(fraction=frac, n_partitions=int(frac * 45), K=K, **{c: r[c] for c in ref.stab.columns if c != "K"}))
    ref.cfg.resample_frac = 0.8
    pd.DataFrame(rows).to_csv(out / "A_library_fraction.csv", index=False)
    print("A done", flush=True)

    # B. seed-level re-runs
    rows = []
    for s in range(a.n_seeds):
        c = Config(seed=s, view_seeds=tuple(range(10 * s, 10 * s + 10)), resample_seed=1000 + s, repeats=a.rerun_repeats)
        m = RHMVECC(c).run(x, msme.VIEWS, dist=dx)
        r = m.selection[m.selection.K == m.k_star].iloc[0]
        rows.append(dict(seed=s, selected_K=m.k_star, SC=r.SC, DBI=r.DBI, CHI=r.CHI, VDI=r.VDI, stability_ARI=r.ARI_mean,
                         ARI_vs_reference=adjusted_rand_score(y, m.labels_star), NMI_vs_reference=normalized_mutual_info_score(y, m.labels_star),
                         ARI_K6_vs_reference=adjusted_rand_score(y, m.labels_at(K)),
                         **{f"w_{v}": float(m.vt.loc[m.vt.view == v, "weight"].iloc[0]) for v in msme.VIEWS},
                         **{f"T_{v}": float(m.vt.loc[m.vt.view == v, "T"].iloc[0]) for v in msme.VIEWS}))
        print("seed", s, m.k_star, rows[-1]["ARI_vs_reference"], flush=True)
    pd.DataFrame(rows).to_csv(out / "B_seed_reruns.csv", index=False)

    # C. sample-level subsampling
    rng = np.random.default_rng(555)
    rows = []
    for b in range(a.n_sub):
        ix = np.sort(rng.choice(n, int(0.8 * n), replace=False))
        xs = x[ix]
        m = RHMVECC(Config(repeats=a.rerun_repeats)).run(xs, msme.VIEWS, dist=dx[np.ix_(ix, ix)])
        r = m.selection[m.selection.K == m.k_star].iloc[0]
        rows.append(dict(subsample=b + 1, n=len(ix), selected_K=m.k_star, SC=r.SC, DBI=r.DBI, VDI=r.VDI, stability_ARI=r.ARI_mean,
                         ARI_vs_reference=adjusted_rand_score(y[ix], m.labels_star),
                         ARI_K6_vs_reference=adjusted_rand_score(y[ix], m.labels_at(K)),
                         **{f"w_{v}": float(m.vt.loc[m.vt.view == v, "weight"].iloc[0]) for v in msme.VIEWS}))
        print("sub", b, m.k_star, rows[-1]["ARI_vs_reference"], flush=True)
    pd.DataFrame(rows).to_csv(out / "C_sample_subsampling.csv", index=False)

    # summary
    summ = []
    for name, f, col in [("B seed re-runs", "B_seed_reruns.csv", "ARI_K6_vs_reference"), ("C 80% MSME subsamples", "C_sample_subsampling.csv", "ARI_K6_vs_reference")]:
        d = pd.read_csv(out / f)
        m_, s_, lo, hi = ci95(d[col])
        summ.append(dict(procedure=name, runs=len(d), share_K6=(d.selected_K == K).mean(), ARI_K6_mean=m_, ARI_K6_sd=s_, ci_low=lo, ci_high=hi,
                         ARI_K6_min=d[col].min()))
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
