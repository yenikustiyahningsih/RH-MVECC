"""Build results/benchmarks/summary_mean.csv from bench_*.csv.
Mean over seeds of ACC, NMI, ARI, Purity per dataset and method, on the ORIGINAL views only
(rows whose dataset name contains 'nuisance' are excluded, so the two blocks are never averaged together).
Usage (from the repository root):  python make_summary_mean.py
"""
from pathlib import Path
import pandas as pd

d = Path("results/benchmarks")
files = sorted(d.glob("bench_*.csv"))
if not files:
    raise SystemExit("No bench_*.csv found in results/benchmarks")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df[~df["dataset"].str.contains("nuisance", case=False)]
cols = [c for c in ["ACC", "NMI", "ARI", "Purity", "K_selected"] if c in df.columns]
g = df.groupby(["dataset", "method"], sort=False)
out = g[cols].mean().round(4)
out.insert(0, "seeds", g["seed"].nunique())
out.reset_index().to_csv(d / "summary_mean.csv", index=False)
print("written", d / "summary_mean.csv", "rows:", len(out), "datasets:", df["dataset"].nunique())