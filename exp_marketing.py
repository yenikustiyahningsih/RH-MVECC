"""Second application study WITHOUT ground truth: Customer Personality Analysis (Kaggle "marketing_campaign").

Reviewer 2 comment 3 / Reviewer 3 comment 2 (evidence beyond a single domain).
The dataset has no class labels, so ACC/NMI/ARI against ground truth are NOT reported.
Instead the study mirrors the MSME analysis: SC, DBI, CHI, ensemble stability (ARI), view weights,
VDI, baselines on the same feature matrix, and cluster profiles. The campaign column `Response`
is never used as a feature or as a label; it is only reported per cluster after clustering.

python exp_marketing.py --input data/marketing_campaign_rapi.xlsx --out results/marketing

Accepted inputs: the .xlsx given with the project (sheet 1) or the original Kaggle file
(marketing_campaign.csv, tab-separated).
"""
import argparse, json, time, hashlib, platform
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rhmvecc.core import Config, RHMVECC, consensus_cut, internal_metrics, vdi, vdi_profile, kmeans, canonical, view_stage

# ----------------------------------------------------------------------------- feature definition
EDU = {"Basic": 0, "Graduation": 1, "2n Cycle": 2, "Master": 2, "PhD": 3}
PARTNER = {"Married": 1, "Together": 1}          # everything else (Single, Divorced, Widow, Alone, Absurd, YOLO) -> 0
SPEND = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CHAN = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth"]
CAMP = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Complain"]

FEATURES = (["Age", "Education", "Partner", "Income", "Kidhome", "Teenhome"] + SPEND + CHAN + ["Recency", "Tenure"] + CAMP)
VIEWS = {"V1": list(range(0, 6)), "V2": list(range(6, 12)), "V3": list(range(12, 19)), "V4": list(range(19, 25))}
VIEW_NAMES = {"V1": "Customer profile", "V2": "Product spending", "V3": "Channels & engagement", "V4": "Campaign response history"}


def load_raw(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path, sep=None, engine="python")


def prepare(df):
    """Deterministic preprocessing (all steps are reported in the manuscript). No row is deleted."""
    log = {}
    d = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    ref = dt.max()
    d["Age"] = ref.year - df["Year_Birth"]
    d["Education"] = df["Education"].map(EDU)
    d["Partner"] = df["Marital_Status"].map(PARTNER).fillna(0).astype(float)
    log["unmapped_education"] = int(d["Education"].isna().sum())
    log["income_missing_imputed_median"] = int(df["Income"].isna().sum())
    d["Income"] = np.log1p(df["Income"].fillna(df["Income"].median()))
    d["Kidhome"] = df["Kidhome"]; d["Teenhome"] = df["Teenhome"]
    for c in SPEND:
        d[c] = np.log1p(df[c])
    for c in CHAN:
        d[c] = df[c]
    d["Recency"] = df["Recency"]
    d["Tenure"] = (ref - dt).dt.days
    for c in CAMP:
        d[c] = df[c]
    d = d[FEATURES].astype(float)
    # winsorise continuous / count features at the 1st and 99th percentiles (outliers such as Year_Birth < 1900, Income = 666,666)
    wins = ["Age", "Income"] + SPEND + CHAN + ["Recency", "Tenure"]
    for c in wins:
        lo, hi = d[c].quantile([.01, .99])
        log[f"winsorised_{c}"] = int(((d[c] < lo) | (d[c] > hi)).sum())
        d[c] = d[c].clip(lo, hi)
    mm = d.copy()
    for c in d.columns:
        r = d[c].max() - d[c].min()
        mm[c] = (d[c] - d[c].min()) / r if r > 0 else 0.0
    log["reference_date"] = str(ref.date())
    return d, mm, log


def to_original(raw_df, df_proc, labels):
    """Readable profile: medians on the original scale per cluster."""
    o = pd.DataFrame({"Age": df_proc["Age"], "Income": raw_df["Income"], "Kidhome": raw_df["Kidhome"], "Teenhome": raw_df["Teenhome"],
                      **{c: raw_df[c] for c in SPEND + CHAN + ["Recency"]}, "Tenure_days": df_proc["Tenure"],
                      "Partner_share": df_proc["Partner"], "Campaigns_accepted_1to5": raw_df[CAMP[:5]].sum(1), "Complain": raw_df["Complain"]})
    o["Cluster"] = labels
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="results/marketing")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--tau", type=float, default=0.85)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    raw = load_raw(a.input)
    proc, mm, plog = prepare(raw)
    x = mm.to_numpy(float)
    n = len(x)
    pd.concat([raw[["ID"]], mm], axis=1).to_csv(out / "marketing_normalized.csv", index=False)   # anonymous, shareable matrix
    pd.DataFrame({"feature": FEATURES, "view": [v for f in range(len(FEATURES)) for v, c in VIEWS.items() if f in c],
                  "mean": x.mean(0), "sd": x.std(0, ddof=1), "unique": [len(np.unique(x[:, j])) for j in range(x.shape[1])]}).to_csv(out / "audit.csv", index=False)
    dx = squareform(pdist(x))

    cfg = Config(tau=a.tau, repeats=a.repeats)
    vr = view_stage(x, VIEWS, cfg)
    model = RHMVECC(cfg).fit(x, VIEWS, dist=dx, view_result=vr, verbose=True)
    model.stability(verbose=True).select()
    k = model.k_star
    if k is None:
        raise SystemExit("No K satisfies VDI <= tau; inspect selection.csv written below and relax tau.")
    y = model.labels_star
    vt = model.vt.copy(); vt.insert(1, "meaning", vt.view.map(VIEW_NAMES))

    T = {"view_reliability": vt, "selection": model.selection, "partitions": model.parts}
    for v, s in model.scans.items():
        T["scan_" + v] = s

    # baselines on the same matrix, all at K*
    base, rt, bl = [], [], {}
    def timed(name, fn):
        s = time.perf_counter(); z = fn(); rt.append(dict(method=name, seconds=time.perf_counter() - s)); return z
    bl["K-Means"] = timed("K-Means", lambda: kmeans(x, k, 42, 20))
    bl["GMM"] = timed("GMM", lambda: GaussianMixture(k, random_state=42, n_init=1, covariance_type="full", reg_covar=1e-6).fit_predict(x))
    bl["Ward hierarchical"] = timed("Ward hierarchical", lambda: cut_tree(linkage(pdist(x), "ward"), n_clusters=[k])[:, 0])
    M = len(model.parts)
    bl["Unweighted consensus"] = consensus_cut(model.labels, np.ones(M), [k])[:, 0]
    bl["Quality-weighted consensus"] = consensus_cut(model.labels, cfg.eps + (1 - cfg.eps) * model.parts.quality.values, [k])[:, 0]
    m_eq = RHMVECC(replace(cfg, view_weight_mode="equal")).fit(x, VIEWS, dist=dx, view_result=vr)
    bl["RH-MVECC, equal view weights"] = m_eq.labels_at(k)
    bl["RH-MVECC"] = y
    for name, z in bl.items():
        base.append(dict(method=name, K=k, **internal_metrics(x, z, dx), VDI=vdi(z, model.refs), ARI_vs_RH=adjusted_rand_score(y, z)))
    T["baselines"] = pd.DataFrame(base)
    T["runtime"] = pd.DataFrame(rt)

    # K-Means over K for reference
    km = []
    for kk in cfg.ks:
        z = kmeans(x, kk, 42, 20)
        km.append(dict(K=kk, **internal_metrics(x, z, dx), VDI=vdi(z, model.refs)))
    T["kmeans_scan"] = pd.DataFrame(km)

    # profiles (post-hoc; `Response` is NOT used for clustering)
    final = canonical(y)
    orig = to_original(raw, proc, final)
    prof = mm.copy(); prof["Cluster"] = final
    P = prof.groupby("Cluster").mean(); P["n"] = prof.groupby("Cluster").size(); P["share"] = P["n"] / n
    T["profiles_normalized"] = P
    T["profiles_original_median"] = orig.groupby("Cluster").median()
    resp = pd.DataFrame({"Cluster": final, "Response": raw["Response"].values}).groupby("Cluster").Response.agg(["mean", "sum", "count"])
    resp.columns = ["response_rate", "responders", "n"]
    T["response_rate_by_cluster_posthoc"] = resp
    asg = pd.DataFrame({"ID": raw["ID"], "Cluster": final})
    T["assignments"] = asg

    for name, tab in T.items():
        tab.to_csv(out / f"{name}.csv", index=name.startswith(("profiles", "response_rate")))

    sel = model.selection
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, lab in zip(axes, ["Composite", "ARI_mean", "VDI"], ["Composite score J(K)", "Stability ARI", "VDI"]):
        ax.plot(sel.K, sel[col], "o-", color="#1f4e79"); ax.set(xlabel="K", ylabel=lab); ax.axvline(k, color="gray", ls="--"); ax.grid(alpha=.3)
    axes[2].axhline(cfg.tau, color="#c00000", ls=":", label=f"τ = {cfg.tau}"); axes[2].legend()
    fig.tight_layout(); fig.savefig(out / "fig_selection.png", dpi=200); plt.close(fig)

    r = sel[sel.K == k].iloc[0]
    summary = dict(n=n, p=x.shape[1], views={v: len(c) for v, c in VIEWS.items()}, selected_K=k,
                   metrics={c: float(r[c]) for c in ["SC", "DBI", "CHI", "ARI_mean", "ARI_ci_low", "ARI_ci_high", "NMI_mean", "VDI"]},
                   vdi_profile=vdi_profile(y, model.refs), view_weights=dict(zip(vt.view, vt.weight.round(6))),
                   preprocessing=plog, seconds_total=time.perf_counter() - t0,
                   input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(), python=platform.python_version(),
                   numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, config=cfg.as_dict())
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()