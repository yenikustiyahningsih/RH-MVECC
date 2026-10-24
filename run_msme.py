"""Main RH-MVECC experiment on the 2,896 MSME dataset (Tables 4-14 of the revised manuscript).

python run_msme.py --input data/data_prep_2896.xlsm --out results/main
"""
import argparse, json, time, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rhmvecc.core import Config, RHMVECC, consensus_cut, internal_metrics, vdi, kmeans, canonical
from rhmvecc import msme


def gower_average(x, k):
    active = np.ptp(x, axis=0) > 0
    gd = np.zeros(len(x) * (len(x) - 1) // 2)
    for j in np.flatnonzero(active):
        gd += pdist(x[:, j, None], metric="cityblock") / np.ptp(x[:, j])
    gd /= active.sum()
    return cut_tree(linkage(gd, method="average"), n_clusters=[k])[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--tau", type=float, default=0.85)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ids, data, x, info = msme.load(a.input)
    n = len(x)
    dx = squareform(pdist(x))
    T = {}
    T["audit"] = pd.DataFrame({"attribute": data.columns, "label": msme.FEATURES_EN, "mean": x.mean(0),
                               "sd": x.std(0, ddof=1), "unique": [len(np.unique(x[:, j])) for j in range(8)],
                               "missing": data.isna().sum().values})
    T["correlation"] = pd.DataFrame(np.corrcoef(x.T), index=data.columns, columns=data.columns)

    cfg = Config(tau=a.tau, repeats=a.repeats)
    t_rh = time.perf_counter()
    model = RHMVECC(cfg).fit(x, msme.VIEWS, dist=dx, verbose=True)
    t_fit = time.perf_counter() - t_rh
    model.stability(verbose=True).select()
    rng = model._rng   # continue the same seeded stream (seed 123) for the weight perturbation
    k = model.k_star
    y = model.labels_star
    vt = model.vt.copy(); vt.insert(1, "meaning", vt.view.map(msme.VIEW_NAMES))
    T["view_reliability"] = vt
    for v, s in model.scans.items():
        T["scan_" + v] = s
    T["view_reference_ARI"] = pd.DataFrame([[adjusted_rand_score(model.refs[u], model.refs[v]) for v in model.refs]
                                            for u in model.refs], index=list(model.refs), columns=list(model.refs))
    T["partitions"] = model.parts
    T["selection"] = model.selection
    T["stability_runs"] = model.stab_runs

    # K-Means baseline over K (Table 6)
    km_rows = []
    for kk in cfg.ks:
        i = model.parts.index[(model.parts.family == "Full") & (model.parts.method == "KMeans") & (model.parts.K == kk)][0]
        km_rows.append(dict(K=kk, **{c: model.parts.loc[i, c] for c in ["SC", "DBI", "CHI"]},
                            VDI=vdi(model.labels[i], model.refs)))
    T["kmeans_scan"] = pd.DataFrame(km_rows)

    # profiles
    final = canonical(y)
    assign = data.copy(); assign.insert(0, "UMKM_ID", ids); assign["Cluster"] = final
    T["assignments"] = assign
    prof = assign.groupby("Cluster")[list(data.columns)].mean()
    prof["n"] = assign.groupby("Cluster").size(); prof["share"] = prof["n"] / n
    T["profiles_normalized"] = prof
    if info is not None:
        orig = msme.to_original(data, x, info); orig["Cluster"] = final
        T["profiles_original_median"] = orig.groupby("Cluster").median()
        T["profiles_original_q25"] = orig.groupby("Cluster").quantile(.25)
        T["profiles_original_q75"] = orig.groupby("Cluster").quantile(.75)
        T["profiles_original_share"] = orig.groupby("Cluster")[["NIB_SKU", "Kepemilikan_Lahan"]].mean()

    # baselines at K* (Table 9) + runtime (Table 12)
    base, rt, base_labels = [], [], {}
    def timed(name, fn):
        s = time.perf_counter(); z = fn(); rt.append(dict(method=name, seconds=time.perf_counter() - s)); return z
    base_labels["K-Means"] = timed("K-Means", lambda: kmeans(x, k, 42, 20))
    base_labels["GMM"] = timed("GMM", lambda: GaussianMixture(k, random_state=42, n_init=1, covariance_type="full",
                                                              reg_covar=1e-6).fit_predict(x))
    base_labels["Ward hierarchical"] = timed("Ward hierarchical", lambda: cut_tree(linkage(pdist(x), "ward"), n_clusters=[k])[:, 0])
    base_labels["Gower-average"] = timed("Gower-average", lambda: gower_average(x, k))
    M = len(model.parts)
    base_labels["Unweighted consensus"] = consensus_cut(model.labels, np.ones(M), [k])[:, 0]
    base_labels["Quality-weighted consensus"] = consensus_cut(model.labels, 0.1 + 0.9 * model.parts.quality.values, [k])[:, 0]
    base_labels["RH-MVECC"] = timed("RH consensus step", lambda: consensus_cut(model.labels, model.parts.weight.values, [k])[:, 0])
    rt.append(dict(method="RH-MVECC view stage + library + consensus (all K)", seconds=t_fit))
    for name, z in base_labels.items():
        base.append(dict(method=name, K=k, **internal_metrics(x, z, dx), VDI=vdi(z, model.refs),
                         ARI_vs_RH=adjusted_rand_score(y, z)))
    T["baselines"] = pd.DataFrame(base)
    T["runtime"] = pd.DataFrame(rt)

    # paired bootstrap of internal-index differences RH-MVECC vs K-Means (fixed labels, resampled MSMEs)
    brng = np.random.default_rng(2024)
    bt = []
    zk = base_labels["K-Means"]
    for b in range(a.n_boot):
        ix = brng.choice(n, int(0.8 * n), replace=False)
        d_sub = dx[np.ix_(ix, ix)]
        mr = internal_metrics(x[ix], y[ix], d_sub); mk = internal_metrics(x[ix], zk[ix], d_sub)
        bt.append(dict(boot=b + 1, **{f"RH_{m}": mr[m] for m in mr}, **{f"KM_{m}": mk[m] for m in mk}))
    bt = pd.DataFrame(bt)
    rows = []
    for m in ["SC", "DBI", "CHI"]:
        d = bt[f"RH_{m}"] - bt[f"KM_{m}"]
        rows.append(dict(metric=m, mean_diff=d.mean(), ci_low=d.quantile(.025), ci_high=d.quantile(.975),
                         share_RH_better=float(((d > 0) if m != "DBI" else (d < 0)).mean())))
    T["bootstrap_vs_kmeans"] = pd.DataFrame(rows)
    T["bootstrap_vs_kmeans_runs"] = bt

    # family / view-partition ablation (Table 11)
    ab = []
    w = model.parts.weight.values
    masks = [(f"Remove {f} family", model.parts.family != f) for f in model.parts.family.unique()]
    masks += [(f"Remove {msme.VIEW_NAMES[v]} view partitions", model.parts["view"] != v) for v in msme.VIEWS]
    for label, mask in masks:
        z = consensus_cut(model.labels[mask.values], w[mask.values], [k])[:, 0]
        ab.append(dict(ablation=label, ARI=adjusted_rand_score(y, z), NMI=normalized_mutual_info_score(y, z),
                       **internal_metrics(x, z, dx), VDI=vdi(z, model.refs)))
    T["ablation_components"] = pd.DataFrame(ab)

    # VDI threshold sensitivity (Table 8) and composite-weight perturbation
    sel = model.selection
    thr = []
    for tau in [.70, .75, .78, .80, .85, .90, .95, 1.0]:
        e = sel[sel.VDI <= tau]
        kk = None if e.empty else int(e.loc[e.Composite.idxmax(), "K"])
        r = sel[sel.K == kk].iloc[0] if kk else None
        thr.append(dict(tau=tau, selected_K=kk, **({c: r[c] for c in ["Composite", "VDI", "SC", "DBI", "ARI_mean"]} if kk else {})))
    T["threshold_sensitivity"] = pd.DataFrame(thr)
    nominal = np.asarray(cfg.comp_w)
    pert = nominal * rng.uniform(.8, 1.2, (10000, 5)); pert /= pert.sum(1, keepdims=True)
    sc_ = model.components @ pert.T; sc_[~sel.Eligible.values, :] = -np.inf
    wins = np.asarray(cfg.ks)[sc_.argmax(0)]
    T["weight_sensitivity"] = pd.Series(wins).value_counts().rename_axis("K").reset_index(name="count")
    # wider perturbation (+-50%) as additional check
    pert = nominal * rng.uniform(.5, 1.5, (10000, 5)); pert /= pert.sum(1, keepdims=True)
    sc_ = model.components @ pert.T; sc_[~sel.Eligible.values, :] = -np.inf
    T["weight_sensitivity_50pct"] = pd.Series(np.asarray(cfg.ks)[sc_.argmax(0)]).value_counts().rename_axis("K").reset_index(name="count")

    # save
    for name, tab in T.items():
        tab.to_csv(out / f"{name}.csv", index=name.startswith(("profiles", "correlation", "view_reference")))
    with pd.ExcelWriter(out / "RH_MVECC_main_results.xlsx", engine="openpyxl") as wr:
        for name, tab in T.items():
            tab.to_excel(wr, sheet_name=name[:31], index=name.startswith(("profiles", "correlation", "view_reference")))
    np.save(out / "consensus_labels_allK.npy", model.consensus)
    np.save(out / "library_labels.npy", model.labels)

    # figures
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, lab in zip(axes, ["Composite", "ARI_mean", "VDI"], ["Composite score J(K)", "Stability ARI (50 runs)", "VDI"]):
        ax.plot(sel.K, sel[col], "o-", color="#1f4e79"); ax.set(xlabel="K", ylabel=lab); ax.axvline(k, color="gray", ls="--")
        ax.grid(alpha=.3)
    axes[2].axhline(cfg.tau, color="#c00000", ls=":", label=f"τ = {cfg.tau}"); axes[2].legend()
    fig.tight_layout(); fig.savefig(out / "fig_selection.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cnt = assign.Cluster.value_counts().sort_index(); ax.bar(cnt.index, cnt.values, color="#1f4e79")
    for i, v in zip(cnt.index, cnt.values): ax.text(i, v + 10, str(v), ha="center", fontsize=9)
    ax.set(xlabel="Cluster", ylabel="Number of MSMEs"); fig.tight_layout(); fig.savefig(out / "fig_cluster_sizes.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(prof[list(data.columns)], vmin=0, vmax=1, aspect="auto", cmap="YlGnBu")
    for i in range(prof.shape[0]):
        for j in range(8):
            ax.text(j, i, f"{prof.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(8), msme.FEATURES_EN, rotation=25, ha="right"); ax.set_yticks(range(k), range(1, k + 1))
    ax.set_ylabel("Cluster"); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / "fig_profiles.png", dpi=200); plt.close(fig)

    r = sel[sel.K == k].iloc[0]
    summary = dict(n=n, p=8, selected_K=k, metrics={c: float(r[c]) for c in ["SC", "DBI", "CHI", "ARI_mean", "ARI_ci_low", "ARI_ci_high", "NMI_mean", "VDI"]},
                   view_weights=dict(zip(vt.view, vt.weight.round(6))), seconds_total=time.perf_counter() - t0,
                   input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(), python=platform.python_version(),
                   numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
                   config=cfg.as_dict(),
                   seeds=dict(kmeans_gmm_random_state=cfg.seed, view_stability_seeds=list(cfg.view_seeds),
                              library_resampling_seed=cfg.resample_seed,
                              weight_perturbation="same numpy Generator(seed=123) stream, after the resampling draws",
                              bootstrap_vs_kmeans_seed=2024))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
