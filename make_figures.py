"""Manuscript figures from the results folders (Fig. 2-4 updated, Fig. 5-7 new)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results"); F = R / "figures"; F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10})
BLUE, ORANGE, GREY, RED = "#1f77b4", "#ff7f0e", "#7f7f7f", "#c00000"


def fig2():
    s = pd.read_csv(R / "main/selection.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.62))
    ax.plot(s.K, s.Composite, "o-", color=BLUE, label="Composite score J(K)")
    ax.plot(s.K, s.ARI_mean, "s-", color=ORANGE, label="Stability ARI (50 runs)")
    ax.set(ylim=(0, 1.05), xlabel="Number of clusters (K)", ylabel="Score"); ax.grid(alpha=.3)
    ax2 = ax.twinx()
    ax2.plot(s.K, s.VDI, "^-.", color=BLUE, label="VDI")
    ax2.axhline(0.85, ls=":", color=BLUE, label="VDI threshold τ = 0.85")
    ax2.set(ylim=(0.65, 1.0), ylabel="View-Dominance Index")
    k = 6; ax.axvline(k, color=GREY, ls="--", lw=.8)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(F / "fig2_selection.png", dpi=250); plt.close(fig)


def fig3():
    a = pd.read_csv(R / "main/assignments.csv")
    cnt = a.Cluster.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.0, 3.51))
    ax.bar(cnt.index, cnt.values, color=BLUE)
    for i, v in zip(cnt.index, cnt.values):
        ax.text(i, v + 15, f"{v} ({v / cnt.sum() * 100:.1f}%)", ha="center", fontsize=8)
    ax.set(xlabel="Cluster", ylabel="Number of MSMEs", ylim=(0, cnt.max() * 1.15)); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(F / "fig3_cluster_sizes.png", dpi=250); plt.close(fig)


def fig4():
    p = pd.read_csv(R / "main/profiles_normalized.csv", index_col=0)
    labels = ["NIB/SKU", "Employees", "Production\ncapacity", "Revenue\n(log1p)", "Assets\n(log1p)", "Social\nmedia", "Marketplace", "Land\nownership"]
    M = p.iloc[:, :8].values
    fig, ax = plt.subplots(figsize=(8.0, 3.97))
    im = ax.imshow(M, vmin=0, vmax=1, aspect="auto", cmap="YlGnBu")
    for i in range(M.shape[0]):
        for j in range(8):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > .6 else "black")
    ax.set_xticks(range(8), labels, fontsize=8); ax.set_yticks(range(len(p)), [f"C{c}" for c in p.index])
    fig.colorbar(im, ax=ax, label="Normalized mean"); fig.tight_layout(); fig.savefig(F / "fig4_profiles.png", dpi=250); plt.close(fig)


def fig5():
    d = pd.read_csv(R / "vdi/information_decomposition.csv")
    keep = ["V1", "V2", "V3", "V4", "V1+V2", "V1+V3", "V1+V4", "V1+V2+V3", "V1+V2+V3+V4"]
    d = d.set_index("views").loc[keep]
    t = pd.read_csv(R / "vdi/surrogate_tree.csv"); t = t[t.max_depth == 6].set_index("views").loc[keep]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    x = np.arange(len(keep)); w = .38
    ax.bar(x - w / 2, d.share_of_H, w, color=BLUE, label="Share of H(P*) explained, I(P*;P_S)/H(P*)")
    ax.bar(x + w / 2, t.balanced_accuracy, w, color=ORANGE, label="Surrogate tree balanced accuracy (5-fold)")
    ax.set_xticks(x, keep, fontsize=8); ax.set_ylim(0, 1.05); ax.set_ylabel("Proportion"); ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(F / "fig5_vdi_decomposition.png", dpi=250); plt.close(fig)


def fig6():
    fs = sorted((R / "benchmarks").glob("bench_*.csv"))
    if not fs: return
    b = pd.concat([pd.read_csv(f) for f in fs])
    meths = ["Concatenated K-Means", "Multi-view spectral (avg. kNN graph)", "Unweighted co-association", "RH-MVECC (K = C)"]
    b = b[b.method.isin(meths) & ~b.dataset.str.contains("nuisance")]
    g = b.groupby(["dataset", "method"]).NMI.mean().unstack()[meths]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    x = np.arange(len(g)); w = .2
    cols = [GREY, "#9467bd", ORANGE, BLUE]
    for i, m in enumerate(meths):
        ax.bar(x + (i - 1.5) * w, g[m], w, label=m, color=cols[i])
    ax.set_xticks(x, g.index, fontsize=9); ax.set_ylabel("NMI (mean over seeds)"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(F / "fig6_benchmarks.png", dpi=250); plt.close(fig)


if __name__ == "__main__":
    for f in [fig2, fig3, fig4, fig5, fig6]:
        f()
    print("figures written")
