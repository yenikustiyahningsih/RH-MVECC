"""Reviewer 2 comments 3 and 5, Reviewer 3 comments 3 and 4.

  simple_ablation.csv        simpler ablation baselines (view weighting, families, partition weighting)
  selection_ablation.csv     K-selection rules with/without View-Dominance Control
  alpha_beta_sensitivity.csv Eq. (4): (1-alpha) + alpha*SC, exponent beta, adaptive (entropy-weight) variant
  eps_sensitivity.csv        Eq. (7): r_m = eps + (1-eps) q_m
  T_effect.csv               does the stability factor T_v change the view weights?
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

from rhmvecc.core import Config, RHMVECC, view_stage, minmax
from rhmvecc import msme


def summarize(name, m, ref, group):
    vt = m.vt
    row = dict(group=group, variant=name, selected_K=m.k_star,
               **{f"w_{v}": float(vt.loc[vt.view == v, "weight"].iloc[0]) for v in vt.view})
    if m.k_star is not None:
        r = m.selection[m.selection.K == m.k_star].iloc[0]
        row.update(SC=r.SC, DBI=r.DBI, CHI=r.CHI, ARI_stability=r.ARI_mean, VDI=r.VDI,
                   ARI_vs_RH=adjusted_rand_score(ref, m.labels_star))
    r6 = m.selection[m.selection.K == 6].iloc[0]
    row.update(SC_K6=r6.SC, DBI_K6=r6.DBI, CHI_K6=r6.CHI, VDI_K6=r6.VDI, ARI_K6_vs_RH=adjusted_rand_score(ref, m.labels_at(6)))
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--repeats", type=int, default=50)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _, _, x, _ = msme.load(a.input)
    dx = squareform(pdist(x))
    base_cfg = Config(repeats=a.repeats)
    vr = view_stage(x, msme.VIEWS, base_cfg)
    main_m = RHMVECC(base_cfg).run(x, msme.VIEWS, dist=dx, view_result=vr)
    ref = main_m.labels_star
    print("reference K", main_m.k_star, flush=True)

    def run(name, group, **kw):
        c = Config(repeats=a.repeats, **kw)
        m = RHMVECC(c).run(x, msme.VIEWS, dist=dx, view_result=vr)
        print(group, name, m.k_star, flush=True)
        return summarize(name, m, ref, group)

    rows = [summarize("RH-MVECC (proposed)", main_m, ref, "reference")]
    rows += [run("Equal view weights (w_v = 1/V)", "view weighting", view_weight_mode="equal"),
             run("Raw SC view weights (w_v ∝ SC_v)", "view weighting", view_weight_mode="sc"),
             run("Without granularity/entropy term", "view weighting", use_GH=False),
             run("Without stability term T_v", "view weighting", use_T=False),
             run("Full family only (K-Means + GMM)", "library", families=("Full",)),
             run("Full family, K-Means only", "library", families=("Full",), full_methods=("KMeans",)),
             run("Full + Weighted (no ViewSpecific)", "library", families=("Full", "Weighted")),
             run("Full + ViewSpecific (no Weighted)", "library", families=("Full", "ViewSpecific")),
             run("Weighted + ViewSpecific (no Full)", "library", families=("Weighted", "ViewSpecific")),
             run("No family balancing", "partition weighting", family_balance=False),
             run("Uniform partition weights within family (eps = 1)", "partition weighting", eps=1.0)]
    pd.DataFrame(rows).to_csv(out / "simple_ablation.csv", index=False)

    # selection-rule ablation on the reference consensus
    sel = main_m.selection.copy(); comps = main_m.components
    def pick(score, mask=None):
        s = pd.Series(score, index=sel.index)
        if mask is not None: s[~mask] = -np.inf
        return int(sel.loc[s.idxmax(), "K"])
    w = np.asarray(base_cfg.comp_w)
    w4 = w[:4] / w[:4].sum()
    rules = {"SC only": pick(sel.SC.values),
             "DBI only": pick(-sel.DBI.values),
             "CHI only": pick(sel.CHI.values),
             "SC + DBI + CHI (equal, no stability, no VDI)": pick(comps[:, :3].mean(1)),
             "Composite without VDI term and without threshold": pick(comps[:, :4] @ w4),
             "Composite with VDI term, without threshold": pick(comps @ w),
             "Composite without VDI term, with threshold tau = 0.85": pick(comps[:, :4] @ w4, (sel.VDI <= .85).values),
             "SC only, with threshold tau = 0.85": pick(sel.SC.values, (sel.VDI <= .85).values),
             "RH-MVECC rule (composite + threshold)": main_m.k_star}
    srows = []
    for name, k in rules.items():
        r = sel[sel.K == k].iloc[0]
        srows.append(dict(rule=name, selected_K=k, SC=r.SC, DBI=r.DBI, CHI=r.CHI, ARI_stability=r.ARI_mean, VDI=r.VDI))
    pd.DataFrame(srows).to_csv(out / "selection_ablation.csv", index=False)

    # alpha / beta / adaptive sensitivity (Eq. 4)
    rows = []
    for al in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rows.append(run(f"alpha = {al}", "alpha (beta = 1)", alpha=al))
    for be in [0.5, 2.0, 4.0]:
        rows.append(run(f"beta = {be}", "beta (alpha = 0.5)", beta=be))
    rows.append(run("Adaptive: entropy-weight exponents", "adaptive", view_weight_mode="adaptive"))
    pd.DataFrame(rows).to_csv(out / "alpha_beta_sensitivity.csv", index=False)

    rows = []
    for e in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        rows.append(run(f"eps = {e}", "eps", eps=e))
    pd.DataFrame(rows).to_csv(out / "eps_sensitivity.csv", index=False)

    # T_v effect: n_init used for the 10 stability runs; n_init = 1 exposes initialization sensitivity
    trows = []
    for nin in [20, 1]:
        c = Config(repeats=a.repeats, view_n_init_T=nin)
        vr2 = view_stage(x, msme.VIEWS, c)
        vt = vr2[0]
        for use_T in [True, False]:
            c2 = Config(repeats=a.repeats, view_n_init_T=nin, use_T=use_T)
            from rhmvecc.core import reliability, view_weights
            vt2 = vt.copy(); vt2["R"] = reliability(vt2, c2); vt2["weight"] = view_weights(vt2, c2)
            for _, r in vt2.iterrows():
                trows.append(dict(n_init_for_T=nin, use_T=use_T, view=r.view, T=r["T"], T_min=r.T_min, R=r.R, weight=r.weight))
        if nin == 1:
            m = RHMVECC(Config(repeats=a.repeats, view_n_init_T=1)).run(x, msme.VIEWS, dist=dx, view_result=vr2)
            s = summarize("T_v estimated with n_init = 1", m, ref, "T effect")
            pd.DataFrame([s]).to_csv(out / "T_effect_run.csv", index=False)
    pd.DataFrame(trows).to_csv(out / "T_effect.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
