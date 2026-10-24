# RH-MVECC: code, seeds and results

Reliability-Weighted Heterogeneous Multi-View Ensemble Consensus Clustering with View-Dominance Control, applied to MSME segmentation and checked on public datasets (seven labelled multi-view benchmarks and the Customer Personality Analysis marketing data). This repository contains the dataset-agnostic library (`rhmvecc/`), one script per experiment, all random seeds, and the raw outputs of every experiment reported in the revised manuscript submitted to *Array*.

## Requirements

Python 3.10 or newer and the packages in `requirements.txt`. The results were reproduced with Python 3.11.9 (Windows; NumPy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1, SciPy 1.17.1) and with scikit-learn 1.8.0 (Linux); the numbers are identical.

On Windows, unpack and run the code from a short path such as `C:\rh\RH-MVECC`. Very long folder paths make scikit-learn fail to load (`WinError 206`).

## Quick check (about 2 minutes)

```bash
pip install -r requirements.txt
python run_msme.py --input data/data_prep_2896_reconstructed.xlsx --out results/main_rerun
```

Expected output in `results/main_rerun/summary.json`:

| Quantity | Value |
|---|---|
| Selected K | 6 |
| SC / DBI / CHI | 0.3885 / 1.0251 / 1233.99 |
| Stability ARI (50 runs) | 0.9900 (95% CI 0.9879–0.9921) |
| VDI | 0.7823 |
| View weights V1 / V2 / V3 / V4 | 0.1798 / 0.3031 / 0.2719 / 0.2452 |
| input SHA-256 | d48faa8258de702967583a150e110c1b111a74d58d9b0b682984ca73465b8abc |

## Structure

| File | Purpose | Reviewer comment |
|---|---|---|
| `rhmvecc/core.py` | Library: views, reliability (Eq. 2–5), 45-partition library, partition weights (Eq. 6–7), weighted co-association (Eq. 8–9), VDI (Eq. 10), selection (Eq. 11), stability. Every design constant is a `Config` field. | all |
| `rhmvecc/msme.py` | MSME loader (`MinMax_2896` + `MinMax_Info` sheets, or CSV) | R2-4 |
| `run_msme.py` | Main experiment: Tables 4–14, Fig. 2–4, paired bootstrap vs K-Means, threshold and weight sensitivity | R2-3, R2-5 |
| `exp_vdi_diagnostics.py` | NMI profile, information decomposition, surrogate trees, reference partitions, nuisance-view injection | R2-2, R3-1 |
| `exp_nuisance_detail.py`, `exp_penguins_svs.py`, `exp_library_redundancy.py` | Nuisance K-scan (Table 17), Penguins VDI vs SVS, library pairwise ARI | R2-2, R2-3 |
| `exp_vdi_share.py` | Asymmetric single-view explained share SVS(K) (Eq. 12) | R2-2 |
| `exp_ablation_sensitivity.py` | Simpler ablation baselines, selection-rule ablation, alpha/beta/adaptive (Eq. 4), epsilon (Eq. 7), effect of T_v | R2-3, R2-5, R3-3, R3-4 |
| `exp_stability.py` | Library-fraction, seed-level re-runs, MSME-level subsampling | R2-5 |
| `exp_benchmarks.py` | Public multi-view benchmarks with ground truth (ACC/NMI/ARI/Purity) + nuisance-view stress test | R2-3, R3-2, R3-5 |
| `exp_marketing.py` | Second application on a public, unlabelled dataset (Customer Personality Analysis): same pipeline, baselines at the selected K, response-rate check | supervisor request; external validity (R2-3, R3-2) |
| `make_figures.py` | Manuscript figures | |

## Running all experiments

```bash
D=data/data_prep_2896_reconstructed.xlsx
python run_msme.py               --input $D --out results/main
python exp_vdi_diagnostics.py    --input $D --out results/vdi
python exp_vdi_share.py          $D
python exp_ablation_sensitivity.py --input $D --out results/ablation --repeats 20
python exp_stability.py          --input $D --out results/stability
bash benchmarks/download_benchmarks.sh   # downloads the .mat files (not stored in this repository)
python exp_benchmarks.py --data benchmarks --out results/benchmarks --only 3Sources,BBCSport,NGs,MSRC-v5,Penguins
python exp_benchmarks.py --data benchmarks --out results/benchmarks --only Handwritten --seeds 3
python exp_benchmarks.py --data benchmarks --out results/benchmarks --only Caltech101-7 --seeds 3
python exp_penguins_svs.py
python exp_nuisance_detail.py $D
python exp_library_redundancy.py
python exp_marketing.py          --input data/marketing_campaign_rapi.xlsx --out results/marketing
python make_figures.py
```

Approximate run time on a 2-vCPU machine: main 2.5 min; diagnostics about 1 h; ablation/sensitivity about 1.5 h; stability about 1 h; benchmarks about 1.5 h; marketing about 2 min. On Windows run the `bash` and `$D` lines in Git Bash.

## Public datasets used and what they are for

| Dataset | Type | n | Views | Classes | Used for |
|---|---|---|---|---|---|
| 3Sources | text (news) | 169 | 3 | 6 | labelled benchmark (Tables 22–23) |
| BBCSport | text (news) | 544 | 2 | 5 | labelled benchmark |
| NGs (20 Newsgroups subset) | text (news) | 500 | 3 | 5 | labelled benchmark (traditional text benchmark requested by R3-2) |
| MSRC-v5 | image | 210 | 5 | 7 | labelled benchmark |
| Handwritten | image (digits) | 2,000 | 6 | 10 | labelled benchmark |
| Caltech101-7 | image (objects) | 1,474 | 6 | 7 | labelled benchmark |
| Palmer Penguins | mixed-type tabular | 333 | 3 | 3 | labelled benchmark with a low-cardinality binary view, like the MSME data |
| Customer Personality Analysis (marketing) | mixed-type tabular | 2,240 | 4 | none | second real-world application, unlabelled (internal indices, stability, VDI) |

The labelled benchmarks use ground truth only for evaluation (ACC, NMI, ARI, purity) and answer the external-validity comments of the reviewers (`exp_benchmarks.py`). The marketing data is a second real application of the same pipeline (`exp_marketing.py`).

## Second application: Customer Personality Analysis (public marketing data)

To check that the framework is not specific to the MSME data, RH-MVECC was applied without modification to the public Customer Personality Analysis dataset (Kaggle, `marketing_campaign`): 2,240 customers, 25 features, four views (V1 profile, V2 product spending, V3 channels and engagement, V4 campaign-response history). The `Response` column is not used as a feature. No ground-truth labels exist, so only internal indices, stability and VDI are reported (no ACC/NMI).

```bash
python exp_marketing.py --input data/marketing_campaign_rapi.xlsx --out results/marketing_rerun
```

Expected output (input SHA-256 `6239fbaeb46f5205fba13640cc219ca9b8238206ea7d3d36a53c63d411047bed`):

| Quantity | Value |
|---|---|
| Selected K | 2 (clusters of 1,168 and 1,072) |
| SC / DBI / CHI | 0.2179 / 1.7534 / 690.93 |
| Stability ARI (95% CI) | 0.9495 (0.9246–0.9743) |
| VDI (dominant view) | 0.7039 (V2) |
| View weights V1 / V2 / V3 / V4 | 0.291 / 0.298 / 0.261 / 0.149 |

The campaign-response rate differs between the two clusters (19.8%, 212/1,072, versus 10.5%, 122/1,168; chi-square p < 0.001) although `Response` was not used for clustering.

Cluster profiles on the original scale (medians, `results/marketing/profiles_original_median.csv`):

| Median | Cluster 1 (n = 1,168) | Cluster 2 (n = 1,072) |
|---|---|---|
| Age | 42 | 47 |
| Income | 36,292 | 69,073.5 |
| Children / teenagers at home | 1 / 1 | 0 / 0 |
| Spending on wines | 25 | 496 |
| Spending on meat products | 16 | 241.5 |
| Catalog / store purchases | 0 / 3 | 4 / 8 |
| Web visits per month | 7 | 4 |
| Response rate to last campaign (post hoc) | 10.5% (122/1,168) | 19.8% (212/1,072) |

Cluster 2 is the higher-income, higher-spending, child-free group that buys through catalog and store channels; Cluster 1 is the lower-spending group with children at home and more web visits. Preprocessing (see `summary.json`): 24 missing `Income` values imputed with the median, continuous features winsorised, reference date 2014-06-29 for age and tenure.

## Comparison: real MSME data versus public datasets

Reviewers asked for comparison datasets. RH-MVECC was therefore run, unchanged, on (i) seven public labelled benchmarks (Tables 22–23 of the manuscript) and (ii) the public, unlabelled marketing data.

### Which dataset gives the better result?

**The real MSME data.** Its clusters are better separated and more stable than those of the public marketing data on 5 of the 6 measures; only VDI is lower on the marketing data.

| Measure | MSME (real) | Marketing (public) | Better |
|---|---|---|---|
| SC (higher is better) | 0.3885 | 0.2179 | MSME |
| DBI (lower is better) | 1.0251 | 1.7534 | MSME |
| CHI (higher is better) | 1233.99 | 690.93 | MSME |
| Stability, ARI (50 re-runs) | 0.9900 | 0.9495 | MSME |
| Stability, NMI (50 re-runs) | 0.9817 | 0.9228 | MSME |
| VDI (lower is better) | 0.7823 | 0.7039 | Marketing |

Reasons:

1. **Better separation.** By the common rule of thumb for the silhouette coefficient, SC below about 0.25 means no substantial cluster structure and 0.26-0.50 a weak-to-moderate one: marketing (0.2179) falls in the first group, MSME (0.3885) in the second. DBI is also much lower on MSME (1.03 versus 1.75).
2. **Higher stability.** When the base-partition library is resampled 50 times, the MSME result hardly changes (ARI 0.99), whereas the marketing result moves slightly more (ARI 0.95).
3. **Likely cause: the nature of the data (not tested separately).** The MSME attributes are mostly binary, ordinal or discrete, so the data form a limited number of patterns (view V1 has only 4 unique patterns) and distinct groups. The marketing attributes are continuous (about 2,030 unique patterns per view), so customers blend into each other and separate into only K = 2 groups.
4. **VDI is the exception.** It is lower on marketing (0.7039 versus 0.7823), but the selected K differs (2 versus 6) and VDI depends on K, so this difference is not a like-for-like comparison.

Caveats: the two datasets differ in size, feature types and selected K, so these numbers show which dataset has the clearer cluster structure, not which method is better; CHI in particular depends on n and K and should be read as supporting evidence only. On the method question, RH-MVECC is on par with K-Means on both datasets (see below). The MSME matrix used is the reconstructed file described under Data.

Side-by-side details:

| | MSME (real, n = 2,896, 4 views) | Marketing (public, n = 2,240, 4 views) |
|---|---|---|
| Selected K | 6 | 2 |
| SC / DBI / CHI | 0.3885 / 1.0251 / 1233.99 | 0.2179 / 1.7534 / 690.93 |
| Stability ARI (50 runs) | 0.9900 | 0.9495 |
| VDI (threshold 0.85) | 0.7823 | 0.7039 (dominant view V2) |
| Smallest view weight | V1, 0.1798 (4 unique patterns) | V4, 0.149 (23 unique patterns) |

Same-K comparison (SC / DBI / CHI / VDI; MSME at K = 6, marketing at K = 2):

| Method | MSME | Marketing |
|---|---|---|
| K-Means | 0.3850 / 1.0373 / 1265.95 / 0.8013 | 0.2181 / 1.7520 / 692.1 / 0.6783 |
| GMM | 0.2934 / 2.5695 / 901.96 / 0.8633 | 0.1938 / 2.4684 / 208.5 / 0.5768 |
| Ward | 0.3788 / 1.0460 / 1231.53 / 0.8006 | 0.1805 / 1.9641 / 542.3 / 0.4338 |
| Unweighted consensus | 0.3885 / 1.0217 / 1261.16 / 0.8053 | 0.1914 / 1.8120 / 608.2 / 0.5055 |
| Quality-weighted consensus | 0.3876 / 1.0281 / 1263.33 / 0.8038 | 0.2165 / 1.7561 / 687.2 / 0.6727 |
| **RH-MVECC** | 0.3885 / 1.0251 / 1233.99 / 0.7823 | 0.2179 / 1.7534 / 690.93 / 0.7039 |

On marketing, RH-MVECC with equal view weights gives 0.2172 / 1.7577 / 687.2 / 0.6591.

Public labelled benchmarks (Table 23, K = C): RH-MVECC improves NMI over concatenated K-Means on 2 of 7 datasets (3Sources, Caltech101-7; Penguins is a tie), with a large ACC gain on Caltech101-7 (0.732 versus 0.518). It is never the single best method in NMI and is clearly behind concatenated K-Means on Handwritten (NMI 0.812 versus 0.860) and NGs (NMI 0.883 versus 0.919).

### What the comparison shows

1. **Internal indices:** on both datasets RH-MVECC is practically tied with K-Means (MSME: SC +0.0035, DBI -0.0119, CHI -25.7; marketing: all three marginally worse). It is not a more accurate clusterer, and the manuscript does not claim that.
2. **Against weaker baselines:** on marketing it is clearly better than unweighted consensus, GMM and Ward on SC/DBI/CHI. On MSME, unweighted consensus is as good on SC, so the gain from weighting is data dependent.
3. **Stability:** high on both datasets (ARI 0.99 and 0.95).
4. **View dominance (VDI, lower is better):** the lowest VDI of all methods on MSME, but not on marketing, where RH-MVECC (0.7039) has a higher VDI than all six baselines at K = 2 (0.43-0.68); its partition is also close to K-Means (ARI 0.954). The VDI advantage is therefore not shown to generalise. VDI stays below the threshold of 0.85 on both datasets, so the diagnostic works as intended (no view is flagged as the de facto partition), but the supported claim is that VDI diagnoses dominance, not that RH-MVECC always reduces it.
5. **Granularity regularization behaves consistently:** the view with the fewest unique patterns but near-perfect separability (V1 in MSME, V4 in marketing) receives the smallest weight in both datasets.
6. **Labelled benchmarks:** competitive but not uniformly superior (see above).

Overall reading: RH-MVECC is supported as a stable, diagnosable consensus method with sensible weighting behaviour, not as a method that beats K-Means on quality indices.

## Public labelled benchmarks: re-run results

All seven labelled benchmarks were re-run with the commands in "Running all experiments" (Python 3.11.9, NumPy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1, SciPy 1.17.1, Windows; the per-seed ACC values were identical to an independent run with scikit-learn 1.8.0 on Linux). Seeds: 0-4 for 3Sources, BBCSport, NGs, MSRC-v5 and Penguins; 0-2 for Handwritten and Caltech101-7. Values are means over seeds on the **original views** (the first block of each `bench_*.csv`; the nuisance-view block is excluded). All methods use K = C (true number of classes), except where noted. Per-seed raw output: `results/benchmarks/bench_*.csv`; means: `results/benchmarks/summary_mean.csv`.

**NMI (higher is better)**

| Dataset | Concat. K-Means | MV spectral | Unweighted consensus | Quality-weighted consensus | **RH-MVECC (K = C)** | Best method |
|---|---|---|---|---|---|---|
| 3Sources | 0.565 | 0.567 | 0.616 | 0.603 | 0.602 | Unweighted consensus |
| BBCSport | 0.778 | 0.584 | 0.815 | 0.785 | 0.752 | Unweighted consensus |
| NGs | 0.919 | 0.843 | 0.881 | 0.850 | 0.883 | Concat. K-Means |
| MSRC-v5 | 0.669 | 0.704 | 0.691 | 0.643 | 0.649 | MV spectral |
| Handwritten | 0.860 | 0.843 | 0.851 | 0.779 | 0.812 | Concat. K-Means |
| Caltech101-7 | 0.453 | 0.569 | 0.504 | 0.550 | 0.512 | MV spectral |
| Penguins | 0.502 | 0.452 | 0.502 | 0.502 | 0.502 | tie (4 methods) |

**ACC (higher is better)**

| Dataset | Concat. K-Means | MV spectral | Unweighted consensus | Quality-weighted consensus | **RH-MVECC (K = C)** | Best method |
|---|---|---|---|---|---|---|
| 3Sources | 0.553 | 0.580 | 0.609 | 0.576 | 0.607 | Unweighted consensus |
| BBCSport | 0.837 | 0.548 | 0.878 | 0.858 | 0.837 | Unweighted consensus |
| NGs | 0.973 | 0.940 | 0.958 | 0.882 | 0.959 | Concat. K-Means |
| MSRC-v5 | 0.730 | 0.762 | 0.762 | 0.718 | 0.724 | MV spectral, Unweighted consensus |
| Handwritten | 0.922 | 0.839 | 0.767 | 0.627 | 0.674 | Concat. K-Means |
| Caltech101-7 | 0.518 | 0.592 | 0.557 | 0.751 | 0.732 | Quality-weighted consensus |
| Penguins | 0.703 | 0.532 | 0.703 | 0.703 | 0.703 | tie (4 methods) |

ARI and Purity are in `summary_mean.csv`. "Best method" ignores the single-view oracle and the automatic-K / equal-weight variants of RH-MVECC. On Penguins four of the compared methods shown (five with the equal-weight RH-MVECC variant) return the identical partition (tie); only the single-view oracle is higher (NMI 0.741).

### How RH-MVECC compares with the public datasets

1. **Versus concatenated K-Means (NMI):** RH-MVECC is higher on 3Sources (0.602 vs 0.565) and Caltech101-7 (0.512 vs 0.453), tied on Penguins (0.502), and lower on BBCSport (0.752 vs 0.778), NGs (0.883 vs 0.919), MSRC-v5 (0.649 vs 0.669) and Handwritten (0.812 vs 0.860). This is 2 wins, 1 tie, 4 losses.
2. **ACC:** the largest gain is on Caltech101-7 (0.732 vs 0.518). On Handwritten RH-MVECC is clearly behind (ACC 0.674 vs 0.922); on NGs it is slightly behind (0.959 vs 0.973).
3. **Never the single best method in NMI.** The best NMI comes from unweighted consensus (3Sources, BBCSport), multi-view spectral (MSRC-v5, Caltech101-7) or concatenated K-Means (Handwritten, NGs).
4. **Weighting does not consistently help on these benchmarks.** Unweighted consensus is equal to or better than RH-MVECC in NMI on 3Sources, BBCSport, MSRC-v5 and Handwritten, and equal on NGs (0.881 vs 0.883); the view weighting helps mainly on Caltech101-7 (quality-weighted consensus is similar: NMI 0.550, ACC 0.751). On NGs the reliability weights do help compared with equal view weights (NMI 0.883 vs 0.819, ACC 0.959 vs 0.805), but not compared with unweighted consensus.
5. **Automatic K is unreliable on labelled data:** with K chosen by the selection rule, ACC drops (e.g. MSRC-v5 0.400, Handwritten 0.488; on NGs it selects K = 2 for C = 5 classes in all seeds, ACC 0.389), because the rule optimises internal indices, not agreement with class labels.
6. **Penguins** is uninformative for comparing methods (all consensus variants return the same partition as K-Means, ACC 0.703).

Overall reading, consistent with the MSME and marketing results: RH-MVECC is competitive on public benchmarks but not uniformly superior, so the supported claim is stable, diagnosable consensus clustering, not higher accuracy than K-Means.

### Three-dataset-type summary (for the comparison in the paper)

| | MSME (real, unlabelled) | Marketing (public, unlabelled) | Seven public labelled benchmarks |
|---|---|---|---|
| Evaluation | SC/DBI/CHI, stability, VDI | SC/DBI/CHI, stability, VDI | ACC/NMI/ARI/Purity vs ground truth |
| RH-MVECC vs K-Means | tied | tied | NMI: 2 wins, 1 tie, 4 losses |
| Strength shown | stability (ARI 0.99), lowest VDI | stability (ARI 0.95) | competitive, large ACC gain on Caltech101-7 |
| Limitation | VDI advantage not general | higher VDI than baselines at K = 2 | behind K-Means on Handwritten and NGs |

## Ablation, sensitivity, stability and VDI diagnostics: re-run results

Re-run on the reconstructed MSME matrix (input SHA-256 `d48faa82...`), Python 3.11.9, scikit-learn 1.9.1, with `exp_ablation_sensitivity.py --repeats 20`, `exp_vdi_diagnostics.py` and `exp_stability.py`. Raw outputs: `results/ablation/`, `results/vdi/`, `results/stability/`. P* denotes the RH-MVECC partition at K = 6.

### Simpler ablation baselines (R2-3)

All variants select K = 6. Differences in SC (0.380-0.390), DBI (1.019-1.049) and CHI (1230-1266) are small, and RH-MVECC is not the single best on any of them.

| Group | Variant | K | SC | DBI | CHI | VDI | ARI vs RH-MVECC |
|---|---|---|---|---|---|---|---|
| reference | RH-MVECC (proposed) | 6 | 0.3885 | 1.0251 | 1234.0 | 0.7823 | 1.0000 |
| view weighting | Equal view weights (w_v = 1/V) | 6 | 0.3850 | 1.0373 | 1266.0 | 0.8013 | 0.9746 |
| view weighting | Raw SC view weights (w_v ∝ SC_v) | 6 | 0.3855 | 1.0240 | 1256.0 | 0.8047 | 0.9841 |
| view weighting | Without granularity/entropy term | 6 | 0.3799 | 1.0487 | 1264.5 | 0.7988 | 0.9547 |
| view weighting | Without stability term T_v | 6 | 0.3885 | 1.0251 | 1234.0 | 0.7823 | 1.0000 |
| library | Full family only (K-Means + GMM) | 6 | 0.3850 | 1.0364 | 1265.9 | 0.8015 | 0.9737 |
| library | Full family, K-Means only | 6 | 0.3850 | 1.0373 | 1266.0 | 0.8013 | 0.9746 |
| library | Full + Weighted (no ViewSpecific) | 6 | 0.3887 | 1.0234 | 1262.0 | 0.8050 | 0.9834 |
| library | Full + ViewSpecific (no Weighted) | 6 | 0.3875 | 1.0293 | 1263.6 | 0.8035 | 0.9910 |
| library | Weighted + ViewSpecific (no Full) | 6 | 0.3898 | 1.0189 | 1230.5 | 0.7805 | 0.9910 |
| partition weighting | No family balancing | 6 | 0.3876 | 1.0283 | 1263.3 | 0.8037 | 0.9910 |
| partition weighting | Uniform partition weights within family (eps = 1) | 6 | 0.3889 | 1.0211 | 1259.4 | 0.8021 | 0.9827 |

Reading: no ablated variant is clearly worse than the full method, and the full method is not clearly better. The lowest VDI of RH-MVECC (0.7823) is matched by "Weighted + ViewSpecific" (0.7805), and the other variants lie at 0.799-0.805. "Without T_v" is identical to the full method (see T_v below).

### Selection-rule ablation (R2-3)

| Selection rule | K | SC | DBI | CHI | Stability ARI | VDI |
|---|---|---|---|---|---|---|
| SC only | 3 | 0.4102 | 1.0623 | 1410.9 | 0.9409 | 0.9347 |
| DBI only | 8 | 0.3854 | 1.0034 | 936.7 | 0.9437 | 0.7956 |
| CHI only | 2 | 0.3574 | 1.2492 | 1513.6 | 0.8861 | 0.7736 |
| SC + DBI + CHI (equal, no stability, no VDI) | 3 | 0.4102 | 1.0623 | 1410.9 | 0.9409 | 0.9347 |
| Composite without VDI term and without threshold | 6 | 0.3885 | 1.0251 | 1234.0 | 0.9906 | 0.7823 |
| Composite with VDI term, without threshold | 6 | 0.3885 | 1.0251 | 1234.0 | 0.9906 | 0.7823 |
| Composite without VDI term, with threshold tau = 0.85 | 6 | 0.3885 | 1.0251 | 1234.0 | 0.9906 | 0.7823 |
| SC only, with threshold tau = 0.85 | 6 | 0.3885 | 1.0251 | 1234.0 | 0.9906 | 0.7823 |
| RH-MVECC rule (composite + threshold) | 6 | 0.3885 | 1.0251 | 1234.0 | 0.9906 | 0.7823 |

Reading: single-index rules choose different K (SC only: 3; DBI only: 8; CHI only: 2) with lower stability (ARI 0.89-0.94). SC only selects K = 3, which has VDI 0.9347 > tau = 0.85 and is excluded by the threshold. The composite score already selects K = 6 without the VDI term and without the threshold, so the threshold changes the outcome only for the SC-only rule.

### Sensitivity to alpha, beta (Eq. 4), the adaptive alternative, and epsilon (Eq. 7) (R3-3, R3-4)

| Group | Variant | K | w_V1 | w_V4 | SC | DBI | VDI | ARI vs RH-MVECC |
|---|---|---|---|---|---|---|---|---|
| alpha (beta = 1) | alpha = 0.0 | 6 | 0.1569 | 0.2161 | 0.3891 | 1.0234 | 0.7822 | 0.9932 |
| alpha (beta = 1) | alpha = 0.25 | 6 | 0.1676 | 0.2296 | 0.3871 | 1.0279 | 0.7804 | 0.9858 |
| alpha (beta = 1) | alpha = 0.5 | 6 | 0.1798 | 0.2452 | 0.3885 | 1.0251 | 0.7823 | 1.0000 |
| alpha (beta = 1) | alpha = 0.75 | 6 | 0.1939 | 0.2632 | 0.3855 | 1.0352 | 0.8018 | 0.9759 |
| alpha (beta = 1) | alpha = 1.0 | 6 | 0.2105 | 0.2842 | 0.3895 | 1.0212 | 0.8061 | 0.9767 |
| beta (alpha = 0.5) | beta = 0.5 | 6 | 0.1682 | 0.2305 | 0.3884 | 1.0258 | 0.7816 | 0.9958 |
| beta (alpha = 0.5) | beta = 2.0 | 6 | 0.2036 | 0.2750 | 0.3876 | 1.0275 | 0.8039 | 0.9901 |
| beta (alpha = 0.5) | beta = 4.0 | 6 | 0.2522 | 0.3340 | 0.3839 | 1.0186 | 0.8057 | 0.9780 |
| adaptive | Adaptive: entropy-weight exponents | 4 | 0.0691 | 0.1583 | 0.3668 | 1.1928 | 0.6544 | 0.8996 |
| eps | eps = 0.0 | 6 | 0.1798 | 0.2452 | 0.3859 | 1.0342 | 0.8020 | 0.9783 |
| eps | eps = 0.05 | 6 | 0.1798 | 0.2452 | 0.3886 | 1.0237 | 0.7825 | 0.9985 |
| eps | eps = 0.1 | 6 | 0.1798 | 0.2452 | 0.3885 | 1.0251 | 0.7823 | 1.0000 |
| eps | eps = 0.2 | 6 | 0.1798 | 0.2452 | 0.3885 | 1.0233 | 0.7824 | 0.9961 |
| eps | eps = 0.3 | 6 | 0.1798 | 0.2452 | 0.3882 | 1.0222 | 0.7826 | 0.9952 |
| eps | eps = 0.5 | 6 | 0.1798 | 0.2452 | 0.3890 | 1.0207 | 0.7799 | 0.9946 |
| eps | eps = 1.0 | 6 | 0.1798 | 0.2452 | 0.3889 | 1.0211 | 0.8021 | 0.9827 |

Reading:
* **alpha, beta:** K = 6 for every setting; partitions stay within ARI 0.976-0.996 of the default. Alpha changes the weight of V1 from 0.157 (alpha = 0) to 0.211 (alpha = 1); beta = 4 raises w_V4 to 0.334. The values alpha = 0.5 and beta = 1 are therefore heuristic defaults to which the result is not sensitive, not optimal values.
* **Adaptive exponents (entropy-weight):** this alternative selects K = 4, gives ARI 0.8996 to the default, and is worse on SC (0.3668) and DBI (1.1928) at its own K. At K = 6 it is comparable (SC 0.3902, ARI 0.975 to the default). It is not shown to be better than fixed values, and it shrinks the weight of V1 to 0.069.
* **epsilon:** it only affects partition weights, so the view weights are identical. K = 6 for all values and ARI to the default is 0.978-0.9985. The value epsilon = 0.1 is a heuristic choice; there is no mathematical derivation, and the sensitivity result is the justification.

### Effect of the stability factor T_v on the view weights (R2-5)

| n_init for T_v | T_v in weights | View | T_v | R_v | Weight |
|---|---|---|---|---|---|
| 20 | True | V1 | 1.0000 | 0.3587 | 0.1798 |
| 20 | True | V2 | 1.0000 | 0.6048 | 0.3031 |
| 20 | True | V3 | 1.0000 | 0.5424 | 0.2719 |
| 20 | True | V4 | 1.0000 | 0.4891 | 0.2452 |
| 20 | False | V1 | 1.0000 | 0.3587 | 0.1798 |
| 20 | False | V2 | 1.0000 | 0.6048 | 0.3031 |
| 20 | False | V3 | 1.0000 | 0.5424 | 0.2719 |
| 20 | False | V4 | 1.0000 | 0.4891 | 0.2452 |
| 1 | True | V1 | 1.0000 | 0.3587 | 0.2005 |
| 1 | True | V2 | 0.8996 | 0.5441 | 0.3042 |
| 1 | True | V3 | 0.7316 | 0.3968 | 0.2219 |
| 1 | True | V4 | 1.0000 | 0.4891 | 0.2735 |
| 1 | False | V1 | 1.0000 | 0.3587 | 0.1798 |
| 1 | False | V2 | 0.8996 | 0.6048 | 0.3031 |
| 1 | False | V3 | 0.7316 | 0.5424 | 0.2719 |
| 1 | False | V4 | 1.0000 | 0.4891 | 0.2452 |

Reading: with the default 20 K-Means initialisations T_v = 1 for all four views, so including or excluding T_v gives identical weights; **in the main setting the stability factor has no effect on the weights.** With a single initialisation T_V2 = 0.900 and T_V3 = 0.732, and including T_v lowers the weight of V3 (0.2219 versus 0.2719); the partition changes little (ARI 0.990 to the default, K = 6).

### Stability procedures (R2-5)

Library fraction (50 draws per fraction, seed 123):

| Library fraction | Partitions | Selected K | ARI mean (95% CI) | ARI min | NMI mean |
|---|---|---|---|---|---|
| 0.6 | 27 | 6 | 0.9850 (0.9833-0.9867) | 0.9699 | 0.9736 |
| 0.7 | 31 | 6 | 0.9871 (0.9856-0.9886) | 0.9758 | 0.9766 |
| 0.8 | 36 | 6 | 0.9900 (0.9879-0.9921) | 0.9736 | 0.9817 |
| 0.9 | 40 | 6 | 0.9926 (0.9907-0.9946) | 0.9758 | 0.9866 |

Seed re-runs (seeds 0-9) and 80% MSME subsamples (10 subsets, `default_rng(555)`), ARI against the full-data reference:

| Procedure | Runs | K=6 in | ARI vs reference mean (SD) | 95% CI | Min |
|---|---|---|---|---|---|
| B seed re-runs | 10 | 100% | 0.9888 (0.0078) | 0.9833-0.9944 | 0.9744 |
| C 80% MSME subsamples | 10 | 100% | 0.9866 (0.0103) | 0.9792-0.9940 | 0.9680 |

Across seed re-runs, the view weights are identical (SD = 0) and T_v = 1 for every view. Across subsamples the weights vary by at most about 0.004 (e.g. w_V1 0.1798-0.1826).

### VDI diagnostics (R2-1, R2-2, R3-1)

NMI between each view's partition and the consensus partition, by K (dominant view is V1 at every K):

| K | VDI | Dominant view | NMI V1 | NMI V2 | NMI V3 | NMI V4 | Eligible (VDI<=0.85) |
|---|---|---|---|---|---|---|---|
| 2 | 0.7736 | V1 | 0.7736 | 0.0131 | 0.0030 | 0.0032 | True |
| 3 | 0.9347 | V1 | 0.9347 | 0.0237 | 0.0093 | 0.0064 | False |
| 4 | 0.8281 | V1 | 0.8281 | 0.2702 | 0.0088 | 0.0416 | True |
| 5 | 0.7502 | V1 | 0.7502 | 0.4807 | 0.0148 | 0.0525 | True |
| 6 | 0.7823 | V1 | 0.7823 | 0.4655 | 0.0145 | 0.0533 | True |
| 7 | 0.7812 | V1 | 0.7812 | 0.4668 | 0.0146 | 0.0539 | True |
| 8 | 0.7956 | V1 | 0.7956 | 0.4600 | 0.0144 | 0.0549 | True |
| 9 | 0.7317 | V1 | 0.7317 | 0.4165 | 0.0135 | 0.1735 | True |
| 10 | 0.7256 | V1 | 0.7256 | 0.4305 | 0.0136 | 0.1738 | True |

Information decomposition of the K = 6 partition (H(P*) = 2.19 bits):

| Views | Share of H(P*) | NMI with P* |
|---|---|---|
| V1 | 65.5% | 0.7823 |
| V2 | 31.6% | 0.4655 |
| V3 | 1.0% | 0.0145 |
| V4 | 6.0% | 0.0533 |
| V1+V2 | 96.5% | 0.9520 |
| V1+V2+V3+V4 | 97.7% | 0.5409 |

Surrogate decision tree predicting P* from view features:

| Views used by tree (depth 3) | Accuracy | Balanced accuracy |
|---|---|---|
| V1 | 0.7686 | 0.6667 |
| V2 | 0.4969 | 0.3316 |
| V3 | 0.4154 | 0.2546 |
| V4 | 0.3747 | 0.1786 |
| V1+V2 | 0.9938 | 0.9895 |
| V1+V2+V3+V4 | 0.9938 | 0.9895 |

Reference partitions and the VDI value they obtain:

| Partition | K | VDI | ARI vs P* |
|---|---|---|---|
| Pure V1 reference (NIB x Land) | 4 | 1.0000 | 0.6661 |
| V1 x V2 reference | 8 | 0.7936 | 0.9710 |
| V1 x V3 reference | 10 | 0.7707 | 0.4615 |
| V1 x V4 reference | 39 | 0.7900 | 0.1870 |
| V2 x V3 reference | 5 | 0.6954 | 0.2296 |
| V1 x V2 x V3 x V4 reference | 120 | 0.6505 | 0.1471 |
| Random partition, sizes of P* | 6 | 0.0044 | 0.0009 |
| RH-MVECC P* (K=6) | 6 | 0.7823 | 1.0000 |
| RH-MVECC consensus K=3 | 3 | 0.9347 | 0.6548 |
| RH-MVECC consensus K=4 | 4 | 0.8281 | 0.8099 |

Contingency of V1 (business registration NIB x land ownership) with the six clusters (`contingency_V1_by_cluster.csv`): clusters 1, 2, 4, 5 and 6 each lie entirely in a single V1 pattern, and cluster 3 is split between two patterns (24 and 336 MSMEs).

**Reading (this partly supports and partly limits the VDI claim):**
1. VDI = 0.7823 < 1: the partition is not a copy of V1 (the "pure V1" reference with K = 4 has VDI = 1.0 and ARI only 0.666 to P*). V1 alone explains 65.5% of the entropy of P*.
2. However, V1 is the dominant view at every K, and **P* is essentially V1 refined by V2**: V1 + V2 explain 96.5% of H(P*), a depth-3 tree on V1 + V2 recovers P* with 99.4% accuracy, and the "V1 x V2 reference" has ARI 0.971 to P*. This confirms the reviewer's observation (R2-2) that the final clusters still correspond strongly to registration status and land ownership.
3. VDI below tau therefore means "no single view reproduces the partition", not "no view has a large influence". V3 and V4 contribute almost nothing (NMI 0.015 and 0.053).
4. V1 has the smallest reliability weight (0.18) yet is the most influential view in the final partition: the view weights do not control the VDI-dominant view here. The partition is determined mainly by the way view-specific partitions combine in the consensus.

### Nuisance-view injection (R2-2)

A random binary nuisance view V5 is added to the MSME views (seeds 11-13, means over seeds):

| View weighting | Selection | Mean K | Weight of nuisance view | VDI (dominant = nuisance) | ARI vs original P* |
|---|---|---|---|---|---|
| reliability | with VDI control | 8 | 0.1729 | 0.8005 | 0.2023 |
| reliability | without VDI control | 8 | 0.1729 | 0.8005 | 0.2023 |
| sc | with VDI control | 8 | 0.2390 | 0.8005 | 0.2023 |
| sc | without VDI control | 8 | 0.2390 | 0.8005 | 0.2023 |
| equal | with VDI control | 8 | 0.2000 | 0.8005 | 0.2023 |
| equal | without VDI control | 8 | 0.2000 | 0.8005 | 0.2023 |
| concatenated K-Means | fixed K=6 | 6 | - | 0.7127 | 0.2363 |

Reading: this is a negative result for dominance control.
* With the nuisance view, K = 8 is selected in all seeds, the nuisance view becomes the dominant view (NMI 0.80), and the partition changes strongly (ARI to the original P* about 0.20).
* VDI = 0.80 is below tau = 0.85, so the threshold does not flag the nuisance view; "with VDI control" and "without VDI control" give identical results.
* Reliability, SC-based and equal view weights give the same partition and the same VDI. The nuisance view's weight is lower under reliability weighting (0.173 versus 0.200 and 0.239) but this does not change the outcome.
* Concatenated K-Means at K = 6 is affected in a similar way (ARI to original 0.18-0.27).
* VDI as a **diagnostic** does identify the nuisance view as the dominant one (reported as `V5_nuisance`), but the fixed threshold tau = 0.85 and the weighting do not prevent the dominance. The supported claim is that VDI reveals which view dominates; it is not shown to prevent dominance.

## Seeds and repeated-run procedures

* K-Means (`n_init = 20`) and GMM (full covariance, `n_init = 1`, `reg_covar = 1e-6`): `random_state = 42`.
* View stability T_v: 10 additional K-Means runs with seeds 0–9 at the selected within-view K; T_v = mean ARI against the seed-42 reference.
* Ensemble stability: `numpy.random.default_rng(123)`; 50 draws of 36 of the 45 partitions (80%) without replacement; weights renormalised; consensus rebuilt; ARI/NMI against the full-library consensus. Every draw (with the indices of dropped partitions) is saved in `results/main/stability_runs.csv`.
* Composite-weight perturbation: 10,000 draws from the same generator after the resampling draws (±20% and ±50%).
* Paired bootstrap RH-MVECC vs K-Means: `default_rng(2024)`, 200 subsamples of 80% MSMEs.
* Seed-level re-runs: master seed s = 0…9 (library seed s, view-stability seeds 10s…10s+9, resampling seed 1000+s).
* MSME subsampling: `default_rng(555)`, 10 subsets of 80% MSMEs, complete pipeline re-run.
* Benchmarks: seeds 0–4 (0–2 for Handwritten and Caltech101-7); nuisance view drawn with `default_rng(99)`; MSME nuisance test seeds 11–13.

## Data

`data/data_prep_2896_reconstructed.xlsx` holds the anonymised, min-max normalised MSME matrix (2,896 × 8, sequential identifiers only) in the sheets `MinMax_2896` and `MinMax_Info`. The SHA-256 of the input file is written to `summary.json`.

Provenance: the original workbook `data_prep_2896.xlsm` is not included in this repository. The file above was rebuilt from the `assignments` sheet of an earlier run (values rounded to 6 decimals), and its `MinMax_Info` sheet (original-scale minima and maxima) was back-solved from the reported original-scale profiles. Running the code on it reproduces the results reported in the manuscript (table above). The original-scale profiles (Table 14) depend on the back-solved parameters.

Marketing data: not stored in this repository. The public Kaggle dataset "Marketing Campaign" by Rodolfo Saldanha (`marketing_campaign.csv`, 2,240 rows, 29 columns; https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign; licence on Kaggle: "Other (specified in description)") must be downloaded from Kaggle by the user, who is responsible for respecting the terms stated on that page. The cleaned input used for the second application, `data/marketing_campaign_rapi.xlsx`, is derived from that file; the preprocessing steps are recorded in `results/marketing/summary.json` (median imputation of 24 missing `Income` values, winsorisation of continuous features, reference date 2014-06-29 for age and tenure).

Benchmarks (`benchmarks/`): 3Sources, BBCSport, MSRC-v5, Handwritten and Caltech101-7 (.mat files from https://github.com/ChuanbinZhang/Multi-view-datasets; checksums in `benchmarks/SHA256SUMS.txt`), NGs (the 20 Newsgroups subset with 500 documents, 3 views of 2,000 features and 5 balanced classes; file `20newsgroups.mat` from https://github.com/zskong/Multi-view-data, SHA-256 `99bd0802...`) and Palmer Penguins (https://github.com/mwaskom/seaborn-data). Any `.mat` file with a cell array `X` (views) and a label vector `y` (e.g. NGs) can be added to `exp_benchmarks.py` by extending the dataset list.

Note on `results/benchmarks/bench_*.csv`: for Handwritten, Caltech101-7 and Penguins each method appears twice per seed. The first block (in file order) is the run on the original views, which gives Table 23; the second block is the run with the random binary nuisance view added. Do not average the two blocks together.

## Differences from the first submission

* Numbers now come from one released code base; small differences from the first-submission tables (e.g. SC 0.3887→0.3885, VDI 0.7794→0.7823, V4 reference K 8→10, cluster sizes ±1–2) are due to the first submission having been produced with an unreleased script.
* Cluster numbers are ordered by size.