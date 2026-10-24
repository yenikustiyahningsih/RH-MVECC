# RH-MVECC: code, seeds and results

Reliability-Weighted Heterogeneous Multi-View Ensemble Consensus Clustering with View-Dominance Control, applied to MSME segmentation and checked on public datasets (six labelled multi-view benchmarks and the Customer Personality Analysis marketing data). This repository contains the dataset-agnostic library (`rhmvecc/`), one script per experiment, all random seeds, and the raw outputs of every experiment reported in the revised manuscript submitted to *Array*.

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
python exp_benchmarks.py --data benchmarks --out results/benchmarks --only 3Sources,BBCSport,MSRC-v5,Penguins
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

Reviewers asked for comparison datasets. RH-MVECC was therefore run, unchanged, on (i) six public labelled benchmarks (Tables 22–23 of the manuscript) and (ii) the public, unlabelled marketing data.

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

Public labelled benchmarks (Table 23, K = C): RH-MVECC improves NMI over concatenated K-Means on 2 of 6 datasets (3Sources, Caltech101-7; Penguins is a tie), with a large ACC gain on Caltech101-7 (0.732 versus 0.518). It is never the single best method in NMI and is clearly behind concatenated K-Means on Handwritten (NMI 0.812 versus 0.860).

### What the comparison shows

1. **Internal indices:** on both datasets RH-MVECC is practically tied with K-Means (MSME: SC +0.0035, DBI -0.0119, CHI -25.7; marketing: all three marginally worse). It is not a more accurate clusterer, and the manuscript does not claim that.
2. **Against weaker baselines:** on marketing it is clearly better than unweighted consensus, GMM and Ward on SC/DBI/CHI. On MSME, unweighted consensus is as good on SC, so the gain from weighting is data dependent.
3. **Stability:** high on both datasets (ARI 0.99 and 0.95).
4. **View dominance (VDI, lower is better):** the lowest VDI of all methods on MSME, but not on marketing, where RH-MVECC (0.7039) has a higher VDI than all six baselines at K = 2 (0.43-0.68); its partition is also close to K-Means (ARI 0.954). The VDI advantage is therefore not shown to generalise. VDI stays below the threshold of 0.85 on both datasets, so the diagnostic works as intended (no view is flagged as the de facto partition), but the supported claim is that VDI diagnoses dominance, not that RH-MVECC always reduces it.
5. **Granularity regularization behaves consistently:** the view with the fewest unique patterns but near-perfect separability (V1 in MSME, V4 in marketing) receives the smallest weight in both datasets.
6. **Labelled benchmarks:** competitive but not uniformly superior (see above).

Overall reading: RH-MVECC is supported as a stable, diagnosable consensus method with sensible weighting behaviour, not as a method that beats K-Means on quality indices.

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

Benchmarks (`benchmarks/`): 3Sources, BBCSport, MSRC-v5, Handwritten and Caltech101-7 (.mat files from https://github.com/ChuanbinZhang/Multi-view-datasets; checksums in `benchmarks/SHA256SUMS.txt`) and Palmer Penguins (https://github.com/mwaskom/seaborn-data). Any `.mat` file with a cell array `X` (views) and a label vector `y` (e.g. NGs) can be added to `exp_benchmarks.py` by extending the dataset list.

Note on `results/benchmarks/bench_*.csv`: for Handwritten, Caltech101-7 and Penguins each method appears twice per seed. The first block (in file order) is the run on the original views, which gives Table 23; the second block is the run with the random binary nuisance view added. Do not average the two blocks together.

## Differences from the first submission

* Numbers now come from one released code base; small differences from the first-submission tables (e.g. SC 0.3887→0.3885, VDI 0.7794→0.7823, V4 reference K 8→10, cluster sizes ±1–2) are due to the first submission having been produced with an unreleased script.
* Cluster numbers are ordered by size.