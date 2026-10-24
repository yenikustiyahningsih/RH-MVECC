# RH-MVECC: code, seeds and results

Reliability-Weighted Heterogeneous Multi-View Ensemble Consensus Clustering with View-Dominance Control, applied to MSME segmentation. This repository contains the dataset-agnostic library (`rhmvecc/`), one script per experiment, all random seeds, and the raw outputs of every experiment reported in the revised manuscript submitted to *Array*.

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
python make_figures.py
```

Approximate run time on a 2-vCPU machine: main 2.5 min; diagnostics about 1 h; ablation/sensitivity about 1.5 h; stability about 1 h; benchmarks about 1.5 h. On Windows run the `bash` and `$D` lines in Git Bash.

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

Benchmarks (`benchmarks/`): 3Sources, BBCSport, MSRC-v5, Handwritten and Caltech101-7 (.mat files from https://github.com/ChuanbinZhang/Multi-view-datasets; checksums in `benchmarks/SHA256SUMS.txt`) and Palmer Penguins (https://github.com/mwaskom/seaborn-data). Any `.mat` file with a cell array `X` (views) and a label vector `y` (e.g. NGs) can be added to `exp_benchmarks.py` by extending the dataset list.

Note on `results/benchmarks/bench_*.csv`: for Handwritten, Caltech101-7 and Penguins each method appears twice per seed. The first block (in file order) is the run on the original views, which gives Table 23; the second block is the run with the random binary nuisance view added. Do not average the two blocks together.

## Differences from the first submission

* Numbers now come from one released code base; small differences from the first-submission tables (e.g. SC 0.3887→0.3885, VDI 0.7794→0.7823, V4 reference K 8→10, cluster sizes ±1–2) are due to the first submission having been produced with an unreleased script.
* Cluster numbers are ordered by size.