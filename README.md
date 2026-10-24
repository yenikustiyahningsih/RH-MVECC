# RH-MVECC — revised code (response to Array reviewers)

Reliability-Weighted Heterogeneous Multi-View Ensemble Consensus Clustering with View-Dominance Control.
This package replaces the single-script reconstruction (`rh_mvecc.py`) with a dataset-agnostic library and
one script per reviewer request. All numbers in the revised manuscript are produced by these scripts.

## Structure

| File | Purpose | Reviewer comment |
|---|---|---|
| `rhmvecc/core.py` | Library: views, reliability (Eq. 2–5), 45-partition library, partition weights (Eq. 6–7), weighted co-association (Eq. 8–9), VDI (Eq. 10), selection (Eq. 11), stability. Every design constant is a `Config` field. | all |
| `rhmvecc/msme.py` | MSME loader (`MinMax_2896` + `MinMax_Info` sheets, or CSV) | R2-4 |
| `run_msme.py` | Main experiment: Tables 4–14, Fig. 2–4, paired bootstrap vs K-Means, threshold & weight sensitivity | R2-3, R2-5 |
| `exp_vdi_diagnostics.py` | NMI profile, information decomposition, surrogate trees, reference partitions, nuisance-view injection | R2-2, R3-1 |
| `exp_nuisance_detail.py`, `exp_penguins_svs.py`, `exp_library_redundancy.py` | Nuisance K-scan (Table 17), Penguins VDI vs SVS, library pairwise ARI | R2-2, R2-3 |
| `exp_vdi_share.py` | Asymmetric single-view explained share SVS(K) (Eq. 12) | R2-2 |
| `exp_ablation_sensitivity.py` | Simpler ablation baselines, selection-rule ablation, α/β/adaptive (Eq. 4), ε (Eq. 7), effect of T_v | R2-3, R2-5, R3-3, R3-4 |
| `exp_stability.py` | Library-fraction, seed-level re-runs, MSME-level subsampling | R2-5 |
| `exp_benchmarks.py` | Public multi-view benchmarks with ground truth (ACC/NMI/ARI/Purity) + nuisance-view stress test | R2-3, R3-2, R3-5 |
| `make_figures.py` | Manuscript figures | |

## Running

```bash
pip install -r requirements.txt
python run_msme.py               --input data/data_prep_2896.xlsm --out results/main
python exp_vdi_diagnostics.py    --input data/data_prep_2896.xlsm --out results/vdi
python exp_vdi_share.py          data/data_prep_2896.xlsm
python exp_ablation_sensitivity.py --input data/data_prep_2896.xlsm --out results/ablation --repeats 20
python exp_stability.py          --input data/data_prep_2896.xlsm --out results/stability
bash benchmarks/download_benchmarks.sh   # if the .mat files are not present
python exp_benchmarks.py --data benchmarks --out results/benchmarks --only 3Sources,BBCSport,MSRC-v5,Penguins
python exp_benchmarks.py --data benchmarks --out results/benchmarks --only Handwritten --seeds 3
python exp_benchmarks.py --data benchmarks --out results/benchmarks --only Caltech101-7 --seeds 3
python exp_penguins_svs.py
python exp_nuisance_detail.py data/data_prep_2896.xlsm
python exp_library_redundancy.py
python make_figures.py
```

Approximate run time on a 2-vCPU machine: main 2.5 min; diagnostics ~1 h; ablation/sensitivity ~1.5 h;
stability ~1 h; benchmarks ~1.5 h.

## Seeds and repeated-run procedures

* K-Means (`n_init = 20`) and GMM (full covariance, `n_init = 1`, `reg_covar = 1e-6`): `random_state = 42`.
* View stability T_v: 10 additional K-Means runs with seeds 0–9 at the selected within-view K; T_v = mean ARI against the seed-42 reference.
* Ensemble stability: `numpy.random.default_rng(123)`; 50 draws of 36 of the 45 partitions (80 %) without replacement; weights renormalised; consensus rebuilt; ARI/NMI against the full-library consensus. Every draw (with the indices of dropped partitions) is saved in `results/main/stability_runs.csv`.
* Composite-weight perturbation: 10,000 draws from the same generator after the resampling draws (±20 % and ±50 %).
* Paired bootstrap RH-MVECC vs K-Means: `default_rng(2024)`, 200 subsamples of 80 % MSMEs.
* Seed-level re-runs: master seed s = 0…9 (library seed s, view-stability seeds 10s…10s+9, resampling seed 1000+s).
* MSME subsampling: `default_rng(555)`, 10 subsets of 80 % MSMEs, complete pipeline re-run.
* Benchmarks: seeds 0–4 (0–2 for Handwritten and Caltech101-7); nuisance view drawn with default_rng(99); MSME nuisance test seeds 11–13.

## Data

`data/` must contain the normalized MSME matrix (`data_prep_2896.xlsm`, sheets `MinMax_2896` and `MinMax_Info`).
The SHA-256 of the input is written to `results/main/summary.json`.

> Note for the authors: the original `data_prep_2896.xlsm` was not part of the package used for this revision.
> The file `data/data_prep_2896_reconstructed.xlsx` was rebuilt from the previous run's `assignments` sheet
> (values rounded to 6 decimals) and its `MinMax_Info` sheet was back-solved from the reported original-scale
> profiles. Re-running the original script on it reproduces the previous results exactly
> (SC = 0.38850, DBI = 1.02508, CHI = 1233.99, ARI = 0.99001, VDI = 0.78230). Please re-run `run_msme.py`
> on the original workbook before submission and deposit that workbook (or the anonymised normalized matrix)
> with the code.

Benchmarks (`benchmarks/`): 3Sources, BBCSport, MSRC-v5, Handwritten, Caltech101-7 (.mat files from the public
collection https://github.com/ChuanbinZhang/Multi-view-datasets) and Palmer Penguins
(https://github.com/mwaskom/seaborn-data). Any `.mat` file with a cell array `X` (views) and label vector `y`
(e.g. NGs) can be added to `exp_benchmarks.py` without code changes other than the dataset list.

## Differences from the first submission

* Numbers now come from one released code base; small differences from the first-submission tables
  (e.g. SC 0.3887→0.3885, VDI 0.7794→0.7823, V4 reference K 8→10, cluster sizes ±1–2) are due to the
  first submission having been produced with an unreleased script.
* Cluster numbers are ordered by size.
