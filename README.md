# Better Imputation Does Not Guarantee Better Forecasting: Transmission Efficiency in Two-Step Time Series Pipelines

This repository contains the code and result tables for a paper currently under submission. It is an empirical study, not a new model proposal.

## Overview

This study examines impute-then-forecast pipelines: how much of the improvement in reconstruction error is transmitted to forecasting error. The central diagnostic is the transmission-efficiency ratio `tau`; the elasticity `eta` is reported as a supplementary relative-error diagnostic. The experiment matrix covers 4 datasets, 3 missingness mechanisms, 4 missing rates, 5 imputers, 3 forecasters, and 4 forecast horizons. The code evaluates existing imputation and forecasting components and analyzes their interaction.

## Repository Structure

```text
paper_repo/
├── analysis/
│   ├── anchor_sensitivity/       Baseline-anchor sensitivity analysis.
│   ├── cluster_bootstrap/        Table 6 cluster-aware bootstrap analysis.
│   ├── epsilon_sensitivity/      Figure 3 epsilon-filter sensitivity analysis.
│   ├── lstm_epochs_ett/          ETTh1 50-epoch versus 100-epoch LSTM check.
│   ├── multi_seed_sign_flip/     Five-seed ETTh1 merge and sign-flip analysis.
│   ├── paired_tests/             McNemar, Wilcoxon, and paired bootstrap tests.
│   ├── robust_stats/             Robust statistics and extreme-tau analysis.
│   └── tau_stratified/           tau/eta, stratified, MSE, and relative-magnitude analyses.
├── data_and_code/
│   ├── ETT/                      ETTh1 main experiment code.
│   ├── ETT_seed123/              ETTh1 code copy with SEED = 123.
│   ├── ETT_seed2026/             ETTh1 code copy with SEED = 2026.
│   ├── KDD-Beijing/              KDD-Beijing preprocessing and experiment code.
│   ├── SWaT/                     SWaT experiment code.
│   └── weather/                  Jena Weather experiment code.
├── photo/                        Figure scripts and generated figures.
├── results/                      Consolidated and analysis result tables/reports.
└── requirements.txt              Python dependency names used by the repository.
```

## Environment

The experiment environment was Python 3.10 and PyTorch 2.12.0+cu130 (the environment used for the experiments). The dependency file lists package names without version pins.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Datasets

No dataset files are redistributed in this repository.

| Dataset | Source | Where to put it | Used by |
|---|---|---|---|
| ETTh1 | ETDataset from the Informer paper: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv | `data/ETTh1.csv` | `data_and_code/ETT`, `ETT_seed123`, `ETT_seed2026` |
| KDD-Beijing | KDD Cup 2018 Beijing subset via the Monash Time Series Forecasting Repository: https://doi.org/10.5281/zenodo.4656719 | `data/kdd_beijing_raw.csv` | `data_and_code/KDD-Beijing` |
| Weather | Jena Weather dataset distributed with Autoformer: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing | `data/weather.csv` | `data_and_code/weather` |
| SWaT | Request access from iTrust, Singapore University of Technology and Design: https://www.sutd.edu.sg/itrust/request-for-datasets/ | `data/normal.csv` | `data_and_code/SWaT` |

For KDD-Beijing, preprocess the downloaded TSF file first:

```powershell
python data_and_code/KDD-Beijing/preprocess_kdd_beijing.py --data_dir data --output data/kdd_beijing_raw.csv
```

SWaT requires an approved iTrust data request. The linked page is iTrust's official request form; after access is approved, the requester downloads the data directly from iTrust. The repository contains SWaT experiment scripts but no SWaT raw data, processed data, or derived cache files. Prepare the processed one-minute mean input as `data/normal.csv`; the 1 Hz to 60-second mean-aggregation code is not included in this repository. The other three datasets must likewise be downloaded from their sources and placed in `data/`.

## Reproduction

### Main experiments

Run each dataset from the repository root. The scripts accept `--data_dir`; the default is `data/` under the repository root.

```powershell
python data_and_code/ETT/run_all_models.py --data_dir data
python data_and_code/KDD-Beijing/run_all_models.py --data_dir data
python data_and_code/weather/run_all_models.py --data_dir data
python data_and_code/SWaT/run_all_models.py --data_dir data
```

The engines write per-model CSV files under `results/training/<code-folder>/`. The analysis scripts expect the consolidated file `results/seed42/all_models_result.csv`; the consolidation script is not present in this repository. The consolidated CSV is included as an analysis input.

### Multi-seed experiment

Results for seeds 42, 123, 2026, 456, and 789 are included under `results/multi_seed_sign_flip/`; source copies for 456 and 789 are not included. In these archived results, `ie_mae` and `ie_mse` are identical across all five seeds for every configuration, while `fe_mae` varies. The multi-seed experiment therefore holds the missingness masks and imputed inputs fixed and varies only the forecasting run.

Implementation note: the included source copies are not fully aligned with that archived protocol. Their `SEED` constants are passed into the missingness/imputation functions, and no separate PyTorch forecast seed is set. A clean rerun can therefore change the masks and may not reproduce the released multi-seed tables. [TODO: encode separate imputation and forecasting seeds before release.]

Run the included seed copies with:

```powershell
python data_and_code/ETT/run_all_models.py --data_dir data
python data_and_code/ETT_seed123/run_all_models.py --data_dir data
python data_and_code/ETT_seed2026/run_all_models.py --data_dir data
python analysis/multi_seed_sign_flip/merge_5seeds_sign_flip.py
```

### Analysis and figures

Run `compute_tau.py` first; the remaining scripts read its output or the consolidated result table.

```powershell
python analysis/tau_stratified/compute_tau.py
python analysis/tau_stratified/compute_tau_stratified.py
python analysis/tau_stratified/compute_mse_sensitivity.py
python analysis/tau_stratified/compute_relative_magnitudes.py
python analysis/tau_stratified/compute_eta_mse_extra.py
python analysis/robust_stats/compute_robust_stats.py
python analysis/anchor_sensitivity/compute_anchor_sensitivity.py
python analysis/epsilon_sensitivity/epsilon_sensitivity_analysis.py
python analysis/cluster_bootstrap/compute_cluster_bootstrap.py
python analysis/paired_tests/compute_paired_significance.py
python analysis/lstm_epochs_ett/compare_50vs100.py
python photo/fig_fe_lines.py
python photo/fig1_tau_distribution.py
python photo/fig4_inversion_heatmap.py
```

All paths are resolved relative to each script or through `--data_dir`; no training script is required for the analysis commands above.

## Mapping to the Paper

| Paper item | Code | Result/output |
|---|---|---|
| Figure 1, FE by imputer and forecaster | `photo/fig_fe_lines.py` | `photo/fig_fe_lines.{pdf,png}` |
| Figure 2, delta-IE/delta-FE quadrant | `photo/fig1_tau_distribution.py` | `photo/fig1_tau_distribution.{pdf,png}` |
| Figure 3, epsilon sensitivity | `analysis/epsilon_sensitivity/epsilon_sensitivity_analysis.py` | `photo/fig3.{pdf,png}`, `results/epsilon_sensitivity/` |
| Figure 4, inversion heatmap | `photo/fig4_inversion_heatmap.py` | `photo/fig4_inversion_heatmap.{pdf,png}` |
| Table 2, dataset-stratified statistics | `analysis/tau_stratified/compute_tau_stratified.py` | `results/tau_stratified/tau_stratified_report.txt` |
| Table 3, epsilon sensitivity | `analysis/epsilon_sensitivity/epsilon_sensitivity_analysis.py` | `results/epsilon_sensitivity/epsilon_sensitivity_summary.csv` |
| Table 4, robust statistics | `analysis/robust_stats/compute_robust_stats.py` | `results/robust_stats/robust_stats_summary.csv`, `robust_stats_report.txt` |
| Table 5, anchor sensitivity | `analysis/anchor_sensitivity/compute_anchor_sensitivity.py` | `results/anchor_sensitivity/` |
| Table 6, stratified inversion and cluster bootstrap | `analysis/cluster_bootstrap/compute_cluster_bootstrap.py` | `results/cluster_bootstrap/table6_cluster_bootstrap.*` |
| Table 7, sign flips over five seeds | `analysis/multi_seed_sign_flip/merge_5seeds_sign_flip.py` | `results/multi_seed_sign_flip/merged_5seeds_ett_results.csv` |
| McNemar/Wilcoxon/paired bootstrap | `analysis/paired_tests/compute_paired_significance.py` | `results/paired_tests/table1_stat_*` |
| MSE sensitivity | `analysis/tau_stratified/compute_mse_sensitivity.py` | `results/tau_stratified/mse_sensitivity_*` |
| 50/100-epoch LSTM check | `analysis/lstm_epochs_ett/compare_50vs100.py` | `test1.csv` (50-epoch baseline), `results_refactor_lstm.csv` (100-epoch check), `compare_50vs100_report.txt` |
| Relative magnitudes (1.72%, 28.17%, 7.59%) | `analysis/tau_stratified/compute_relative_magnitudes.py` | `results/tau_stratified/relative_magnitude_*` |

`results/robust_stats/outliers_breakdown.csv` and
`results/robust_stats/outliers_by_dataset.csv` are supplementary details for the 66 samples with
`|tau| > 2` discussed in Section 4.2. They provide row-level and dataset-level breakdowns of the
extreme-value analysis and are not the Table 4 robust-statistics output.

`compute_eta_mse_extra.py` writes the supplementary elasticity and MSE-sensitivity statistics to
`results/tau_stratified/eta_mse_extra_report.txt`; these support the paper's elasticity and error-metric
sensitivity paragraphs rather than numbered tables. The exact paper label for the paired-tests table is
[TODO: confirm exact paper label].

## Naming Note

`run_DLinear.py` implements the paper's **Linear (per-channel)** forecaster: it has no trend-seasonal decomposition. `run_PatchTST.py` implements **PatchTST (simplified)**: it has no RevIN. The filenames are retained to match the source code.

## What Is Not Included

Dataset files; any SWaT raw, processed, or derived data/cache files; trained model weights (`.pt`, `.pth`, `.ckpt`); intermediate imputation caches (`.npy`, `.pkl`, `.parquet`); logs; and notebook checkpoints.

## License

MIT License. Copyright (c) 2026 Yu Chang. See [`LICENSE`](LICENSE).

## Citation

Citation information will be added after the paper is accepted.

## Contact

krys4taa1@gmail.com
