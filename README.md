# An Empirical Study of Impute-then-Forecast

This repository studies a two-stage impute-then-forecast pipeline for time-series data: missing values are first imputed, followed by multi-step forecasting. The transfer-efficiency metric `tau` is used to analyze the relationship between imputation error and forecasting error. The experiments use ETTh1, KDD-Beijing, Weather, and SWaT; cover the MCAR, MAR_Block, and MNAR missingness mechanisms with missing rates of 10%, 30%, 50%, and 70%; use Mean, Spline, KNN, BRITS, and SAITS for imputation; and use LSTM, DLinear, and PatchTST for forecasting.

## Repository Structure

| Path | Description |
|---|---|
| `data_and_code/ETT/` | Main ETTh1 experiments |
| `data_and_code/ETT_seed123/` | ETTh1 experiments with random seed 123 |
| `data_and_code/ETT_seed2026/` | ETTh1 experiments with random seed 2026 |
| `data_and_code/KDD-Beijing/` | Main KDD-Beijing experiments |
| `data_and_code/weather/` | Main Weather experiments |
| `data_and_code/SWaT/` | Main SWaT experiments |
| `data_and_code/SWaT_seed123/` | SWaT experiments with random seed 123 |
| `data_and_code/SWaT_seed2026/` | SWaT experiments with random seed 2026 |
| `photo/` | Paper-figure scripts and generated PNG figures |

Each experiment directory contains `engine.py`, `tool.py`, `run_all_models.py`, `run_LSTM.py`, `run_DLinear.py`, and `run_PatchTST.py`. The `engine.py` script handles training and evaluation, `tool.py` handles missing-value injection and imputation, and `run_all_models.py` runs the three forecasting models sequentially. The remaining scripts run individual models.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Data

Data files are not included in this repository.

| Dataset | Access | Placement | Experiment Directory |
|---|---|---|---|
| ETTh1 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) | `data_and_code/ETT*/ETTh1.csv` | `ETT*` |
| KDD-Beijing | [KDD Cup 2018](https://www.kdd.org/kdd-cup/view/kdd-cup-2018) | `data_and_code/KDD-Beijing/kdd_beijing_raw.csv` | `KDD-Beijing` |
| Weather | [Informer2020 data directory](https://github.com/zhouhaoyi/Informer2020/tree/main/data/weather) | `data_and_code/weather/weather.csv` | `weather` |
| SWaT | Request access from [iTrust Labs](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) | `data_and_code/SWaT*/normal.csv` | `SWaT*` |

SWaT uses data from normal operating conditions. The numeric features are downsampled to one-minute means according to the timestamp and saved as `normal.csv`. Timestamps, attack labels, and non-informative index columns are not used as input features.

## Running the Experiments

Main experiments:

```bash
python data_and_code/ETT/run_all_models.py
python data_and_code/KDD-Beijing/run_all_models.py
python data_and_code/weather/run_all_models.py
python data_and_code/SWaT/run_all_models.py
```

Multi-seed experiments:

```bash
python data_and_code/ETT_seed123/run_all_models.py
python data_and_code/ETT_seed2026/run_all_models.py
python data_and_code/SWaT_seed123/run_all_models.py
python data_and_code/SWaT_seed2026/run_all_models.py
```

## Result Files

The experiment scripts generate `results_refactor_*.csv` files in their respective directories, but the result CSV files are not included in this repository. The summary table `tau_results.csv` is also not included. Figure scripts such as `photo/plot_all_figures.py` read `D:/ei/result/seed42/tau_results.csv` and write output to `D:/ei/mended_photo/`.

The main summary-table columns are: `dataset` for the dataset; `model` for the forecasting model; `missing_mode` for the missingness mechanism; `missing_rate` for the missing rate; `pred_len` for the forecasting horizon; `impute_method` for the imputation method; `ie_mae_*` for imputation error; `fe_mae*` for forecasting error; `delta_ie` and `delta_fe` for error changes relative to the Mean baseline; `tau` for transfer efficiency; and `tau_status` indicating whether `tau` is valid.

This repository does not include model weights, training checkpoints, cache files, or experiment-result CSV files.

## License

MIT License. See `LICENSE`.

## Citation

The paper is under submission. Citation information will be updated after acceptance.

Zenodo DOI: `[DOI to be filled]`

## Contact

`[Email address]`
