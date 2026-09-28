# Impute-then-Forecast 实证研究

本仓库研究时间序列的 impute-then-forecast 两步流水线：先插补缺失值，再进行多步预测，并使用传递效率指标 `τ` 分析插补误差与预测误差之间的关系。实验使用 ETTh1、KDD-Beijing、Weather 和 SWaT，覆盖 MCAR、MAR_Block、MNAR 三种缺失机制，缺失率为 10%、30%、50%、70%；插补方法包括 Mean、Spline、KNN、BRITS、SAITS，预测模型为 LSTM、DLinear 和 PatchTST。

## 仓库结构

| 路径 | 说明 |
|---|---|
| `data_and_code/ETT/` | ETTh1 主实验 |
| `data_and_code/ETT_seed123/` | ETTh1 随机种子 123 实验 |
| `data_and_code/ETT_seed2026/` | ETTh1 随机种子 2026 实验 |
| `data_and_code/KDD-Beijing/` | KDD-Beijing 主实验 |
| `data_and_code/weather/` | Weather 主实验 |
| `data_and_code/SWaT/` | SWaT 主实验 |
| `data_and_code/SWaT_seed123/` | SWaT 随机种子 123 实验 |
| `data_and_code/SWaT_seed2026/` | SWaT 随机种子 2026 实验 |
| `photo/` | 论文图生成脚本及已生成的 PNG 图片 |

每个实验目录包含 `engine.py`、`tool.py`、`run_all_models.py`、`run_LSTM.py`、`run_DLinear.py` 和 `run_PatchTST.py`。其中 `engine.py` 负责训练与评估，`tool.py` 负责缺失注入和插补，`run_all_models.py` 依次运行三种预测模型，其余脚本用于运行单个模型。

## 环境安装

```bash
python -m pip install -r requirements.txt
```

## 数据说明

数据文件不包含在本仓库中。

| 数据集 | 获取方式 | 放置位置 | 对应实验目录 |
|---|---|---|---|
| ETTh1 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) | `data_and_code/ETT*/ETTh1.csv` | `ETT*` |
| KDD-Beijing | [KDD Cup 2018](https://www.kdd.org/kdd-cup/view/kdd-cup-2018) | `data_and_code/KDD-Beijing/kdd_beijing_raw.csv` | `KDD-Beijing` |
| Weather | [Informer2020 数据目录](https://github.com/zhouhaoyi/Informer2020/tree/main/data/weather) | `data_and_code/weather/weather.csv` | `weather` |
| SWaT | 向 [iTrust Labs](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) 申请 | `data_and_code/SWaT*/normal.csv` | `SWaT*` |

SWaT 使用正常运行工况数据，按时间戳将数值特征降采样为一分钟均值，并保存为 `normal.csv`；时间戳、攻击标签和无意义索引列不作为输入特征。

## 运行实验

主实验：

```bash
python data_and_code/ETT/run_all_models.py
python data_and_code/KDD-Beijing/run_all_models.py
python data_and_code/weather/run_all_models.py
python data_and_code/SWaT/run_all_models.py
```

多种子实验：

```bash
python data_and_code/ETT_seed123/run_all_models.py
python data_and_code/ETT_seed2026/run_all_models.py
python data_and_code/SWaT_seed123/run_all_models.py
python data_and_code/SWaT_seed2026/run_all_models.py
```

## 结果文件

实验脚本会在各自目录生成 `results_refactor_*.csv`，但结果 CSV 未包含在当前仓库中。汇总表 `tau_results.csv` 也未包含；`photo/plot_all_figures.py` 等绘图脚本读取 `D:/ei/result/seed42/tau_results.csv`，输出到 `D:/ei/mended_photo/`。

汇总表主要列：`dataset` 为数据集，`model` 为预测模型，`missing_mode` 为缺失机制，`missing_rate` 为缺失率，`pred_len` 为预测步长，`impute_method` 为插补方法，`ie_mae_*` 为插补误差，`fe_mae*` 为预测误差，`delta_ie`、`delta_fe` 为相对 Mean 基线的误差变化，`tau` 为传递效率，`tau_status` 表示 τ 是否有效。

本仓库不包含模型权重、训练断点、缓存文件和实验结果 CSV。

## License

MIT License，见 `LICENSE`。

## Citation

论文投稿中，录用后更新引用信息。

Zenodo DOI：`[DOI 待填]`

## Contact

`[邮箱待填]`
