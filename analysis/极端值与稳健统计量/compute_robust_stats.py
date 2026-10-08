from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SRC = REPO_ROOT / "results" / "seed42" / "tau_results.csv"
OUT_DIR = REPO_ROOT / "results" / "极端值与稳健统计量"
CSV_OUT = OUT_DIR / "robust_stats_summary.csv"
REPORT_OUT = OUT_DIR / "robust_stats_report.txt"


def trimmed_mean(values, proportion):
    ordered = np.sort(values)
    cut = int(np.floor(len(ordered) * proportion))
    return float(ordered[cut : len(ordered) - cut].mean())


data = pd.read_csv(SRC)
data = data[data["tau_status"] == "valid"].copy()
tau = data["tau"].to_numpy()
median = float(np.median(tau))

statistics = {
    "样本总量": float(len(tau)),
    "算术均值(Raw Mean)": float(np.mean(tau)),
    "样本标准差(Std)": float(np.std(tau, ddof=1)),
    "中位数(Median)": median,
    "中位数绝对偏差(MAD)": float(np.median(np.abs(tau - median))),
    "5%截尾均值": trimmed_mean(tau, 0.05),
    "10%截尾均值": trimmed_mean(tau, 0.10),
}
summary = pd.DataFrame(
    {"统计量名称": list(statistics), "数值": list(statistics.values())}
)

extreme = data[np.abs(data["tau"]) > 2.0].copy()
extreme_negative = int((extreme["tau"] < -2.0).sum())
extreme_positive = int((extreme["tau"] > 2.0).sum())
dataset_counts = extreme.groupby("dataset").size().sort_values(ascending=False)
mean_abs_delta_ie = float(extreme["delta_ie"].abs().mean())
median_abs_delta_ie = float(extreme["delta_ie"].abs().median())
boundary_share = float(
    extreme["delta_ie"].abs().between(0.010, 0.025, inclusive="both").mean()
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(CSV_OUT, index=False)

dataset_lines = "\n".join(
    f"                 {dataset:<15} {count:>3}" for dataset, count in dataset_counts.items()
)
report = f"""# 传导效率指标τ的极端值与稳健统计量分析报告
================================================================================

## 一、基础与稳健统计量汇总
总有效样本数 N = {len(tau)}

| 统计量名称               | 数值          |
|--------------------------|---------------|
| 算术均值（Raw Mean）     | {statistics['算术均值(Raw Mean)']:.4f} |
| 样本标准差（Std）        | {statistics['样本标准差(Std)']:.4f} |
| 中位数（Median）        | {statistics['中位数(Median)']:.4f} |
| 中位数绝对偏差（MAD）   | {statistics['中位数绝对偏差(MAD)']:.4f}     |
| 5%截尾均值（5% Trimmed） | {statistics['5%截尾均值']:.4f}   |
| 10%截尾均值（10% Trimmed）| {statistics['10%截尾均值']:.4f}   |

## 二、极端值（|τ| > 2.0）归因分析
1.  极端值总样本数：{len(extreme)}（占总样本的 {len(extreme) / len(tau):.2%}）
2.  极端负值（τ < -2.0）样本数：{extreme_negative}
3.  极端正值（τ > 2.0）样本数：{extreme_positive}
4.  极端值按数据集分布：
                 数据集          极端值样本数
{dataset_lines}

5.  极端值样本的|ΔIE|分布特征：
    - 平均|ΔIE|: {mean_abs_delta_ie:.4f}
    - 中位数|ΔIE|: {median_abs_delta_ie:.4f}
    - 位于临界边界[0.010, 0.025]的比例：{boundary_share:.2%}

## 三、算术均值与中位数严重偏离的归因总结
当前τ的算术均值为 {statistics['算术均值(Raw Mean)']:.4f}，严重偏离中位数 {statistics['中位数(Median)']:.4f}，核心原因如下：
1.  **数值放大效应**：极端负值样本的分母ΔIE接近阈值0.01，导致τ被剧烈放大。
2.  **边界样本验证**：{boundary_share:.2%}的极端值样本位于[0.010, 0.025]的临界边界附近。
3.  **稳健统计量佐证**：5%截尾均值({statistics['5%截尾均值']:.4f})和10%截尾均值({statistics['10%截尾均值']:.4f})支持使用中位数和截尾均值描述典型传导效率。
"""
REPORT_OUT.write_text(report, encoding="utf-8")
print(summary.to_string(index=False))
print(f"Saved: {CSV_OUT}")
print(f"Saved: {REPORT_OUT}")
