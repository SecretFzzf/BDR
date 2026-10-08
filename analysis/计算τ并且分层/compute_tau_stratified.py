# -*- coding: utf-8 -*-
"""
数据集分层传导效率 (Tau) 独立统计脚本
读取仓库 results/seed42/tau_results.csv，按 4 个数据集分层统计并输出 txt 报告。
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DATA_FILE = REPO_ROOT / "results" / "seed42" / "tau_results.csv"
OUTPUT_FILE = REPO_ROOT / "results" / "计算τ并且分层" / "tau_stratified_report.txt"

# 必须包含的列
REQUIRED_COLS = ["dataset", "delta_fe", "delta_ie", "tau"]

# ── 1. 读取数据 ──────────────────────────────────────────────────────────────
print(f"读取数据: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print(f"  原始行数: {len(df)}")
print(f"  列名: {list(df.columns)}")

# 校验必要列
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise KeyError(f"缺少必要列: {missing}")

# ── 2. 数据清洗 ──────────────────────────────────────────────────────────────
# 仅保留 tau 有效（非 NaN 且有限）的行
df_clean = df[df["tau"].notna() & np.isfinite(df["tau"])].copy()
# 同时要求 delta_fe 和 delta_ie 有效
df_clean = df_clean[
    df_clean["delta_fe"].notna()
    & df_clean["delta_ie"].notna()
    & np.isfinite(df_clean["delta_fe"])
    & np.isfinite(df_clean["delta_ie"])
].copy()
print(f"  有效行数: {len(df_clean)} (剔除 {len(df) - len(df_clean)} 条无效记录)")

# ── 3. 全局汇总 ──────────────────────────────────────────────────────────────
N_total = len(df_clean)
global_median_tau = float(df_clean["tau"].median())

# 纯倒挂率：插补更优且预测更差（ΔIE < 0 且 ΔFE > 0）的样本占比
pool_improve = df_clean[(df_clean["delta_ie"] < 0) & (df_clean["delta_fe"] > 0)]
global_inversion_rate = float(len(pool_improve) / len(df_clean) * 100)

# ── 4. 按数据集分层统计 ──────────────────────────────────────────────────────
datasets = sorted(df_clean["dataset"].unique())
stratified = []

for ds in datasets:
    sub = df_clean[df_clean["dataset"] == ds].copy()
    N = len(sub)

    # Tau 统计
    tau_median = float(sub["tau"].median())
    tau_q25 = float(sub["tau"].quantile(0.25))
    tau_q75 = float(sub["tau"].quantile(0.75))
    tau_iqr = tau_q75 - tau_q25

    # ΔFE 和 ΔIE 中位数
    delta_fe_median = float(sub["delta_fe"].median())
    delta_ie_median = float(sub["delta_ie"].median())

    # 纯倒挂率：ΔIE < 0 且 ΔFE > 0 的样本占比
    improve = sub[(sub["delta_ie"] < 0) & (sub["delta_fe"] > 0)]
    inversion_rate = float(len(improve) / len(sub) * 100)

    # 极端离群点（3×IQR 原则）
    lower_fence = tau_q25 - 3 * tau_iqr
    upper_fence = tau_q75 + 3 * tau_iqr
    outlier_count = int(
        ((sub["tau"] < lower_fence) | (sub["tau"] > upper_fence)).sum()
    )

    stratified.append(
        {
            "dataset": ds,
            "N": N,
            "tau_median": tau_median,
            "tau_q25": tau_q25,
            "tau_q75": tau_q75,
            "tau_iqr": tau_iqr,
            "delta_fe_median": delta_fe_median,
            "delta_ie_median": delta_ie_median,
            "inversion_rate": inversion_rate,
            "outlier_count": outlier_count,
        }
    )

# ── 5. 统计结论摘要 ──────────────────────────────────────────────────────────
# 结论1：各数据集 Tau 中位数是否均贴近零点（|median| < 0.1 视为贴近）
all_near_zero = all(abs(s["tau_median"]) < 0.1 for s in stratified)
conclusion_1 = "是" if all_near_zero else "否"

# 结论2：倒挂现象是否在所有数据集上普遍存在（纯倒挂率 > 10% 视为普遍）
all_have_inversion = all(s["inversion_rate"] > 10.0 for s in stratified)
conclusion_2 = "是" if all_have_inversion else "否"

# 结论3：极端离群点的主要分布数据集（离群点数最多的数据集）
max_outlier_ds = max(stratified, key=lambda s: s["outlier_count"])
conclusion_3 = (
    f"{max_outlier_ds['dataset']}（{max_outlier_ds['outlier_count']} 个离群点）"
    if max_outlier_ds["outlier_count"] > 0
    else "各数据集均无显著极端离群点"
)

# ── 6. 生成报告 ──────────────────────────────────────────────────────────────
lines = []
sep = "=" * 80
dash = "-" * 80

lines.append(sep)
lines.append("                    数据集分层传导效率 (Tau) 独立统计报告")
lines.append(sep)
lines.append("")
lines.append("[全局汇总 (Pooled Summary)]")
lines.append(f"全样本总量: N = {N_total}")
lines.append(f"全局 Tau 中位数: {global_median_tau:.4f}")
lines.append(f"全局 纯倒挂率: {global_inversion_rate:.2f}%")
lines.append("")
lines.append(dash)
lines.append("[数据集分层详细统计]")
lines.append("")

for s in stratified:
    lines.append(f"数据集: {s['dataset']}")
    lines.append(f"  - 样本量 (N): {s['N']}")
    lines.append(f"  - Tau 中位数: {s['tau_median']:.4f}")
    lines.append(f"  - Tau 四分位距 (IQR): {s['tau_iqr']:.4f}")
    lines.append(f"  - ΔFE 中位数: {s['delta_fe_median']:.4f}")
    lines.append(f"  - ΔIE 中位数: {s['delta_ie_median']:.4f}")
    lines.append(f"  - 纯倒挂率 (Pure Reversal Rate): {s['inversion_rate']:.2f}%")
    lines.append(f"  - 极端离群点数 (Outliers Count): {s['outlier_count']}")
    lines.append("")

lines.append(dash)
lines.append("[统计结论摘要]")
lines.append(f"1. 各数据集 Tau 中位数是否均贴近零点: {conclusion_1}")
lines.append(f"2. 倒挂现象是否在所有数据集上普遍存在: {conclusion_2}")
lines.append(f"3. 极端离群点的主要分布数据集: {conclusion_3}")
lines.append(sep)

report = "\n".join(lines)

# ── 7. 写入文件 ──────────────────────────────────────────────────────────────
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n报告已写入: {OUTPUT_FILE}")
print()
print(report)
