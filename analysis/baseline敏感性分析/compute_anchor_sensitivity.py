"""
基线锚点 (Baseline Anchor) 敏感性分析
--------------------------------------
验证将 τ 的基线锚点从 Mean 切换为 Spline 后，传导效率统计结论是否稳健。

原基线（Mean 锚点）：
    ΔIE = IE(φ) - IE(Mean)
    ΔFE = FE(φ) - FE(Mean)
    τ = ΔFE / ΔIE

新基线（Spline 锚点，排除 imputer == 'spline' 本身）：
    ΔIE = IE(φ) - IE(Spline)
    ΔFE = FE(φ) - FE(Spline)
    τ = ΔFE / ΔIE
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SRC = REPO_ROOT / "results" / "seed42" / "all_models_result.csv"
OUT_DIR = REPO_ROOT / "results" / "baseline敏感性分析"
OUT_FILE = OUT_DIR / "anchor_sensitivity_report.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 0.01   # 与论文主实验一致的数值稳定性阈值

# ── 读取数据 ───────────────────────────────────────────
df = pd.read_csv(SRC)

# 只保留成功运行的行
df = df[df["status"] == "success"].copy()

# 统一缺失机制命名（MAR_Block / MAR_Block）
df["missing_mode"] = df["missing_mode"].str.replace("MAR_Block", "MAR_Block", regex=False)

# 只保留需要的列
GROUP_KEYS = ["dataset", "model", "missing_mode", "missing_rate", "pred_len"]
df = df[GROUP_KEYS + ["impute_method", "ie_mae", "fe_mae"]].copy()

print(f"原始成功记录数: {len(df)}")
print(f"插补方法种类: {sorted(df['impute_method'].unique())}")
print()

# ── 辅助函数 ───────────────────────────────────────────

def compute_tau_stats(df_long, anchor_method, epsilon=EPSILON):
    """
    以 anchor_method 为基线，计算 τ 的全样本统计量。
    排除 anchor_method 自身作为 φ 的样本（分母为 0）。
    """
    # 每个 group 中取出 anchor 的 IE / FE
    anchor = (
        df_long[df_long["impute_method"] == anchor_method]
        .set_index(GROUP_KEYS)[["ie_mae", "fe_mae"]]
        .rename(columns={"ie_mae": "ie_anchor", "fe_mae": "fe_anchor"})
    )

    # 左连接：每行减去同组 anchor 值
    # impute_method 保留为普通列，不作为索引
    idx = df_long.set_index(GROUP_KEYS).index
    df_indexed = df_long.set_index(GROUP_KEYS)
    merged = df_indexed.join(anchor, on=GROUP_KEYS, how="inner").reset_index()

    # 去掉 anchor 自身（ΔIE = 0）
    merged = merged[merged["impute_method"] != anchor_method]

    delta_ie = merged["ie_mae"] - merged["ie_anchor"]
    delta_fe = merged["fe_mae"] - merged["fe_anchor"]

    # 分母过滤（与论文主实验一致）
    valid_mask = np.abs(delta_ie) > epsilon
    near_zero = (~valid_mask).sum()
    total = len(valid_mask)

    tau = (delta_fe[valid_mask] / delta_ie[valid_mask]).replace([np.inf, -np.inf], np.nan).dropna()

    n_valid = len(tau)
    tau_median = tau.median()
    tau_iqr = tau.quantile(0.75) - tau.quantile(0.25)

    # 纯倒挂：ΔIE < 0（φ 比 anchor 插补更好）且 ΔFE > 0（预测反而更差）
    # 即"插补更优但预测更差"——性能倒挂的核心定义
    valid_delta_ie = delta_ie[valid_mask]
    valid_delta_fe = delta_fe[valid_mask]
    pure_reversal_mask = (valid_delta_ie < 0) & (valid_delta_fe > 0)
    # 纯倒挂率以有效样本数 N_valid 为分母（与论文主实验口径一致）
    pure_reversal_rate = pure_reversal_mask.sum() / n_valid * 100

    # τ < 0 的全倒挂率（同样以 N_valid 为分母）
    tau_neg_rate = (tau < 0).sum() / n_valid * 100

    return {
        "N_total": total,
        "N_valid": n_valid,
        "near_zero_count": near_zero,
        "near_zero_pct": near_zero / total * 100,
        "tau_median": tau_median,
        "tau_iqr": tau_iqr,
        "pure_reversal_rate": pure_reversal_rate,
        "tau_neg_rate": tau_neg_rate,
        "tau": tau,
    }


def fmt(x, decimals=4):
    return f"{x:.{decimals}f}"


# ── 计算 ───────────────────────────────────────────────

stats_mean = compute_tau_stats(df, "mean")
stats_spline = compute_tau_stats(df, "spline")

# ── 打印报告 ───────────────────────────────────────────

SEP = "=" * 80
DASH = "-" * 80

lines = []
lines.append(SEP)
lines.append("          基线锚点 (Baseline Anchor) 敏感性分析报告")
lines.append(SEP)
lines.append("")
lines.append(f"数据来源: {SRC.relative_to(REPO_ROOT).as_posix()}")
lines.append(f"数值稳定性阈值 ε = {EPSILON}")
lines.append("")
lines.append(DASH)
lines.append(f"{'指标 / 统计项':<38} {'Mean 锚点 (原论文)':>22} {'Spline 锚点 (新验证)':>22}")
lines.append(DASH)

rows = [
    ("有效评估样本量 (N)", f"{stats_mean['N_valid']:,}", f"{stats_spline['N_valid']:,}"),
    ("总评估样本量 (含分母趋零)", f"{stats_mean['N_total']:,}", f"{stats_spline['N_total']:,}"),
    ("Tau 中位数", fmt(stats_mean["tau_median"]), fmt(stats_spline["tau_median"])),
    ("Tau 四分位距 (IQR)", fmt(stats_mean["tau_iqr"]), fmt(stats_spline["tau_iqr"])),
    ("Tau 均值 (参考，受重尾影响)", fmt(stats_mean["tau"].mean()), fmt(stats_spline["tau"].mean())),
    ("纯倒挂率 (%) (ΔIE<0 且 ΔFE>0)", f"{stats_mean['pure_reversal_rate']:.2f}%", f"{stats_spline['pure_reversal_rate']:.2f}%"),
    ("全倒挂率 (%) (τ < 0)", f"{stats_mean['tau_neg_rate']:.2f}%", f"{stats_spline['tau_neg_rate']:.2f}%"),
    ("接近零点样本占比 (|ΔIE|<0.01)", f"{stats_mean['near_zero_pct']:.2f}%", f"{stats_spline['near_zero_pct']:.2f}%"),
]

for label, mean_val, spline_val in rows:
    lines.append(f"{label:<38} {mean_val:>22} {spline_val:>22}")

lines.append(DASH)
lines.append("")

# ── 结论 ───────────────────────────────────────────────

median_close_to_zero = abs(stats_spline["tau_median"]) < 0.1
reversal_significant = stats_spline["pure_reversal_rate"] > 10

lines.append("结论验证：")
lines.append(
    f"1. 切换为 Spline 锚点后，Tau 中位数是否依然贴近零点: "
    f"[{'是' if median_close_to_zero else '否'}]  "
    f"(Spline τ 中位数 = {fmt(stats_spline['tau_median'])})"
)
lines.append(
    f"2. 倒挂现象是否依然显著存在: "
    f"[{'是' if reversal_significant else '否'}]  "
    f"(Spline 纯倒挂率 = {stats_spline['pure_reversal_rate']:.2f}%)"
)
lines.append("")
lines.append("稳健性判断：")
delta_median = stats_spline["tau_median"] - stats_mean["tau_median"]
lines.append(
    f"  - Tau 中位数变化量: {fmt(delta_median)}  "
    f"({'增幅' if delta_median > 0 else '降幅'}，相对变化 {abs(delta_median)/stats_mean['tau_median']*100:.1f}%)"
)
lines.append(
    f"  - 纯倒挂率变化量: {stats_spline['pure_reversal_rate'] - stats_mean['pure_reversal_rate']:+.2f} pp"
)
lines.append("")
if median_close_to_zero and reversal_significant:
    lines.append(
        "  => 核心结论稳健：无论以 Mean 还是 Spline 为锚点，"
    )
    lines.append(
        "    τ 中位数均贴近零点，且纯倒挂率均超过 10%，"
    )
    lines.append(
        "    '弱传导'与'性能倒挂普遍存在'的结论不依赖于基线选择。"
    )
else:
    lines.append("  => 警告：核心结论对基线选择敏感，需在论文中补充说明。")
lines.append("")
lines.append(SEP)

report = "\n".join(lines)
print(report)

# ── 写入文件 ───────────────────────────────────────────
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n报告已保存至: {OUT_FILE}")

# ── 额外输出：τ 分布对比 CSV ──────────────────────────
csv_out = os.path.join(OUT_DIR, "anchor_sensitivity_tau_comparison.csv")
cmp_df = pd.DataFrame({
    "tau_mean_anchor": stats_mean["tau"],
    "tau_spline_anchor": stats_spline["tau"],
}).reset_index(drop=True)
cmp_df.to_csv(csv_out, index=False)
print(f"τ 逐样本对比已保存至: {csv_out}")
