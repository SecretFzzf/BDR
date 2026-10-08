# -*- coding: utf-8 -*-
"""
新图4：预测模型 × 插补方法 3×4 倒挂率热力图（替代原按缺失机制的箱线图）
- 审稿意见 5.4：正文 4.5 节称"倒挂集中于特定组合"，但表 3 只有各自边缘分布，
  缺少「预测模型 × 插补方法」交互图 → 本图直接展示每个组合的真倒挂率
  (τ<0 & ΔIE<0 & ΔFE>0 占该组合有效 τ 样本的比例)
- 数据源 results/seed42/tau_results.csv（有效 τ = 2064）
- 输出 PDF（矢量）+ PNG 到 photo/
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    }
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
ROOT = REPO_ROOT / "results"
OUT = BASE_DIR
OUT.mkdir(parents=True, exist_ok=True)

tau_df = pd.read_csv(ROOT / "seed42" / "tau_results.csv")
valid = tau_df.dropna(subset=["tau"]).copy()
valid["is_true_inv"] = (
    (valid["tau"] < 0) & (valid["delta_ie"] < 0) & (valid["delta_fe"] > 0)
)

# 行：预测模型（LSTM 靠上，对应正文关注）；列：插补方法（Mean 是 τ 的锚点基线，不在列中）
model_order = ["LSTM", "DLinear", "PatchTST"]
display_names = {
    "LSTM": "LSTM",
    "DLinear": "Linear (per-channel)",
    "PatchTST": "PatchTST (simplified)",
}
impute_order = ["knn", "brits", "saits", "spline"]

rate = valid.groupby(["model", "impute_method"])["is_true_inv"].mean().mul(100)
n_cell = valid.groupby(["model", "impute_method"])["is_true_inv"].count()

rate_mat = np.array([[rate.loc[m, im] for im in impute_order] for m in model_order])
n_mat = np.array([[n_cell.loc[m, im] for im in impute_order] for m in model_order])
overall = valid["is_true_inv"].mean() * 100

# vmax 动态跟随实际数据（向上取整到 10 的倍数，且不低于 50），避免最高的几个
# 组合因为顶到写死的色阶上限而被 clip 成同一种颜色、失去彼此的区分度——
# 而这几个组合恰恰是本图和图注想强调的"倒挂集中"的重点。
vmax = max(50, int(np.ceil(rate_mat.max() / 10.0)) * 10)

fig, ax = plt.subplots(figsize=(9, 4.6))
fig.subplots_adjust(left=0.20, right=0.90, top=0.88, bottom=0.22)

sns.heatmap(
    rate_mat,
    ax=ax,
    annot=False,
    cmap="YlOrRd",
    vmin=0,
    vmax=vmax,
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "True Inversion Rate (%)", "shrink": 0.8},
)

# 单元格内标注：倒挂率 + 样本数
# 文字黑/白按该单元格实际颜色的亮度自动判断对比度，而不是写死一个百分比
# 阈值——这样阈值不会因为上面 vmax 变化而错位。
cmap = plt.get_cmap("YlOrRd")
norm = plt.Normalize(vmin=0, vmax=vmax)
for i in range(len(model_order)):
    for j in range(len(impute_order)):
        r, g, b, _ = cmap(norm(rate_mat[i, j]))
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        ax.text(
            j + 0.5,
            i + 0.5,
            f"{rate_mat[i, j]:.1f}%\n(n={n_mat[i, j]})",
            ha="center",
            va="center",
            fontsize=9,
            color="black" if luminance > 0.6 else "white",
        )

ax.set_xticks(np.arange(len(impute_order)) + 0.5)
ax.set_xticklabels(impute_order)
ax.set_yticks(np.arange(len(model_order)) + 0.5)
ax.set_yticklabels([display_names[m] for m in model_order], rotation=0)
ax.set_xlabel("Imputation Method")
ax.set_ylabel("Predictive Model")
ax.set_title("True Inversion Rate by (Predictive Model x Imputation Method)")

# note = (
#     f"True inversion: tau<0 & delta_IE<0 & delta_FE>0. Mean is the tau baseline anchor, "
#     f"hence not a column. Overall inversion rate = {overall:.1f}% across {len(valid)} valid-tau samples. "
#     f"Inversion concentrates on specific combos: PatchTST (simplified)+brits {rate.loc['PatchTST','brits']:.1f}%, "
#     f"Linear (per-channel)+spline {rate.loc['DLinear','spline']:.1f}%, LSTM+brits {rate.loc['LSTM','brits']:.1f}%."
# )
# fig.text(0.01, 0.02, note, fontsize=7.5, va="bottom", ha="left", wrap=True)

fig.savefig(OUT / "fig4_inversion_heatmap.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig4_inversion_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("[OK] fig4 heatmap saved to", OUT)
print("model | " + " | ".join(impute_order))
for i, m in enumerate(model_order):
    print(
        f"{m} | "
        + " | ".join(f"{rate_mat[i, j]:.1f}%" for j in range(len(impute_order)))
    )
print(f"overall inversion rate = {overall:.2f}%")
