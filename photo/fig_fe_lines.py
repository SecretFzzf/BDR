# -*- coding: utf-8 -*-
"""
新图（前置，审稿意见"站得住的部分"）：不同插补方法的下游预测误差 FE 折线图
- 第 4.1 节：不同插补方法对应的折线相互交叉、没有一致的优胜者——
  这是全文最干净、最不依赖 τ 构造的证据，单独即可支撑"更好的插补 ≠ 更好的预测"
- x 轴：缺失率 4 档（10%/30%/50%/70%）；y 轴：FE-MAE（跨数据集/机制）
- 按预测模型分面 3 子图：真实交叉主要出现在特定模型下（如 LSTM 下 Spline 反超 SAITS）
- 用中位数（均值会被 Spline+DLinear+KDD-Beijing 的极端点拉爆，中位数对此稳健，
  这里不是把这些点从数据里"剔除"，只是统计量选了对离群值不敏感的中位数）
- 数据源 results/seed42/tau_results.csv
- 输出 PDF（矢量）+ PNG 到 photo/

[本次修复]
1. 四条线的数值标签原来一律"往上偏移固定 5pt"，缺失率低的几个点四条线本来就
   贴得很近，标签跟着叠在一起认不清。改成按同一 x 位置的四个数值做聚类堆叠：
   离得近的标签自动错开、往上叠放，并配一条细引线指回其对应的点。
2. 图例原来只出现在 LSTM 这一格，DLinear/PatchTST 两格没有，读者得回头看颜色。
   改成整张图共用一个图例，放在三个子图上方。
3. 原代码里 spline+DLinear+KDD-Beijing 的异常点计数（n_anom_total）是把三个模型
   面板的"spline 且 FE>1"都加总，但小字文案写死说这是"Spline+DLinear on
   KDD-Beijing"——如果其它模型面板下也偶然出现 spline 极端值，会被这句话张冠
   李戴地算进去。改成直接按 model=='DLinear' & impute_method=='spline' &
   dataset=='KDD-Beijing' 精确过滤，文案和数字才对得上。
4. 底部小字大幅精简（原因见下方说明），只保留图内必须自解释的一句话；完整的
   方法学说明（为什么用中位数、异常点怎么处理）放去 LaTeX 图注/正文，不再塞进
   图片像素里。
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
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

missing_rates = sorted(tau_df["missing_rate"].unique())
model_order = ["LSTM", "DLinear", "PatchTST"]
display_names = {
    "LSTM": "LSTM",
    "DLinear": "Linear (per-channel)",
    "PatchTST": "PatchTST (simplified)",
}
impute_order = ["saits", "knn", "brits", "spline"]
colors = {"saits": "#2ca02c", "knn": "#1f77b4", "brits": "#ff7f0e", "spline": "#d62728"}
markers = {"saits": "o", "knn": "s", "brits": "^", "spline": "D"}

# 精确定位小字里要引用的那一个异常条件：DLinear x spline x KDD-Beijing
anom_mask = (
    (tau_df["model"] == "DLinear")
    & (tau_df["impute_method"] == "spline")
    & (tau_df["dataset"] == "KDD-Beijing")
    & (tau_df["fe_mae"] > 1.0)
)
n_anom = int(anom_mask.sum())
anom_max = tau_df.loc[anom_mask, "fe_mae"].max() if n_anom else float("nan")

# 先算出所有面板的中位数，得到全局 y 范围，用来定义"标签算不算挤在一起"的阈值
all_meds = {}
for model in model_order:
    sub = tau_df[tau_df["model"] == model]
    all_meds[model] = {
        im: [
            sub.loc[
                (sub["impute_method"] == im) & (sub["missing_rate"] == r), "fe_mae"
            ].median()
            for r in missing_rates
        ]
        for im in impute_order
    }
y_all = [v for m in all_meds.values() for series in m.values() for v in series]
y_range = max(y_all) - min(y_all)
cluster_gap = 0.045 * y_range  # 两个数值差小于这个阈值就算"挤在一起"
base_gap = 0.05 * y_range  # 单个标签相对于自己的点，默认往上留的间距
stack_gap = 0.11 * y_range  # 堆叠标签之间的最小间距


def stacked_label_positions(values):
    """给同一 x 位置上的若干个 y 值分配不重叠的标签 y 坐标：
    从低到高排序后，若相邻两点本身离得很近就顺势往上堆叠，
    离得远的保持贴近自己的点，返回 {原始索引: label_y}。"""
    order = sorted(range(len(values)), key=lambda k: values[k])
    label_y = {}
    prev = None
    for k in order:
        y = values[k] + base_gap
        if prev is not None and y < prev + stack_gap:
            y = prev + stack_gap
        label_y[k] = y
        prev = y
    return label_y


fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.15, wspace=0.08)

x = np.arange(len(missing_rates))
legend_handles = []
for ax, model in zip(axes, model_order):
    med = all_meds[model]

    for im in impute_order:
        (line,) = ax.plot(
            x,
            med[im],
            marker=markers[im],
            color=colors[im],
            linewidth=1.8,
            markersize=6,
            label=im.capitalize(),
            zorder=5,
        )
        if model == model_order[0]:
            legend_handles.append(line)

    # 每个 x 位置单独处理四个数值的标签堆叠，离得近的自动错开
    for i in range(len(missing_rates)):
        values = [med[im][i] for im in impute_order]
        label_y = stacked_label_positions(values)
        for k, im in enumerate(impute_order):
            y = values[k]
            ly = label_y[k]
            if ly - y > base_gap + 1e-9:
                # 标签被明显往上推挤过，画一条细引线标明归属
                ax.plot(
                    [x[i], x[i]],
                    [y + base_gap * 0.3, ly - base_gap * 0.3],
                    color=colors[im],
                    linewidth=0.6,
                    alpha=0.5,
                    zorder=4,
                )
            ax.annotate(
                f"{y:.2f}",
                xy=(x[i], ly),
                ha="center",
                va="bottom",
                fontsize=7,
                color=colors[im],
                zorder=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r * 100)}%" for r in missing_rates])
    ax.set_xlim(-0.3, len(missing_rates) - 1 + 0.3)
    ax.set_xlabel("Missing Rate")
    ax.set_title(display_names[model])

# 顶部留白多留了一点，给堆叠后的标签一些安全边距
ymax_data = max(y_all)
ymin_data = min(y_all)
for ax in axes:
    ax.set_ylim(ymin_data - 0.06 * y_range, ymax_data + 0.62 * y_range)

axes[0].set_ylabel("Prediction Error (FE-MAE, median)")

# 整张图共用一个图例，放在三个子图上方，不再只出现在 LSTM 那一格
fig.legend(
    legend_handles,
    [h.get_label() for h in legend_handles],
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 0.98),
    frameon=True,
    facecolor="white",
    framealpha=0.9,
)

# 底部小字（已注释）：完整的方法学解释放进 LaTeX 图注/正文，不再塞进图片本身
# note = f"FE-MAE median across datasets/mechanisms. n={n_anom} extreme DLinear+Spline+KDD-Beijing points (up to {anom_max:.0f}) are why medians, not means, are used."
# fig.text(0.01, 0.015, note, fontsize=7.5, va="bottom", ha="left")

fig.savefig(OUT / "fig_fe_lines.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_fe_lines.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("[OK] FE lines (faceted) saved to", OUT)
print(f"  DLinear+Spline+KDD-Beijing anomalies: n={n_anom}, max={anom_max:.2f}")
