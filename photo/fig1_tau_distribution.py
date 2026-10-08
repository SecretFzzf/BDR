# -*- coding: utf-8 -*-
"""
新图1：ΔIE–ΔFE 四象限散点图（替代原 τ 分布直方图）
- 四个象限天然对应四类情形：真倒挂 / 反向噪声 / 正常传导(双更优) / 正常传导(双更差)
- τ = ΔFE/ΔIE 为过原点射线的斜率，用灰色虚线画出若干参考斜率
- 中轴灰色竖带表示 ε=0.01 排除区（|ΔIE| ≤ ε 时 τ 无定义）
- 输出 PDF（矢量）+ PNG 到 photo/

[视觉修复说明]
1. τ 参考线标签坐标改为"与绘图框边界的真实交点"，不再出现算出来的
   坐标落在坐标轴范围之外的情况（原代码对 τ=1.0 / τ=-1.0 会算出
   y=±1.2，远超出 y 轴的 [-0.3, 0.8]，导致 bbox_inches="tight" 保存
   时把画布强行撑大，出现大片空白、标题错位）。
2. 图例移到绘图区外右侧，不再遮住"True Inversion"象限里的散点和文字。
3. 角标只保留类别名，不再和图例重复打印 count/百分比，减少拥挤。
4. 所有文字标注加半透明白底，避免和背景色块/散点/其它文字糊在一起。
5. 用 constrained_layout 替代手动 tight_layout(rect=...)，标题、双行轴
   标签、底部脚注不再互相挤压。
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.patches import Rectangle

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
valid = tau_df.dropna(subset=["tau"])
excluded = tau_df[tau_df["tau"].isna()]
n_valid = len(valid)
tau_median = np.nanmedian(valid["tau"])

di = valid["delta_ie"].values
df = valid["delta_fe"].values

# 四象限归属（ΔIE<0 表示插补更优，ΔFE<0 表示预测更优）
q1 = (di < 0) & (df > 0)  # 真倒挂
q2 = (di < 0) & (df < 0)  # 正常：插补与预测同时更好
q3 = (di > 0) & (df < 0)  # 反向噪声
q4 = (di > 0) & (df > 0)  # 正常：插补与预测同时更差
n_q1, n_q2, n_q3, n_q4 = int(q1.sum()), int(q2.sum()), int(q3.sum()), int(q4.sum())
n_ex = len(excluded)
n_normal = n_q2 + n_q4

# 显示范围（ΔFE 上界截断 KDD-Beijing 的极端发散点）
x_lo, x_hi = -1.2, 0.3
y_lo, y_hi = -0.3, 0.8
eps = 0.01
n_cut = int(((df < y_lo) | (df > y_hi)).sum())
n_cut_pct = n_cut / n_valid * 100

# 图更宽一些，给移到图外的图例留位置；底部/顶部留够边距给脚注和标题
fig, ax = plt.subplots(figsize=(10.5, 6.4))
fig.subplots_adjust(left=0.08, right=0.76, top=0.92, bottom=0.14)

label_bbox = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5)

# 象限底色
ax.add_patch(Rectangle((x_lo, 0), -x_lo, y_hi, color="#ff7f0e", alpha=0.12, zorder=0))
ax.add_patch(
    Rectangle((x_lo, y_lo), -x_lo, -y_lo, color="#1f77b4", alpha=0.12, zorder=0)
)
ax.add_patch(Rectangle((0, y_lo), x_hi, -y_lo, color="#d62728", alpha=0.12, zorder=0))
ax.add_patch(Rectangle((0, 0), x_hi, y_hi, color="#1f77b4", alpha=0.12, zorder=0))

# ε 排除带（|ΔIE| ≤ ε 时 τ 无定义）
ax.axvspan(-eps, eps, color="#999999", alpha=0.30, zorder=1)
ax.text(
    0,
    0.55,
    f"excluded: {n_ex}\n(|$\\Delta$IE| <= $\\epsilon$={eps})",
    ha="center",
    va="center",
    fontsize=7.5,
    color="#555555",
    zorder=4,
    bbox=label_bbox,
    clip_on=True,
)

# 原点轴线
ax.axhline(0, color="black", lw=0.9, zorder=2)
ax.axvline(0, color="black", lw=0.9, zorder=2)


def edge_point(t, x_lo, x_hi, y_lo, y_hi, margin=0.04):
    """在 x=x_lo 处沿斜率 t 求 y；若越界则改用与上/下边界的真实交点，
    并向框内收进 margin，保证标签始终落在坐标轴范围内。"""
    y = t * x_lo
    if y < y_lo:
        y = y_lo + margin
        x = y / t
    elif y > y_hi:
        y = y_hi - margin
        x = y / t
    else:
        x = x_lo + margin
        y = t * x
    x = min(max(x, x_lo), x_hi)
    return x, y


# τ 斜率参考线（过原点射线，τ = ΔFE/ΔIE），标签坐标限制在绘图框内
for t in [0.05, 0.2, 1.0, -0.2, -1.0]:
    xs = np.linspace(x_lo, x_hi, 2)
    ax.plot(
        xs,
        t * xs,
        color="grey",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        zorder=2,
        clip_on=True,
    )
    lx, ly = edge_point(t, x_lo, x_hi, y_lo, y_hi)
    ax.text(
        lx,
        ly,
        f"τ={t:g}",
        fontsize=7.5,
        color="grey",
        zorder=4,
        bbox=label_bbox,
        clip_on=True,
    )

# 散点
ax.scatter(
    di[q2],
    df[q2],
    s=14,
    c="#1f77b4",
    alpha=0.65,
    edgecolors="none",
    zorder=3,
    label=f"Normal ($\\tau$ > 0): {n_normal} ({n_normal/n_valid*100:.1f}%)",
)
ax.scatter(di[q4], df[q4], s=14, c="#1f77b4", alpha=0.65, edgecolors="none", zorder=3)
ax.scatter(
    di[q1],
    df[q1],
    s=14,
    c="#ff7f0e",
    alpha=0.65,
    edgecolors="none",
    zorder=3,
    label=f"True Inversion ($\\Delta$IE<0, $\\Delta$FE>0): {n_q1} ({n_q1/n_valid*100:.1f}%)",
)
ax.scatter(
    di[q3],
    df[q3],
    s=14,
    c="#d62728",
    alpha=0.65,
    edgecolors="none",
    zorder=3,
    label=f"Reverse Noise ($\\Delta$IE>0, $\\Delta$FE<0): {n_q3} ({n_q3/n_valid*100:.1f}%)",
)
ax.scatter(
    excluded["delta_ie"],
    excluded["delta_fe"],
    s=8,
    c="#999999",
    alpha=0.5,
    edgecolors="none",
    zorder=3,
    label=f"Excluded (|$\\Delta$IE| <= $\\epsilon$): {n_ex} ({n_ex/len(tau_df)*100:.1f}% of all)",
)

# 象限角标：只保留类别名 + 简短占比，字号更小、留在角落、加白底，
# 具体计数交给图例（避免两处重复打印同一组数字）
ax.text(
    x_lo + 0.03,
    y_hi - 0.03,
    f"True Inversion ({n_q1/n_valid*100:.1f}%)",
    fontsize=8,
    va="top",
    ha="left",
    color="#b36b00",
    zorder=4,
    bbox=label_bbox,
)
ax.text(
    x_lo + 0.03,
    y_lo + 0.03,
    f"Both Better ({n_q2/n_valid*100:.1f}%)",
    fontsize=8,
    va="bottom",
    ha="left",
    color="#1a4f8a",
    zorder=4,
    bbox=label_bbox,
)
ax.text(
    x_hi - 0.03,
    y_lo + 0.03,
    f"Reverse Noise ({n_q3/n_valid*100:.1f}%)",
    fontsize=8,
    va="bottom",
    ha="right",
    color="#a31616",
    zorder=4,
    bbox=label_bbox,
)
ax.text(
    x_hi - 0.03,
    y_hi - 0.03,
    f"Both Worse ({n_q4/n_valid*100:.1f}%)",
    fontsize=8,
    va="top",
    ha="right",
    color="#1a4f8a",
    zorder=4,
    bbox=label_bbox,
)

ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel(
    "$\\Delta$IE = IE($\\varphi$) $-$ IE(Mean)  (lower = better imputation)", labelpad=8
)
ax.set_ylabel("$\\Delta$FE = FE($\\varphi$) $-$ FE(Mean)\n(lower = better forecast)")
ax.set_title(
    "Quadrant Decomposition of Transfer Efficiency $\\tau$ ($\\tau$ = $\\Delta$FE/$\\Delta$IE)"
)

# 图例移到绘图区外右侧，不再遮住数据和角标
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    frameon=True,
    facecolor="white",
    framealpha=0.9,
    fontsize=8.5,
    borderaxespad=0.0,
)

# note = (
#     f"valid $\\tau$ = {n_valid} | Median $\\tau$ = {tau_median:.4f} | "
#     f"display truncated to $\\Delta$FE in [{y_lo:.1f}, {y_hi:.1f}]; "
#     f"{n_cut} pts ({n_cut_pct:.1f}%) with $\\Delta$FE > {y_hi} not shown (all from KDD-Beijing)"
# )
# fig.text(0.01, 0.015, note, fontsize=7.5, va="bottom", ha="left")

fig.savefig(OUT / "fig1_tau_distribution.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1_tau_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(
    f"[OK] fig1 saved to {OUT} | normal={n_normal} ({n_normal/n_valid*100:.1f}%) | "
    f"true_inv={n_q1} ({n_q1/n_valid*100:.1f}%) | rev_noise={n_q3} ({n_q3/n_valid*100:.1f}%) | "
    f"excluded={n_ex} | median={tau_median:.4f}"
)
