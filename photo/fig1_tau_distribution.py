# -*- coding: utf-8 -*-
"""
New figure 1: DeltaIE - DeltaFE four-quadrant scatter plot (replaces the previous tau distribution histogram)
- The four quadrants naturally correspond to four cases: true inversion / reverse noise / normal transfer (both better) / normal transfer (both worse)
- tau = DeltaFE/DeltaIE is the slope of a ray through the origin; several reference slopes are drawn as grey dashed lines
- The central grey vertical band indicates the epsilon=0.01 exclusion zone (tau is undefined when |DeltaIE| <= epsilon)
- Outputs PDF (vector) + PNG to D:/ei/mended_photo/

[Visual fix notes]
1. The tau reference line label coordinates now use "the true intersection with the plot-frame boundary", eliminating the previous
   situation where the computed coordinates fell outside the axis range (the original code would compute y=+/-1.2 for tau=1.0 / tau=-1.0,
   which far exceeds the y-axis range [-0.3, 0.8], so bbox_inches="tight" when saving forced the canvas to grow, producing large empty
   areas and misaligned titles).
2. The legend is moved outside the plot area to the right, so it no longer covers the scatter points and text inside the "True Inversion" quadrant.
3. The corner labels now keep only the category name, no longer duplicating the count/percentage from the legend, reducing clutter.
4. All text annotations have a semi-transparent white background to avoid blending with background color blocks / scatter points / other text.
5. constrained_layout is used instead of the manual tight_layout(rect=...), so titles, two-line axis labels, and the bottom footnote no
   longer squeeze each other.
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

ROOT = Path(r"D:/ei/result")
OUT = Path(r"D:/ei/mended_photo")
OUT.mkdir(parents=True, exist_ok=True)

tau_df = pd.read_csv(ROOT / "seed42" / "tau_results.csv")
valid = tau_df.dropna(subset=["tau"])
excluded = tau_df[tau_df["tau"].isna()]
n_valid = len(valid)
tau_median = np.nanmedian(valid["tau"])

di = valid["delta_ie"].values
df = valid["delta_fe"].values

# Four-quadrant assignment (DeltaIE<0 means imputation is better, DeltaFE<0 means forecast is better)
q1 = (di < 0) & (df > 0)  # true inversion
q2 = (di < 0) & (df < 0)  # normal: imputation and forecast are both better
q3 = (di > 0) & (df < 0)  # reverse noise
q4 = (di > 0) & (df > 0)  # normal: imputation and forecast are both worse
n_q1, n_q2, n_q3, n_q4 = int(q1.sum()), int(q2.sum()), int(q3.sum()), int(q4.sum())
n_ex = len(excluded)
n_normal = n_q2 + n_q4

# Display range (upper bound of DeltaFE truncates the extreme divergent KDD-Beijing points)
x_lo, x_hi = -1.2, 0.3
y_lo, y_hi = -0.3, 0.8
eps = 0.01
n_cut = int(((df < y_lo) | (df > y_hi)).sum())
n_cut_pct = n_cut / n_valid * 100

# The figure is a bit wider, to leave room for the legend moved outside the plot; bottom/top margins are generous enough for the footnote and the title
fig, ax = plt.subplots(figsize=(10.5, 6.4))
fig.subplots_adjust(left=0.08, right=0.76, top=0.92, bottom=0.14)

label_bbox = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5)

# Quadrant background colors
ax.add_patch(Rectangle((x_lo, 0), -x_lo, y_hi, color="#ff7f0e", alpha=0.12, zorder=0))
ax.add_patch(
    Rectangle((x_lo, y_lo), -x_lo, -y_lo, color="#1f77b4", alpha=0.12, zorder=0)
)
ax.add_patch(Rectangle((0, y_lo), x_hi, -y_lo, color="#d62728", alpha=0.12, zorder=0))
ax.add_patch(Rectangle((0, 0), x_hi, y_hi, color="#1f77b4", alpha=0.12, zorder=0))

# epsilon exclusion band (tau is undefined when |DeltaIE| <= epsilon)
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

# Origin axes
ax.axhline(0, color="black", lw=0.9, zorder=2)
ax.axvline(0, color="black", lw=0.9, zorder=2)


def edge_point(t, x_lo, x_hi, y_lo, y_hi, margin=0.04):
    """Compute y at x=x_lo along slope t; if out of range, fall back to the true intersection with the top/bottom boundary,
    and shrink inward by margin so the label always stays inside the axis range."""
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


# tau slope reference lines (rays through the origin, tau = DeltaFE/DeltaIE); label coordinates constrained to the plot frame
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

# Scatter points
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

# Quadrant corner labels: keep only the category name + short percentage, smaller font, positioned in the corners, with white background,
# leaving the exact counts to the legend (avoid duplicating the same numbers in two places)
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

# Legend moved outside the plot area to the right, so it no longer covers the data and the corner labels
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
