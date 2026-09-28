# -*- coding: utf-8 -*-
"""
New figure (front-matter, "the parts that hold up under reviewer scrutiny"): downstream forecast error FE line plots for different imputation methods
- Section 4.1: the line plots corresponding to different imputation methods cross each other, with no consistent winner —
  this is the cleanest evidence in the whole paper that does not depend on the τ construction, and by itself it supports "better imputation ≠ better prediction"
- x-axis: 4 missing-rate levels (10%/30%/50%/70%); y-axis: FE-MAE (across datasets/mechanisms)
- Faceted into 3 subplots by forecast model: the real crossings mainly appear under specific models (e.g. Spline overtakes SAITS under LSTM)
- Uses the median (the mean gets blown up by the Spline+DLinear+KDD-Beijing extreme points; the median is robust to them.
  We are not "removing" these points from the data, we just chose a statistic that is insensitive to outliers)
- Data source: D:/ei/result/seed42/tau_results.csv
- Outputs PDF (vector) + PNG to D:/ei/mended_photo/

[Fixes this round]
1. The numeric labels for the four lines were previously all shifted up by a fixed 5pt. At low missing rates the four lines
   are already very close together, so the labels piled up and were unreadable. Changed to cluster the four values at the same
   x position and stack them: labels that are close together are automatically nudged apart and stacked upwards, and a thin
   leader line points back to the corresponding data point.
2. The legend previously only appeared in the LSTM panel; the DLinear/PatchTST panels had none, so readers had to look back
   at the colors. Changed to a single legend shared across the whole figure, placed above the three subplots.
3. The old code counted the anomaly points (n_anom_total) for spline+DLinear+KDD-Beijing by summing up "spline and FE>1"
   across all three model panels, but the small-text caption was hard-coded to say "Spline+DLinear on KDD-Beijing" — if
   other model panels happened to have spline extremes, they'd be incorrectly attributed by that sentence. Changed to filter
   exactly with model=='DLinear' & impute_method=='spline' & dataset=='KDD-Beijing' so the caption matches the number.
4. The bottom small-text caption was substantially simplified (reasons in the note below); only the one self-explanatory
   sentence that must live inside the figure is kept. The full methodological explanation (why median, how anomalies are
   handled) is moved to the LaTeX figure caption / main text and no longer crammed into the pixels of the image.
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

ROOT = Path(r"D:/ei/result")
OUT = Path(r"D:/ei/mended_photo")
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

# Precisely locate the single anomaly condition referenced in the small-text caption: DLinear x spline x KDD-Beijing
anom_mask = (
    (tau_df["model"] == "DLinear")
    & (tau_df["impute_method"] == "spline")
    & (tau_df["dataset"] == "KDD-Beijing")
    & (tau_df["fe_mae"] > 1.0)
)
n_anom = int(anom_mask.sum())
anom_max = tau_df.loc[anom_mask, "fe_mae"].max() if n_anom else float("nan")

# First compute the median for all panels to get the global y range, used to define the "are the labels crowded together" threshold
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
cluster_gap = 0.045 * y_range  # two values differ by less than this threshold are considered "crowded together"
base_gap = 0.05 * y_range  # default spacing a single label leaves above its own point
stack_gap = 0.11 * y_range  # minimum spacing between stacked labels


def stacked_label_positions(values):
    """Assign non-overlapping label y-coordinates for several y values at the same x position:
    after sorting low-to-high, if adjacent points are already very close, push them upwards and stack;
    if they are far apart, keep the label close to its own point. Returns {original index: label_y}."""
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

    # Handle label stacking of the four values at each x position individually; close labels are automatically nudged apart
    for i in range(len(missing_rates)):
        values = [med[im][i] for im in impute_order]
        label_y = stacked_label_positions(values)
        for k, im in enumerate(impute_order):
            y = values[k]
            ly = label_y[k]
            if ly - y > base_gap + 1e-9:
                # The label was pushed noticeably upward; draw a thin leader line to indicate ownership
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

# Left a bit more whitespace at the top to give the stacked labels some safety margin
ymax_data = max(y_all)
ymin_data = min(y_all)
for ax in axes:
    ax.set_ylim(ymin_data - 0.06 * y_range, ymax_data + 0.62 * y_range)

axes[0].set_ylabel("Prediction Error (FE-MAE, median)")

# A single legend shared across the whole figure, placed above the three subplots, not just in the LSTM panel
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

# Bottom small-text caption (already commented out): the full methodological explanation goes into the LaTeX figure caption / main text, no longer crammed into the image itself
# note = f"FE-MAE median across datasets/mechanisms. n={n_anom} extreme DLinear+Spline+KDD-Beijing points (up to {anom_max:.0f}) are why medians, not means, are used."
# fig.text(0.01, 0.015, note, fontsize=7.5, va="bottom", ha="left")

fig.savefig(OUT / "fig_fe_lines.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_fe_lines.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("[OK] FE lines (faceted) saved to", OUT)
print(f"  DLinear+Spline+KDD-Beijing anomalies: n={n_anom}, max={anom_max:.2f}")
