# -*- coding: utf-8 -*-
"""
New figure 4: 3x4 true-inversion-rate heatmap of prediction model x imputation method (replaces the previous boxplot by missing mechanism)
- Reviewer comment 5.4: Section 4.5 of the main text claims "inversions concentrate on specific combinations", but Table 3 only has marginal distributions,
  and lacks an interactive "prediction model x imputation method" plot -> this figure directly shows the true-inversion rate
  of each combination (proportion of tau<0 & delta_IE<0 & delta_FE>0 among the valid-tau samples of that combination)
- Data source: D:/ei/result/seed42/tau_results.csv (valid tau = 2064)
- Outputs PDF (vector) + PNG to D:/ei/mended_photo/
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

ROOT = Path(r"D:/ei/result")
OUT = Path(r"D:/ei/mended_photo")
OUT.mkdir(parents=True, exist_ok=True)

tau_df = pd.read_csv(ROOT / "seed42" / "tau_results.csv")
valid = tau_df.dropna(subset=["tau"]).copy()
valid["is_true_inv"] = (
    (valid["tau"] < 0) & (valid["delta_ie"] < 0) & (valid["delta_fe"] > 0)
)

# Rows: prediction models (LSTM on top, matching the focus of the main text); columns: imputation methods (Mean is the tau baseline anchor, so not in the columns)
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

# vmax dynamically follows the actual data (rounded up to a multiple of 10, and at least 50), avoiding the highest
# few combinations being clipped to the same color because they hit the hard-coded color-scale upper limit and losing
# their distinctiveness -- and those few combinations are precisely the "inversion concentration" that this figure and
# its caption want to emphasize.
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

# In-cell annotation: inversion rate + sample count
# Text color black/white is auto-decided by the luminance of the cell's actual color to judge contrast, rather than a
# hard-coded percentage threshold -- this way the threshold does not drift when vmax changes above.
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
