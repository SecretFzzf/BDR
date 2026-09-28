# -*- coding: utf-8 -*-
"""
New figure (version 2, pooled single plot): downstream forecast error FE line plots for different imputation methods
- Coexists with fig_fe_lines.py (faceted into 3 subplots by model) for comparison
- x-axis: 4 missing-rate levels (10%/30%/50%/70%); y-axis: FE-MAE median (pooled across datasets / models / mechanisms)
- Uses the median (the mean gets blown up by the 28 divergent Spline+DLinear+KDD-Beijing points)
- Data source: D:/ei/result/seed42/tau_results.csv
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

missing_rates = sorted(tau_df["missing_rate"].unique())
impute_order = ["saits", "knn", "brits", "spline"]
colors = {"saits": "#2ca02c", "knn": "#1f77b4", "brits": "#ff7f0e", "spline": "#d62728"}
markers = {"saits": "o", "knn": "s", "brits": "^", "spline": "D"}

medians = {
    im: [
        tau_df.loc[
            (tau_df["impute_method"] == im) & (tau_df["missing_rate"] == r), "fe_mae"
        ].median()
        for r in missing_rates
    ]
    for im in impute_order
}

n_spline_anom = int(
    ((tau_df["impute_method"] == "spline") & (tau_df["fe_mae"] > 1.0)).sum()
)
max_fe = tau_df["fe_mae"].max()

fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.13)
x = np.arange(len(missing_rates))

for im in impute_order:
    ax.plot(
        x,
        medians[im],
        marker=markers[im],
        color=colors[im],
        linewidth=2,
        markersize=7,
        label=f"{im.capitalize()} (median FE-MAE)",
        zorder=5,
    )
    for i, v in enumerate(medians[im]):
        ax.annotate(
            f"{v:.3f}",
            xy=(x[i], v),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors[im],
        )

ax.set_xticks(x)
ax.set_xticklabels([f"{int(r * 100)}%" for r in missing_rates])
ax.set_xlim(-0.3, len(missing_rates) - 1 + 0.3)
ax.set_xlabel("Missing Rate")
ax.set_ylabel("Prediction Error (FE-MAE)")
ax.set_title("Downstream Prediction Error by Imputation Method")

# note = (
#     f"FE-MAE median across datasets/models/mechanisms. Lines cross with no consistently "
#     f"best imputer. A dataset-specific anomaly is excluded: Spline + DLinear on KDD-Beijing "
#     f"gives {n_spline_anom} divergent conditions (FE-MAE up to {max_fe:.1f}); under the mean "
#     f"metric these would lift the Spline line far above the rest, so medians are shown "
#     f"(consistent with the IE analysis in Section 4.1)."
# )
# fig.text(0.01, 0.015, note, fontsize=7.5, va="bottom", ha="left", wrap=True)

ax.legend(loc="upper left")

fig.savefig(OUT / "fig_fe_lines_pooled.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_fe_lines_pooled.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("[OK] FE lines (pooled) saved to", OUT)
for im in impute_order:
    print(f"  {im}: " + " | ".join(f"{v:.4f}" for v in medians[im]))
print(f"  spline anomalies (KDD+DLinear, FE>1): {n_spline_anom}, max FE = {max_fe:.1f}")
