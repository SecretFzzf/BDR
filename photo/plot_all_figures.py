# ==================== fig1_tau_distribution.py ====================

# -*- coding: utf-8 -*-
""""""

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

# （ΔIE<0 ，ΔFE<0 ）
q1 = (di < 0) & (df > 0)  # 
q2 = (di < 0) & (df < 0)  # ：
q3 = (di > 0) & (df < 0)  # 
q4 = (di > 0) & (df > 0)  # ：
n_q1, n_q2, n_q3, n_q4 = int(q1.sum()), int(q2.sum()), int(q3.sum()), int(q4.sum())
n_ex = len(excluded)
n_normal = n_q2 + n_q4

# （ΔFE  KDD-Beijing ）
x_lo, x_hi = -1.2, 0.3
y_lo, y_hi = -0.3, 0.8
eps = 0.01
n_cut = int(((df < y_lo) | (df > y_hi)).sum())
n_cut_pct = n_cut / n_valid * 100

# ，；/
fig, ax = plt.subplots(figsize=(10.5, 6.4))
fig.subplots_adjust(left=0.08, right=0.76, top=0.92, bottom=0.14)

label_bbox = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5)

# 
ax.add_patch(Rectangle((x_lo, 0), -x_lo, y_hi, color="#ff7f0e", alpha=0.12, zorder=0))
ax.add_patch(
    Rectangle((x_lo, y_lo), -x_lo, -y_lo, color="#1f77b4", alpha=0.12, zorder=0)
)
ax.add_patch(Rectangle((0, y_lo), x_hi, -y_lo, color="#d62728", alpha=0.12, zorder=0))
ax.add_patch(Rectangle((0, 0), x_hi, y_hi, color="#1f77b4", alpha=0.12, zorder=0))

# ε （|ΔIE| ≤ ε  τ ）
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

# 
ax.axhline(0, color="black", lw=0.9, zorder=2)
ax.axvline(0, color="black", lw=0.9, zorder=2)


def edge_point(t, x_lo, x_hi, y_lo, y_hi, margin=0.04):
    """"""
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


# τ （，τ = ΔFE/ΔIE），
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

# 
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

# ： + ，、、，
# （）
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

# ，
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


# ==================== fig4_inversion_heatmap.py ====================

# -*- coding: utf-8 -*-
""""""

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

# ：（LSTM ，）；：（Mean  τ ，）
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

# vmax （ 10 ， 50），
#  clip 、——
# ""。
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

# ： + 
# /，
# —— vmax 。
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


# ==================== fig_fe_lines.py ====================

# -*- coding: utf-8 -*-
""""""

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

# ：DLinear x spline x KDD-Beijing
anom_mask = (
    (tau_df["model"] == "DLinear")
    & (tau_df["impute_method"] == "spline")
    & (tau_df["dataset"] == "KDD-Beijing")
    & (tau_df["fe_mae"] > 1.0)
)
n_anom = int(anom_mask.sum())
anom_max = tau_df.loc[anom_mask, "fe_mae"].max() if n_anom else float("nan")

# ， y ，""
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
cluster_gap = 0.045 * y_range  # ""
base_gap = 0.05 * y_range  # ，
stack_gap = 0.11 * y_range  # 


def stacked_label_positions(values):
    """"""
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

    #  x ，
    for i in range(len(missing_rates)):
        values = [med[im][i] for im in impute_order]
        label_y = stacked_label_positions(values)
        for k, im in enumerate(impute_order):
            y = values[k]
            ly = label_y[k]
            if ly - y > base_gap + 1e-9:
                # ，
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

# ，
ymax_data = max(y_all)
ymin_data = min(y_all)
for ax in axes:
    ax.set_ylim(ymin_data - 0.06 * y_range, ymax_data + 0.62 * y_range)

axes[0].set_ylabel("Prediction Error (FE-MAE, median)")

# ，， LSTM 
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

# （）： LaTeX /，
# note = f"FE-MAE median across datasets/mechanisms. n={n_anom} extreme DLinear+Spline+KDD-Beijing points (up to {anom_max:.0f}) are why medians, not means, are used."
# fig.text(0.01, 0.015, note, fontsize=7.5, va="bottom", ha="left")

fig.savefig(OUT / "fig_fe_lines.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_fe_lines.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("[OK] FE lines (faceted) saved to", OUT)
print(f"  DLinear+Spline+KDD-Beijing anomalies: n={n_anom}, max={anom_max:.2f}")


# ==================== fig_fe_lines_pooled.py ====================

# -*- coding: utf-8 -*-
""""""

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
