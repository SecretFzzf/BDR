"""
baseline anchor point (Baseline Anchor) sensitivity analysis
--------------------------------------
Verification will τ The baseline anchor point is from Mean switch to Spline Finally, is the statistical conclusion of conduction efficiency robust?。

original baseline（Mean anchor point）：
    ΔIE = IE(φ) - IE(Mean)
    ΔFE = FE(φ) - FE(Mean)
    τ = ΔFE / ΔIE

new baseline（Spline anchor point, exclude imputer == 'spline' itself）：
    ΔIE = IE(φ) - IE(Spline)
    ΔFE = FE(φ) - FE(Spline)
    τ = ΔFE / ΔIE
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── path ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SRC = REPO_ROOT / "results" / "seed42" / "all_models_result.csv"
OUT_DIR = REPO_ROOT / "results" / "anchor_sensitivity"
OUT_FILE = OUT_DIR / "anchor_sensitivity_report.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 0.01   # Numerical stability threshold consistent with the main experiment of the paper

# ── Read data ───────────────────────────────────────────
df = pd.read_csv(SRC)

# Keep only rows that run successfully
df = df[df["status"] == "success"].copy()

# Unified missing mechanism naming（MAR_Block / MAR_Block）
df["missing_mode"] = df["missing_mode"].str.replace("MAR_Block", "MAR_Block", regex=False)

# Keep only the columns you need
GROUP_KEYS = ["dataset", "model", "missing_mode", "missing_rate", "pred_len"]
df = df[GROUP_KEYS + ["impute_method", "ie_mae", "fe_mae"]].copy()

print(f"Original number of successful records: {len(df)}")
print(f"Types of interpolation methods: {sorted(df['impute_method'].unique())}")
print()

# ── Helper function ───────────────────────────────────────────

def compute_tau_stats(df_long, anchor_method, epsilon=EPSILON):
    """
    by anchor_method As the baseline, calculate τ The full sample statistic of。
    exclude anchor_method one's own actions φ sample (the denominator is 0）。
    """
    # each group Take out anchor of IE / FE
    anchor = (
        df_long[df_long["impute_method"] == anchor_method]
        .set_index(GROUP_KEYS)[["ie_mae", "fe_mae"]]
        .rename(columns={"ie_mae": "ie_anchor", "fe_mae": "fe_anchor"})
    )

    # Left join: subtract the same group from each row anchor value
    # impute_method Reserved as a normal column, not used as an index
    idx = df_long.set_index(GROUP_KEYS).index
    df_indexed = df_long.set_index(GROUP_KEYS)
    merged = df_indexed.join(anchor, on=GROUP_KEYS, how="inner").reset_index()

    # remove anchor itself（ΔIE = 0）
    merged = merged[merged["impute_method"] != anchor_method]

    delta_ie = merged["ie_mae"] - merged["ie_anchor"]
    delta_fe = merged["fe_mae"] - merged["fe_anchor"]

    # Denominator filtering (consistent with the main experiment of the paper）
    valid_mask = np.abs(delta_ie) > epsilon
    near_zero = (~valid_mask).sum()
    total = len(valid_mask)

    tau = (delta_fe[valid_mask] / delta_ie[valid_mask]).replace([np.inf, -np.inf], np.nan).dropna()

    n_valid = len(tau)
    tau_median = tau.median()
    tau_iqr = tau.quantile(0.75) - tau.quantile(0.25)

    # Pure upside down：ΔIE < 0（φ Compare anchor interpolation is better) and ΔFE > 0（The forecast is worse）
    # Right now"Imputation is better but prediction is worse"——The core definition of performance inversion
    valid_delta_ie = delta_ie[valid_mask]
    valid_delta_fe = delta_fe[valid_mask]
    pure_reversal_mask = (valid_delta_ie < 0) & (valid_delta_fe > 0)
    # The pure inversion rate is measured by the number of effective samples N_valid as the denominator (consistent with the main experimental caliber of the paper）
    pure_reversal_rate = pure_reversal_mask.sum() / n_valid * 100

    # τ < 0 full inversion rate (also expressed as N_valid as the denominator）
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


# ── calculate ───────────────────────────────────────────────

stats_mean = compute_tau_stats(df, "mean")
stats_spline = compute_tau_stats(df, "spline")

# ── Print report ───────────────────────────────────────────

SEP = "=" * 80
DASH = "-" * 80

lines = []
lines.append(SEP)
lines.append("          baseline anchor point (Baseline Anchor) Sensitivity analysis report")
lines.append(SEP)
lines.append("")
lines.append(f"Data source: {SRC.relative_to(REPO_ROOT).as_posix()}")
lines.append(f"numerical stability threshold ε = {EPSILON}")
lines.append("")
lines.append(DASH)
lines.append(f"{'index / Statistical items':<38} {'Mean anchor point (Original paper)':>22} {'Spline anchor point (New verification)':>22}")
lines.append(DASH)

rows = [
    ("Effectively assess sample size (N)", f"{stats_mean['N_valid']:,}", f"{stats_spline['N_valid']:,}"),
    ("Total evaluation sample size (Contains denominator tending to zero)", f"{stats_mean['N_total']:,}", f"{stats_spline['N_total']:,}"),
    ("Tau median", fmt(stats_mean["tau_median"]), fmt(stats_spline["tau_median"])),
    ("Tau interquartile range (IQR)", fmt(stats_mean["tau_iqr"]), fmt(stats_spline["tau_iqr"])),
    ("Tau mean (Reference, affected by heavy tails)", fmt(stats_mean["tau"].mean()), fmt(stats_spline["tau"].mean())),
    ("pure inversion rate (%) (ΔIE<0 and ΔFE>0)", f"{stats_mean['pure_reversal_rate']:.2f}%", f"{stats_spline['pure_reversal_rate']:.2f}%"),
    ("full inversion rate (%) (τ < 0)", f"{stats_mean['tau_neg_rate']:.2f}%", f"{stats_spline['tau_neg_rate']:.2f}%"),
    ("Sample proportion close to zero (|ΔIE|<0.01)", f"{stats_mean['near_zero_pct']:.2f}%", f"{stats_spline['near_zero_pct']:.2f}%"),
]

for label, mean_val, spline_val in rows:
    lines.append(f"{label:<38} {mean_val:>22} {spline_val:>22}")

lines.append(DASH)
lines.append("")

# ── in conclusion ───────────────────────────────────────────────

median_close_to_zero = abs(stats_spline["tau_median"]) < 0.1
reversal_significant = stats_spline["pure_reversal_rate"] > 10

lines.append("Conclusion verification：")
lines.append(
    f"1. switch to Spline after anchor point，Tau Is the median still close to zero?: "
    f"[{'yes' if median_close_to_zero else 'no'}]  "
    f"(Spline τ median = {fmt(stats_spline['tau_median'])})"
)
lines.append(
    f"2. Is the inversion phenomenon still evident?: "
    f"[{'yes' if reversal_significant else 'no'}]  "
    f"(Spline pure inversion rate = {stats_spline['pure_reversal_rate']:.2f}%)"
)
lines.append("")
lines.append("Robustness Judgment：")
delta_median = stats_spline["tau_median"] - stats_mean["tau_median"]
lines.append(
    f"  - Tau Median change: {fmt(delta_median)}  "
    f"({'increase' if delta_median > 0 else 'Decrease'}，relative change {abs(delta_median)/stats_mean['tau_median']*100:.1f}%)"
)
lines.append(
    f"  - Pure inversion rate change: {stats_spline['pure_reversal_rate'] - stats_mean['pure_reversal_rate']:+.2f} pp"
)
lines.append("")
if median_close_to_zero and reversal_significant:
    lines.append(
        "  => The core conclusion is robust: regardless of Mean still Spline as anchor point，"
    )
    lines.append(
        "    τ The medians are all close to zero, and the pure inversion rates are all above 10%，"
    )
    lines.append(
        "    'Weak conduction'and'Performance inversion is common'The conclusion does not depend on the baseline selection。"
    )
else:
    lines.append("  => Warning: The core conclusions are sensitive to baseline selection and require additional explanation in the paper.。")
lines.append("")
lines.append(SEP)

report = "\n".join(lines)
print(report)

# ── write file ───────────────────────────────────────────
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\nReport saved to: {OUT_FILE}")

# ── extra output：τ Distribution comparison CSV ──────────────────────────
csv_out = os.path.join(OUT_DIR, "anchor_sensitivity_tau_comparison.csv")
cmp_df = pd.DataFrame({
    "tau_mean_anchor": stats_mean["tau"],
    "tau_spline_anchor": stats_spline["tau"],
}).reset_index(drop=True)
cmp_df.to_csv(csv_out, index=False)
print(f"τ Sample-by-sample comparison has been saved to: {csv_out}")
