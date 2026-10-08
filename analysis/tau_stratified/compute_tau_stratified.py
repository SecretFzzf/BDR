# -*- coding: utf-8 -*-
"""
Data set hierarchical conduction efficiency (Tau) Standalone statistics script
Read warehouse results/seed42/tau_results.csv，according to 4 hierarchical statistics and output of data sets txt Report。
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DATA_FILE = REPO_ROOT / "results" / "seed42" / "tau_results.csv"
OUTPUT_FILE = REPO_ROOT / "results" / "tau_stratified" / "tau_stratified_report.txt"

# Required columns
REQUIRED_COLS = ["dataset", "delta_fe", "delta_ie", "tau"]

# ── 1. Read data ──────────────────────────────────────────────────────────────
print(f"Read data: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print(f"  Original number of rows: {len(df)}")
print(f"  List: {list(df.columns)}")

# Verify necessary columns
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise KeyError(f"Missing required column: {missing}")

# ── 2. Data cleaning ──────────────────────────────────────────────────────────────
# Only keep tau valid (not NaN and limited) rows
df_clean = df[df["tau"].notna() & np.isfinite(df["tau"])].copy()
# Also required delta_fe and delta_ie efficient
df_clean = df_clean[
    df_clean["delta_fe"].notna()
    & df_clean["delta_ie"].notna()
    & np.isfinite(df_clean["delta_fe"])
    & np.isfinite(df_clean["delta_ie"])
].copy()
print(f"  Number of valid rows: {len(df_clean)} (Cull {len(df) - len(df_clean)} invalid records)")

# ── 3. global summary ──────────────────────────────────────────────────────────────
N_total = len(df_clean)
global_median_tau = float(df_clean["tau"].median())

# Pure inversion rate: better interpolation and worse prediction（ΔIE < 0 and ΔFE > 0）sample proportion
pool_improve = df_clean[(df_clean["delta_ie"] < 0) & (df_clean["delta_fe"] > 0)]
global_inversion_rate = float(len(pool_improve) / len(df_clean) * 100)

# ── 4. Stratified statistics by data set ──────────────────────────────────────────────────────
datasets = sorted(df_clean["dataset"].unique())
stratified = []

for ds in datasets:
    sub = df_clean[df_clean["dataset"] == ds].copy()
    N = len(sub)

    # Tau statistics
    tau_median = float(sub["tau"].median())
    tau_q25 = float(sub["tau"].quantile(0.25))
    tau_q75 = float(sub["tau"].quantile(0.75))
    tau_iqr = tau_q75 - tau_q25

    # ΔFE and ΔIE median
    delta_fe_median = float(sub["delta_fe"].median())
    delta_ie_median = float(sub["delta_ie"].median())

    # pure inversion rate：ΔIE < 0 and ΔFE > 0 sample proportion
    improve = sub[(sub["delta_ie"] < 0) & (sub["delta_fe"] > 0)]
    inversion_rate = float(len(improve) / len(sub) * 100)

    # Extreme outliers（3×IQR in principle）
    lower_fence = tau_q25 - 3 * tau_iqr
    upper_fence = tau_q75 + 3 * tau_iqr
    outlier_count = int(
        ((sub["tau"] < lower_fence) | (sub["tau"] > upper_fence)).sum()
    )

    stratified.append(
        {
            "dataset": ds,
            "N": N,
            "tau_median": tau_median,
            "tau_q25": tau_q25,
            "tau_q75": tau_q75,
            "tau_iqr": tau_iqr,
            "delta_fe_median": delta_fe_median,
            "delta_ie_median": delta_ie_median,
            "inversion_rate": inversion_rate,
            "outlier_count": outlier_count,
        }
    )

# ── 5. Summary of statistical conclusions ──────────────────────────────────────────────────────────
# in conclusion1：Each data set Tau Is the median close to zero?（|median| < 0.1 regarded as close）
all_near_zero = all(abs(s["tau_median"]) < 0.1 for s in stratified)
conclusion_1 = "yes" if all_near_zero else "no"

# in conclusion2：Is inversion prevalent across all datasets (pure inversion rate > 10% regarded as universal）
all_have_inversion = all(s["inversion_rate"] > 10.0 for s in stratified)
conclusion_2 = "yes" if all_have_inversion else "no"

# in conclusion3：The main distribution data set of extreme outlier points (the data set with the largest number of outlier points）
max_outlier_ds = max(stratified, key=lambda s: s["outlier_count"])
conclusion_3 = (
    f"{max_outlier_ds['dataset']}（{max_outlier_ds['outlier_count']} outliers）"
    if max_outlier_ds["outlier_count"] > 0
    else "There are no significant extreme outliers in each data set."
)

# ── 6. Generate report ──────────────────────────────────────────────────────────────
lines = []
sep = "=" * 80
dash = "-" * 80

lines.append(sep)
lines.append("                    Data set hierarchical conduction efficiency (Tau) independent statistical report")
lines.append(sep)
lines.append("")
lines.append("[global summary (Pooled Summary)]")
lines.append(f"Total sample size: N = {N_total}")
lines.append(f"overall situation Tau median: {global_median_tau:.4f}")
lines.append(f"overall situation pure inversion rate: {global_inversion_rate:.2f}%")
lines.append("")
lines.append(dash)
lines.append("[Detailed statistics of data set stratification]")
lines.append("")

for s in stratified:
    lines.append(f"Dataset: {s['dataset']}")
    lines.append(f"  - sample size (N): {s['N']}")
    lines.append(f"  - Tau median: {s['tau_median']:.4f}")
    lines.append(f"  - Tau interquartile range (IQR): {s['tau_iqr']:.4f}")
    lines.append(f"  - ΔFE median: {s['delta_fe_median']:.4f}")
    lines.append(f"  - ΔIE median: {s['delta_ie_median']:.4f}")
    lines.append(f"  - pure inversion rate (Pure Reversal Rate): {s['inversion_rate']:.2f}%")
    lines.append(f"  - extreme outlier points (Outliers Count): {s['outlier_count']}")
    lines.append("")

lines.append(dash)
lines.append("[Summary of statistical conclusions]")
lines.append(f"1. Each data set Tau Is the median close to zero?: {conclusion_1}")
lines.append(f"2. Is the inversion phenomenon prevalent across all datasets?: {conclusion_2}")
lines.append(f"3. Main distribution data sets of extreme outliers: {conclusion_3}")
lines.append(sep)

report = "\n".join(lines)

# ── 7. write file ──────────────────────────────────────────────────────────────
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\nReport has been written: {OUTPUT_FILE}")
print()
print(report)
