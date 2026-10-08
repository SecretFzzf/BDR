"""
ETTh1 5Seed Merging and Symbol Flip Rate（Sign Flip Rate）analyze
--------------------------------------------------
enter：results/multi_seed_sign_flip/ETT_seed{42,123,2026,456,789}/
      results_refactor_{lstm,dlinear,patchtst}.csv
output：
  - Console: Three-dimensional statistical summary table
  - disk：merged_5seeds_ett_results.csv
"""

import os
import glob
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
BASE = str(REPO_ROOT / "results" / "multi_seed_sign_flip")
OUT_CSV = os.path.join(BASE, "merged_5seeds_ett_results.csv")

# ── 1. Read all recursively result_refactor_*.csv ──────────────
rows = []
for folder in sorted(glob.glob(os.path.join(BASE, "ETT_seed*"))):
    seed_str = folder.split("_")[-1]
    seed = int(seed_str.replace("seed", ""))
    csv_files = glob.glob(os.path.join(folder, "results_refactor_*.csv"))
    for f in csv_files:
        df = pd.read_csv(f)
        df["seed"] = seed
        rows.append(df)
        print(f"[read] seed={seed:4d}  {os.path.basename(f):35s}  {len(df)} OK")

raw = pd.concat(rows, ignore_index=True)
print(f"\nOriginal number of records after merging: {len(raw)}")

# ── 2. Clean ─────────────────────────────────────────────
# Keep only successful rows
raw = raw[raw["status"] == "success"].copy()

# Unified listing：dataset fixed to ETTh1（Folder is hidden）
raw["dataset"] = "ETTh1"

# standardization impute_method Case
raw["impute_method"] = raw["impute_method"].str.lower()

# standardization missing_mode
raw["missing_mode"] = raw["missing_mode"].str.replace("MAR_Block", "MAR_Block", regex=False)

# Keep only the columns you need
KEEP_COLS = ["dataset", "model", "missing_mode", "impute_method",
             "missing_rate", "pred_len", "ie_mae", "fe_mae", "seed"]
df = raw[KEEP_COLS].copy()

# ── 3. calculate ΔFE = fe_phi - fe_mean ─────────────────────
# fe_mean：each (model, seed, missing_mode, missing_rate, pred_len) Down Mean of fe_mae
mean_ref = (
    df[df["impute_method"] == "mean"]
    .groupby(["model", "seed", "missing_mode", "missing_rate", "pred_len"])["fe_mae"]
    .mean()
    .rename("fe_mean")
    .reset_index()
)

df = df.merge(mean_ref, on=["model", "seed", "missing_mode", "missing_rate", "pred_len"], how="left")
df["delta_fe"] = df["fe_mae"] - df["fe_mean"]
df["delta_fe"] = df["delta_fe"].fillna(0.0)

# sign / inversion
df["sign"] = df["delta_fe"].apply(
    lambda x: "positive" if x > 0 else ("negative" if x < 0 else "zero")
)
df["inversion"] = df["delta_fe"] > 0.0

# remove mean itself (does not participate in flip rate calculation）
df_phi = df[df["impute_method"] != "mean"].copy()

print(f"After cleaning (excluding Mean）: {len(df_phi)} OK")
print(f"seed covering: {sorted(df_phi['seed'].unique().tolist())}")
print(f"Model coverage: {sorted(df_phi['model'].unique().tolist())}")
print()

# ── 4. Dimension I: Sample-level sign flip rate ────────────────────────
# Experimental unit primary key
UNIT_KEYS = ["model", "impute_method", "missing_mode", "missing_rate", "pred_len"]

def sign_flip_rate(sign_series):
    """within group 5 Proportion of sign-discordant pairs among seeds. What is passed in is sign column Series。"""
    signs = sign_series.tolist()
    n = len(signs)
    if n < 2:
        return np.nan
    discordant = sum(1 for a, b in combinations(signs, 2) if a != b)
    total_pairs = n * (n - 1) // 2
    return discordant / total_pairs

unit_stats = df_phi.groupby(UNIT_KEYS).agg(
    n_seeds=("seed", "nunique"),
    sign_flip_rate=("sign", sign_flip_rate),
    delta_fe_mean=("delta_fe", "mean"),
    delta_fe_std=("delta_fe", "std"),
).reset_index()

# Only keep ≥2 Seed unit (otherwise the flip rate is undefined）
unit_valid = unit_stats[unit_stats["n_seeds"] >= 2].copy()

mean_sfr = unit_valid["sign_flip_rate"].mean()
median_sfr = unit_valid["sign_flip_rate"].median()
deterministic_pct = (unit_valid["sign_flip_rate"] == 0).sum() / len(unit_valid) * 100

print("=" * 65)
print("  Dimension I: Sample-level sign flip rate (Sign Flip Rate)")
print("=" * 65)
print(f"  Effective number of experimental units (≥2 seed):  {len(unit_valid)}")
print(f"  average symbol flip rate:            {mean_sfr:.4f}  ({mean_sfr*100:.2f}%)")
print(f"  Median sign flip rate:          {median_sfr:.4f}  ({median_sfr*100:.2f}%)")
print(f"  Certainty remains unchanged (flip rate=0%) Proportion: {deterministic_pct:.2f}%")
print()

# ── 5. Dimension II: Algorithmic combination (Model × Imputer) Summary ──────
algo_stats = (
    df_phi.groupby(["model", "impute_method"])
    .agg(
        evaluated_samples=("delta_fe", "count"),
        inversion_samples=("inversion", "sum"),
        pure_inversion_rate_pct=("inversion", "mean"),
        delta_fe_mean=("delta_fe", "mean"),
        delta_fe_median=("delta_fe", "median"),
    )
    .reset_index()
)

# Combined average symbol flip rate for this combination
algo_sfr = (
    unit_valid.groupby(["model", "impute_method"])["sign_flip_rate"]
    .mean()
    .rename("average symbol flip rate")
    .reset_index()
)
algo_stats = algo_stats.merge(algo_sfr, on=["model", "impute_method"], how="left")

algo_stats["pure_inversion_rate_pct"] = algo_stats["pure_inversion_rate_pct"] * 100

print("=" * 65)
print("  Dimension II: Algorithmic combination (Model × Imputer) Summary")
print("=" * 65)
print(algo_stats.to_string(index=False))
print()

# ── 6. Dimension three: press random seed (Seed) Summary ─────────────────
seed_stats = (
    df_phi.groupby("seed")
    .agg(
        evaluated_samples=("delta_fe", "count"),
        inversion_samples=("inversion", "sum"),
        pure_inversion_rate_pct=("inversion", "mean"),
        delta_fe_median=("delta_fe", "median"),
        delta_fe_mean=("delta_fe", "mean"),
    )
    .reset_index()
)
seed_stats["pure_inversion_rate_pct"] = seed_stats["pure_inversion_rate_pct"] * 100
seed_stats["delta_fe_median"] = seed_stats["delta_fe_median"].round(4)
seed_stats["delta_fe_mean"] = seed_stats["delta_fe_mean"].round(4)

print("=" * 65)
print("  Dimension three: press random seed (Seed) Summary")
print("=" * 65)
print(seed_stats.to_string(index=False))
print()

# ── 7. Export full merge CSV ────────────────────────────────
# seq_len in all files 96，directly from raw Fill in a column
seq_len_map = raw[["model", "seed", "missing_mode", "impute_method",
                   "missing_rate", "pred_len", "seq_len"]].drop_duplicates()
df = df.merge(seq_len_map,
              on=["model", "seed", "missing_mode", "impute_method",
                  "missing_rate", "pred_len"],
              how="left", suffixes=("", "_dup"))
# Remove possible duplicate columns
dup_cols = [c for c in df.columns if c.endswith("_dup")]
df = df.drop(columns=dup_cols)

export_cols = ["dataset", "model", "missing_mode", "impute_method",
               "missing_rate", "pred_len", "seq_len",
               "ie_mae", "fe_mae", "fe_mean", "delta_fe", "sign", "inversion", "seed"]
df[export_cols].sort_values(["seed", "model", "impute_method",
                              "missing_mode", "missing_rate", "pred_len"]
                            ).to_csv(OUT_CSV, index=False)
print(f"[write] {OUT_CSV}  ({len(df)} OK)")
