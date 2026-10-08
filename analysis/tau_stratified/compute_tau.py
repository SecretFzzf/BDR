"""
Calculated according to methodological formula tau:

    tau(g, phi, m, r, h) =
        (FE(g,phi,m,r,h) - FE(g,Mean,m,r,h)) / (IE(phi,m,r) - IE(Mean,m,r))
            if |IE(phi,m,r) - IE(Mean,m,r)| > epsilon
        undefined  otherwise

Agreement:
- IE only rely on (dataset, phi, m, r) —— across (model, pred_len) Take the mean to eliminate small residuals
- The baseline interpolation method is Mean
- epsilon = 0.01
- Error measurement is used uniformly MAE
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SRC = REPO_ROOT / "results" / "seed42" / "all_models_result.csv"
OUT = REPO_ROOT / "results" / "seed42" / "tau_results.csv"
EPSILON = 0.01
BASELINE = "mean"

df = pd.read_csv(SRC)
df = df[df["status"] == "success"].copy()

# IE only rely on (dataset, impute, mode, rate)
ie_table = (
    df.groupby(["dataset", "impute_method", "missing_mode", "missing_rate"])["ie_mae"]
    .mean()
    .reset_index()
    .rename(columns={"ie_mae": "ie_mae_phi"})
)

# Mean baseline IE
ie_mean = (
    ie_table[ie_table["impute_method"] == BASELINE]
    .drop(columns=["impute_method"])
    .rename(columns={"ie_mae_phi": "ie_mae_mean"})
)

# Mean baseline FE: rely (dataset, model, mode, rate, pred_len)
fe_mean = (
    df[df["impute_method"] == BASELINE]
    [["dataset", "model", "missing_mode", "missing_rate", "pred_len", "fe_mae"]]
    .rename(columns={"fe_mae": "fe_mae_mean"})
)

# only right phi != Mean row calculation tau
out = df[df["impute_method"] != BASELINE].copy()
out = out.merge(
    ie_table.rename(columns={"ie_mae_phi": "ie_mae_phi_avg"}),
    on=["dataset", "impute_method", "missing_mode", "missing_rate"],
    how="left",
)
out = out.merge(
    ie_mean,
    on=["dataset", "missing_mode", "missing_rate"],
    how="left",
)
out = out.merge(
    fe_mean,
    on=["dataset", "model", "missing_mode", "missing_rate", "pred_len"],
    how="left",
)

out["delta_ie"] = out["ie_mae_phi_avg"] - out["ie_mae_mean"]
out["delta_fe"] = out["fe_mae"] - out["fe_mae_mean"]

guard = out["delta_ie"].abs() > EPSILON
out["tau"] = np.where(guard, out["delta_fe"] / out["delta_ie"], np.nan)
out["tau_status"] = np.where(guard, "valid", "dropped_denom")

# Arrange column order
cols = [
    "dataset", "model", "missing_mode", "missing_rate", "pred_len",
    "impute_method",
    "ie_mae_phi_avg", "ie_mae_mean", "delta_ie",
    "fe_mae", "fe_mae_mean", "delta_fe",
    "tau", "tau_status",
]
out = out[cols].sort_values(
    ["dataset", "model", "missing_mode", "missing_rate", "pred_len", "impute_method"]
).reset_index(drop=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

# summary
n_total = len(out)
n_valid = (out["tau_status"] == "valid").sum()
n_dropped = (out["tau_status"] == "dropped_denom").sum()
print(f"Saved to: {OUT}")
print(f"Total rows (phi != Mean): {n_total}")
print(f"  valid tau:     {n_valid}  ({n_valid/n_total:.1%})")
print(f"  dropped_denom: {n_dropped}  ({n_dropped/n_total:.1%})")
print()
print("tau distribution (valid only):")
v = out.loc[out["tau_status"] == "valid", "tau"]
print(f"  count   = {len(v)}")
print(f"  median  = {v.median():.4f}")
print(f"  mean    = {v.mean():.4f}")
print(f"  >0 rate = {(v > 0).mean():.1%}")
print(f"  <0 rate = {(v < 0).mean():.1%}  (Inversion rate)")
print(f"  range   = [{v.min():.3f}, {v.max():.3f}]")
print()
print("By dataset:")
print(out[out["tau_status"]=="valid"].groupby("dataset")["tau"].agg(["count","median","mean"]))
print()
print("By model:")
print(out[out["tau_status"]=="valid"].groupby("model")["tau"].agg(["count","median","mean"]))
print()
print("By impute_method:")
print(out[out["tau_status"]=="valid"].groupby("impute_method")["tau"].agg(["count","median","mean"]))
