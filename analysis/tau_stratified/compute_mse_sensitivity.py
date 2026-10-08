from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SRC = REPO_ROOT / "results" / "seed42" / "all_models_result.csv"
OUT_DIR = REPO_ROOT / "results" / "tau_stratified"
CSV_OUT = OUT_DIR / "mse_sensitivity_summary.csv"
REPORT_OUT = OUT_DIR / "mse_sensitivity_report.txt"

EPSILON = 0.01


def build_tau(data, ie_column, fe_column):
    ie_table = (
        data.groupby(["dataset", "impute_method", "missing_mode", "missing_rate"])[
            ie_column
        ]
        .mean()
        .reset_index()
        .rename(columns={ie_column: "ie_phi"})
    )
    ie_mean = (
        ie_table[ie_table["impute_method"] == "mean"]
        .drop(columns="impute_method")
        .rename(columns={"ie_phi": "ie_mean"})
    )
    fe_mean = (
        data[data["impute_method"] == "mean"][
            ["dataset", "model", "missing_mode", "missing_rate", "pred_len", fe_column]
        ]
        .rename(columns={fe_column: "fe_mean"})
    )
    result = data[data["impute_method"] != "mean"].copy()
    result = result.merge(
        ie_table, on=["dataset", "impute_method", "missing_mode", "missing_rate"], how="left"
    )
    result = result.merge(
        ie_mean, on=["dataset", "missing_mode", "missing_rate"], how="left"
    )
    result = result.merge(
        fe_mean,
        on=["dataset", "model", "missing_mode", "missing_rate", "pred_len"],
        how="left",
    )
    result["delta_ie"] = result["ie_phi"] - result["ie_mean"]
    result["delta_fe"] = result[fe_column] - result["fe_mean"]
    result = result[result["delta_ie"].abs() > EPSILON].copy()
    result["tau"] = result["delta_fe"] / result["delta_ie"]
    return result


data = pd.read_csv(SRC)
data = data[data["status"] == "success"].copy()

rows = []
for metric, ie_column, fe_column in [
    ("MAE", "ie_mae", "fe_mae"),
    ("MSE", "ie_mse", "fe_mse"),
]:
    result = build_tau(data, ie_column, fe_column)
    rows.append(
        {
            "Metric": metric,
            "N valid": len(result),
            "Median tau": float(result["tau"].median()),
            "Pure inversion (%)": float(
                ((result["delta_ie"] < 0) & (result["delta_fe"] > 0)).mean() * 100
            ),
            "Global tau < 0 (%)": float((result["tau"] < 0).mean() * 100),
        }
    )

summary = pd.DataFrame(rows).set_index("Metric")
mae = summary.loc["MAE"]
mse = summary.loc["MSE"]
if round(float(mse["Median tau"]), 4) != 0.0328:
    raise ValueError("MSE median tau does not match the paper")
if round(float(mse["Pure inversion (%)"]), 2) != 24.35:
    raise ValueError("MSE pure inversion rate does not match the paper")
if round(float(mse["Global tau < 0 (%)"]), 2) != 29.58:
    raise ValueError("MSE global inversion rate does not match the paper")

OUT_DIR.mkdir(parents=True, exist_ok=True)
summary.reset_index().to_csv(CSV_OUT, index=False)
report = f"""MSE sensitivity reproduction
{"=" * 72}

MAE: N={int(mae['N valid'])}, median tau={mae['Median tau']:.4f},
     pure inversion={mae['Pure inversion (%)']:.2f}%,
     global tau<0={mae['Global tau < 0 (%)']:.2f}%.

MSE: N={int(mse['N valid'])}, median tau={mse['Median tau']:.4f},
     pure inversion={mse['Pure inversion (%)']:.2f}%,
     global tau<0={mse['Global tau < 0 (%)']:.2f}%.
"""
REPORT_OUT.write_text(report, encoding="utf-8")
print(summary.to_string())
print(f"Saved: {CSV_OUT}")
print(f"Saved: {REPORT_OUT}")
