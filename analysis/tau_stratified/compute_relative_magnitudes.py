from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SRC = REPO_ROOT / "results" / "seed42" / "tau_results.csv"
OUT_DIR = REPO_ROOT / "results" / "tau_stratified"
CSV_OUT = OUT_DIR / "relative_magnitude_summary.csv"
REPORT_OUT = OUT_DIR / "relative_magnitude_report.txt"


data = pd.read_csv(SRC)
data = data[data["tau_status"] == "valid"].copy()
data["rel_dFE"] = data["delta_fe"] / data["fe_mae_mean"]
data["rel_dIE"] = data["delta_ie"] / data["ie_mae_mean"]
data["eta"] = data["rel_dFE"] / data["rel_dIE"]

rows = []
for label, column in [
    ("FE relative change (%)", "rel_dFE"),
    ("IE relative change (%)", "rel_dIE"),
    ("eta (%)", "eta"),
]:
    quantiles = data[column].quantile([0.25, 0.50, 0.75]) * 100
    rows.append(
        {
            "Metric": label,
            "Q25 (%)": float(quantiles.loc[0.25]),
            "Median (%)": float(quantiles.loc[0.50]),
            "Q75 (%)": float(quantiles.loc[0.75]),
        }
    )

summary = pd.DataFrame(rows)
fe = summary.iloc[0]
ie = summary.iloc[1]
eta = summary.iloc[2]
expected = {
    "FE": (-6.81, -1.72, 1.44),
    "IE": (-51.29, -28.17, -12.69),
    "eta": (-4.34, 7.59, 24.57),
}
for name, row in [("FE", fe), ("IE", ie), ("eta", eta)]:
    got = tuple(round(float(row[key]), 2) for key in ["Q25 (%)", "Median (%)", "Q75 (%)"])
    if got != expected[name]:
        raise ValueError(f"{name} mismatch: {got} != {expected[name]}")

delta_fe_median = float(data["delta_fe"].median())
delta_ie_median = float(data["delta_ie"].median())
ratio_relative_medians = float(fe["Median (%)"] / ie["Median (%)"] * 100)
ratio_absolute_medians = float(delta_fe_median / delta_ie_median * 100)

OUT_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(CSV_OUT, index=False)
report = f"""Relative magnitude reproduction
{"=" * 72}
N = {len(data)}

median(rel_dFE) = {fe['Median (%)']:.2f}%
IQR(rel_dFE) = [{fe['Q25 (%)']:.2f}%, {fe['Q75 (%)']:+.2f}%]

median(rel_dIE) = {ie['Median (%)']:.2f}%
IQR(rel_dIE) = [{ie['Q25 (%)']:.2f}%, {ie['Q75 (%)']:.2f}%]

median(eta) = {eta['Median (%)']:.2f}%
IQR(eta) = [{eta['Q25 (%)']:.2f}%, {eta['Q75 (%)']:+.2f}%]

median(delta_FE) = {delta_fe_median:.4f}
median(delta_IE) = {delta_ie_median:.4f}
ratio of relative medians = {ratio_relative_medians:.2f}%
ratio of absolute medians = {ratio_absolute_medians:.2f}%
"""
REPORT_OUT.write_text(report, encoding="utf-8")
print(summary.to_string(index=False))
print(f"Saved: {CSV_OUT}")
print(f"Saved: {REPORT_OUT}")
