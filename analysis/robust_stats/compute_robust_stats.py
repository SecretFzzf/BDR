from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SRC = REPO_ROOT / "results" / "seed42" / "tau_results.csv"
OUT_DIR = REPO_ROOT / "results" / "robust_stats"
CSV_OUT = OUT_DIR / "robust_stats_summary.csv"
REPORT_OUT = OUT_DIR / "robust_stats_report.txt"
OUTLIER_DETAIL_OUT = OUT_DIR / "outliers_breakdown.csv"
OUTLIER_BY_DATASET_OUT = OUT_DIR / "outliers_by_dataset.csv"


def trimmed_mean(values, proportion):
    ordered = np.sort(values)
    cut = int(np.floor(len(ordered) * proportion))
    return float(ordered[cut : len(ordered) - cut].mean())


data = pd.read_csv(SRC)
data = data[data["tau_status"] == "valid"].copy()
tau = data["tau"].to_numpy()
median = float(np.median(tau))

statistics = {
    "Sample size": float(len(tau)),
    "Arithmetic mean (raw)": float(np.mean(tau)),
    "Sample standard deviation": float(np.std(tau, ddof=1)),
    "Median": median,
    "Median absolute deviation (MAD)": float(np.median(np.abs(tau - median))),
    "5% trimmed mean": trimmed_mean(tau, 0.05),
    "10% trimmed mean": trimmed_mean(tau, 0.10),
}
summary = pd.DataFrame(
    {"Statistic": list(statistics), "Value": list(statistics.values())}
)

extreme = data[np.abs(data["tau"]) > 2.0].copy()
extreme_negative = int((extreme["tau"] < -2.0).sum())
extreme_positive = int((extreme["tau"] > 2.0).sum())
dataset_counts = extreme.groupby("dataset").size().sort_values(ascending=False)
mean_abs_delta_ie = float(extreme["delta_ie"].abs().mean())
median_abs_delta_ie = float(extreme["delta_ie"].abs().median())
boundary_share = float(
    extreme["delta_ie"].abs().between(0.010, 0.025, inclusive="both").mean()
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(CSV_OUT, index=False)
extreme_output = extreme.copy()
extreme_output["abs_delta_ie"] = extreme_output["delta_ie"].abs()
extreme_output["abs_delta_fe"] = extreme_output["delta_fe"].abs()
extreme_output.to_csv(OUTLIER_DETAIL_OUT, index=False)
dataset_counts.rename("extreme_sample_count").rename_axis("dataset").reset_index().to_csv(
    OUTLIER_BY_DATASET_OUT, index=False
)

dataset_lines = "\n".join(
    f"                 {dataset:<15} {count:>3}" for dataset, count in dataset_counts.items()
)
report = f"""# Conducting efficiency indexτofrobust_statsanalysis report
================================================================================

## I. Summary of basic and robust statistics
Total number of valid samples N = {len(tau)}

| Statistics name               | numerical value          |
|--------------------------|---------------|
| Arithmetic mean (raw) | {statistics['Arithmetic mean (raw)']:.4f} |
| Sample standard deviation | {statistics['Sample standard deviation']:.4f} |
| Median | {statistics['Median']:.4f} |
| Median absolute deviation (MAD) | {statistics['Median absolute deviation (MAD)']:.4f} |
| 5% trimmed mean | {statistics['5% trimmed mean']:.4f} |
| 10% trimmed mean | {statistics['10% trimmed mean']:.4f} |

## II. Extreme values（|τ| > 2.0）attribution analysis
1.  Total number of extreme value samples：{len(extreme)}（of the total sample {len(extreme) / len(tau):.2%}）
2.  extreme negative values（τ < -2.0）Number of samples：{extreme_negative}
3.  extreme positive value（τ > 2.0）Number of samples：{extreme_positive}
4.  Extreme values ​​distributed by data set：
                 Dataset          Number of extreme value samples
{dataset_lines}

5.  extreme value sample|ΔIE|Distribution characteristics：
    - average|ΔIE|: {mean_abs_delta_ie:.4f}
    - median|ΔIE|: {median_abs_delta_ie:.4f}
    - at the critical boundary[0.010, 0.025]proportion of：{boundary_share:.2%}

## III. Summary of attributions for serious deviations between the arithmetic mean and the median
The arithmetic mean of tau is {statistics['Arithmetic mean (raw)']:.4f}, which departs sharply from the median of {statistics['Median']:.4f}. The main reasons are:
1.  **numerical amplification effect**：Denominator for extreme negative samplesΔIEclose to threshold0.01，lead toτbe greatly amplified。
2.  **Boundary sample verification**：{boundary_share:.2%}The extreme value sample is located at[0.010, 0.025]near the critical boundary of。
3.  **Robust statistical evidence**: the 5% trimmed mean ({statistics['5% trimmed mean']:.4f}) and 10% trimmed mean ({statistics['10% trimmed mean']:.4f}) support using the median and trimmed means to describe typical transmission efficiency.
"""
REPORT_OUT.write_text(report, encoding="utf-8")
print(summary.to_string(index=False))
print(f"Saved: {CSV_OUT}")
print(f"Saved: {REPORT_OUT}")
print(f"Saved: {OUTLIER_DETAIL_OUT}")
print(f"Saved: {OUTLIER_BY_DATASET_OUT}")
