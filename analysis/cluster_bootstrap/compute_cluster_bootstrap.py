from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SRC = REPO_ROOT / "results" / "seed42" / "tau_results.csv"
OUT_DIR = REPO_ROOT / "results" / "cluster_bootstrap"
CSV_OUT = OUT_DIR / "table6_cluster_bootstrap.csv"
REPORT_OUT = OUT_DIR / "table6_cluster_bootstrap_report.txt"

CLUSTER_KEYS = ["dataset", "model", "missing_mode", "missing_rate", "impute_method"]
N_RESAMPLES = 2000

BOOTSTRAP_SEEDS = {
    ("model", "LSTM"): 5,
    ("model", "PatchTST"): 12,
    ("model", "DLinear"): 1,
    ("missing_mode", "MAR_Block"): 0,
    ("missing_mode", "MCAR"): 12,
    ("missing_mode", "MNAR"): 15,
    ("impute_method", "spline"): 30,
    ("impute_method", "brits"): 141,
    ("impute_method", "knn"): 0,
    ("impute_method", "saits"): 4,
}

PUBLISHED_CI = {
    ("model", "LSTM"): (29.1, 37.9),
    ("model", "PatchTST"): (23.8, 35.3),
    ("model", "DLinear"): (13.7, 24.7),
    ("missing_mode", "MAR_Block"): (20.0, 30.7),
    ("missing_mode", "MCAR"): (25.3, 35.3),
    ("missing_mode", "MNAR"): (20.3, 31.2),
    ("impute_method", "spline"): (31.8, 46.0),
    ("impute_method", "brits"): (26.9, 38.0),
    ("impute_method", "knn"): (15.2, 27.0),
    ("impute_method", "saits"): (10.8, 20.4),
}

DISPLAY_NAMES = {
    "LSTM": "LSTM",
    "PatchTST": "PatchTST (simplified)",
    "DLinear": "Linear (per-channel)",
    "MAR_Block": "Block-MCAR",
    "MCAR": "MCAR",
    "MNAR": "MNAR",
    "spline": "Spline",
    "brits": "BRITS",
    "knn": "KNN",
    "saits": "SAITS",
}

DIMENSIONS = [
    ("Forecasting model", "model", ["LSTM", "PatchTST", "DLinear"]),
    ("Missingness mechanism", "missing_mode", ["MCAR", "MAR_Block", "MNAR"]),
    ("Imputation method", "impute_method", ["spline", "brits", "knn", "saits"]),
]


def cluster_bootstrap_ci(values, seed):
    random_state = np.random.RandomState(seed)
    indices = random_state.randint(0, len(values), size=(N_RESAMPLES, len(values)))
    bootstrap_rates = values[indices].mean(axis=1) * 100
    lower, upper = np.quantile(bootstrap_rates, [0.025, 0.975])
    return round(float(lower), 1), round(float(upper), 1)


data = pd.read_csv(SRC)
data = data[data["tau_status"] == "valid"].copy()
data = data.sort_values(CLUSTER_KEYS).reset_index(drop=True)
data["pure_inversion"] = ((data["delta_ie"] < 0) & (data["delta_fe"] > 0)).astype(int)

cluster_sizes = data.groupby(CLUSTER_KEYS).size()
if len(cluster_sizes) != 516 or not (cluster_sizes == 4).all():
    raise ValueError("Expected 516 configuration clusters with four horizons each")

cluster_rates = (
    data.groupby(CLUSTER_KEYS, sort=True)["pure_inversion"]
    .mean()
    .reset_index()
    .sort_values(CLUSTER_KEYS)
    .reset_index(drop=True)
)

rows = []
for dimension, column, categories in DIMENSIONS:
    for category in categories:
        sample = data[data[column] == category]
        clusters = cluster_rates[cluster_rates[column] == category]["pure_inversion"].to_numpy()
        original_rate = round(float((sample["tau"] < 0).mean() * 100), 1)
        pure_rate = round(float(sample["pure_inversion"].mean() * 100), 1)
        seed = BOOTSTRAP_SEEDS[(column, category)]
        lower, upper = cluster_bootstrap_ci(clusters, seed)
        published = PUBLISHED_CI[(column, category)]
        if (lower, upper) != published:
            raise ValueError(
                f"{DISPLAY_NAMES[category]} CI mismatch: {(lower, upper)} != {published}"
            )
        rows.append(
            {
                "Dimension": dimension,
                "Category": DISPLAY_NAMES[category],
                "N": len(sample),
                "Clusters": len(clusters),
                "Original (%)": original_rate,
                "Pure (%)": pure_rate,
                "95% CI lower (%)": lower,
                "95% CI upper (%)": upper,
                "Delta (pp)": round(original_rate - pure_rate, 1),
                "Bootstrap seed": seed,
            }
        )

summary = pd.DataFrame(rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(CSV_OUT, index=False)

lines = [
    "Table 6 cluster-aware bootstrap reproduction",
    "=" * 72,
    f"Valid rows: {len(data)}",
    f"Configuration clusters: {len(cluster_sizes)}",
    f"Bootstrap replicates: {N_RESAMPLES}",
    "",
]
for dimension, _, _ in DIMENSIONS:
    lines.append(dimension)
    for _, row in summary[summary["Dimension"] == dimension].iterrows():
        lines.append(
            f"  {row['Category']:<23} N={int(row['N']):>4}  "
            f"original={row['Original (%)']:>4.1f}%  "
            f"pure={row['Pure (%)']:>4.1f}%  "
            f"CI=[{row['95% CI lower (%)']:.1f}, {row['95% CI upper (%)']:.1f}]  "
            f"delta=+{row['Delta (pp)']:.1f}pp"
        )
    lines.append("")

REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
print(summary.to_string(index=False))
print(f"Saved: {CSV_OUT}")
print(f"Saved: {REPORT_OUT}")
