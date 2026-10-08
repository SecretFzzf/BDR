from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SRC = REPO_ROOT / "results" / "seed42" / "tau_results.csv"
OUT_DIR = REPO_ROOT / "results" / "配对显著性检验"
CSV_OUT = OUT_DIR / "table1_stat_tests.csv"
REPORT_OUT = OUT_DIR / "table1_stat_report.txt"

N_RESAMPLES = 2000
BOOTSTRAP_SEED = 42

MODEL_KEYS = ["dataset", "impute_method", "missing_mode", "missing_rate", "pred_len"]
IMPUTER_KEYS = ["dataset", "model", "missing_mode", "missing_rate", "pred_len"]

COMPARISONS = [
    ("LSTM vs DLinear", "model", "LSTM", "DLinear", MODEL_KEYS),
    ("LSTM vs PatchTST", "model", "LSTM", "PatchTST", MODEL_KEYS),
    ("PatchTST vs DLinear", "model", "PatchTST", "DLinear", MODEL_KEYS),
    ("spline vs saits", "impute_method", "spline", "saits", IMPUTER_KEYS),
    ("spline vs knn", "impute_method", "spline", "knn", IMPUTER_KEYS),
    ("brits vs saits", "impute_method", "brits", "saits", IMPUTER_KEYS),
]


def paired_sample(data, column, method_a, method_b, keys):
    sample_a = data[data[column] == method_a][keys + ["pure_inversion"]].rename(
        columns={"pure_inversion": "method_a"}
    )
    sample_b = data[data[column] == method_b][keys + ["pure_inversion"]].rename(
        columns={"pure_inversion": "method_b"}
    )
    paired = sample_a.merge(sample_b, on=keys, how="inner")
    if paired.empty:
        raise ValueError(f"No paired observations for {method_a} vs {method_b}")
    return paired


data = pd.read_csv(SRC)
data = data[data["tau_status"] == "valid"].copy()
data["pure_inversion"] = ((data["delta_ie"] < 0) & (data["delta_fe"] > 0)).astype(int)

rows = []
for label, column, method_a, method_b, keys in COMPARISONS:
    paired = paired_sample(data, column, method_a, method_b, keys)
    values_a = paired["method_a"].to_numpy()
    values_b = paired["method_b"].to_numpy()
    differences = values_a.astype(float) - values_b.astype(float)

    rate_a = float(values_a.mean() * 100)
    rate_b = float(values_b.mean() * 100)
    point_difference = float(differences.mean() * 100)

    random_state = np.random.RandomState(BOOTSTRAP_SEED)
    indices = random_state.randint(
        0, len(differences), size=(N_RESAMPLES, len(differences))
    )
    bootstrap_differences = differences[indices].mean(axis=1) * 100
    bootstrap_mean = float(bootstrap_differences.mean())
    ci_lower, ci_upper = np.quantile(bootstrap_differences, [0.025, 0.975])

    method_a_only = int(((values_a == 1) & (values_b == 0)).sum())
    method_b_only = int(((values_a == 0) & (values_b == 1)).sum())
    discordant = method_a_only + method_b_only
    mcnemar_p = float(
        binomtest(method_a_only, discordant, 0.5, alternative="two-sided").pvalue
    )
    wilcoxon_p = float(
        wilcoxon(
            values_a,
            values_b,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        ).pvalue
    )

    rows.append(
        {
            "对比组合": label,
            "方法A纯倒挂率(%)": rate_a,
            "方法B纯倒挂率(%)": rate_b,
            "差值(%)": point_difference,
            "Bootstrap均值差(%)": bootstrap_mean,
            "95%CI下限(%)": float(ci_lower),
            "95%CI上限(%)": float(ci_upper),
            "McNemar_p值": mcnemar_p,
            "Wilcoxon_p值": wilcoxon_p,
            "配对样本数": len(paired),
        }
    )

summary = pd.DataFrame(rows)
for column in [
    "方法A纯倒挂率(%)",
    "方法B纯倒挂率(%)",
    "差值(%)",
    "Bootstrap均值差(%)",
    "95%CI下限(%)",
    "95%CI上限(%)",
]:
    summary[column] = summary[column].round(2)
OUT_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(CSV_OUT, index=False)

lines = [
    "配对显著性检验复现",
    "=" * 78,
    "统计方法：McNemar 配对二值检验 + Wilcoxon 符号秩检验 + 配对 Bootstrap(2000次)",
    f"Bootstrap 随机种子：{BOOTSTRAP_SEED}",
    "",
    summary.to_string(index=False),
    "",
]
REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
print(summary.to_string(index=False))
print(f"Saved: {CSV_OUT}")
print(f"Saved: {REPORT_OUT}")
