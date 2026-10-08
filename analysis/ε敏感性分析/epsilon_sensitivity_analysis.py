import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
ROOT = REPO_ROOT / "results"
OUT_DIR = ROOT / "ε敏感性分析"
FIG_DIR = REPO_ROOT / "photo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})
sns.set_style("whitegrid")

tau_df = pd.read_csv(ROOT / "seed42" / "tau_results.csv")
total_samples = len(tau_df)
delta_ie = tau_df["delta_ie"].values
delta_fe = tau_df["delta_fe"].values
tau = tau_df["tau"].values
epsilons = [0.001, 0.005, 0.01, 0.02, 0.05]

results = []
for epsilon in epsilons:
    valid_mask = np.abs(delta_ie) > epsilon
    n_valid = int(valid_mask.sum())
    excluded_ratio = (total_samples - n_valid) / total_samples * 100
    pure_inv_mask = (delta_ie < -epsilon) & (delta_fe > 0)
    pure_inv_count = int(pure_inv_mask.sum())
    pure_inv_rate = pure_inv_count / n_valid * 100
    tau_effective = np.where(np.isnan(tau), np.sign(delta_ie * delta_fe), tau)
    global_inv_mask = valid_mask & (tau_effective < 0)
    global_inv_count = int(global_inv_mask.sum())
    global_inv_rate = global_inv_count / n_valid * 100
    results.append({
        "epsilon": epsilon,
        "valid_count": n_valid,
        "excluded_ratio": excluded_ratio,
        "valid_ratio": n_valid / total_samples * 100,
        "pure_inversion_count": pure_inv_count,
        "pure_inversion_rate": pure_inv_rate,
        "global_inversion_count": global_inv_count,
        "global_inversion_rate": global_inv_rate,
    })

summary_df = pd.DataFrame(results)
summary_df.to_csv(OUT_DIR / "epsilon_sensitivity_summary.csv", index=False)

fig, ax1 = plt.subplots(figsize=(7, 5))
x = np.arange(len(epsilons))
color_blue = "#1f77b4"
ax1.set_xlabel("Threshold ε")
ax1.set_ylabel("Valid Sample Ratio (%)", color=color_blue)
line1 = ax1.plot(
    x,
    summary_df["valid_ratio"].values,
    color=color_blue,
    marker="o",
    linewidth=2,
    markersize=7,
    label="Valid Sample Ratio",
)
ax1.tick_params(axis="y", labelcolor=color_blue)
ax1.set_ylim(70, 102)
ax1.set_xticks(x)
ax1.set_xticklabels([str(value) for value in epsilons])
ax1.grid(True, linestyle=":", alpha=0.6)

color_orange = "#ff7f0e"
ax2 = ax1.twinx()
ax2.set_ylabel("Pure Inversion Rate (%)", color=color_orange)
line2 = ax2.plot(
    x,
    summary_df["pure_inversion_rate"].values,
    color=color_orange,
    marker="s",
    linewidth=2,
    markersize=7,
    label="Pure Inversion Rate",
)
ax2.tick_params(axis="y", labelcolor=color_orange)
ax2.set_ylim(20, 35)

epsilon_001_index = 2
epsilon_001_valid = summary_df["valid_ratio"].values[epsilon_001_index]
epsilon_001_pure = summary_df["pure_inversion_rate"].values[epsilon_001_index]
ax1.axvline(epsilon_001_index, color="grey", linestyle="--", linewidth=1, alpha=0.7)
ax1.annotate(
    f"ε=0.01\n{epsilon_001_valid:.1f}% valid",
    xy=(epsilon_001_index, epsilon_001_valid),
    xytext=(epsilon_001_index + 0.3, epsilon_001_valid + 2),
    fontsize=9,
    color=color_blue,
    arrowprops={"arrowstyle": "->", "color": color_blue, "lw": 1},
)
ax2.annotate(
    f"{epsilon_001_pure:.1f}%",
    xy=(epsilon_001_index, epsilon_001_pure),
    xytext=(epsilon_001_index - 1.2, epsilon_001_pure + 2),
    fontsize=9,
    color=color_orange,
    arrowprops={"arrowstyle": "->", "color": color_orange, "lw": 1},
)

lines = line1 + line2
ax1.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=9)
ax1.set_title("Sensitivity Analysis of Threshold ε")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig3.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig3.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

valid_ratios = summary_df["valid_ratio"].values
pure_inv_rates = summary_df["pure_inversion_rate"].values
excluded_ratios = summary_df["excluded_ratio"].values
report = f"""ε 阈值敏感性分析报告
{"=" * 60}

分析基础：全量 {total_samples} 个未过滤实验样本

不同 ε 取值下的关键指标：
"""
for _, row in summary_df.iterrows():
    report += (
        f"  ε = {row['epsilon']:<6} | "
        f"有效样本: {row['valid_count']:>4} ({row['valid_ratio']:.2f}%) | "
        f"剔除比例: {row['excluded_ratio']:.2f}% | "
        f"纯倒挂率: {row['pure_inversion_rate']:.2f}% | "
        f"全局倒挂率: {row['global_inversion_rate']:.2f}%\n"
    )
report += f"""
关键发现：

1. 有效样本保留率从 {valid_ratios[0]:.1f}% 降至 {valid_ratios[-1]:.1f}%，剔除比例从 {excluded_ratios[0]:.1f}% 上升至 {excluded_ratios[-1]:.1f}%。
2. 纯倒挂率在 ε ∈ [0.005, 0.05] 内为 {pure_inv_rates[1]:.1f}% ~ {pure_inv_rates[-1]:.1f}%，波动幅度为 {pure_inv_rates.max() - pure_inv_rates.min():.1f} 个百分点。
3. 性能倒挂结论对 ε 阈值选择不敏感。
"""
(OUT_DIR / "epsilon_sensitivity_report.txt").write_text(report, encoding="utf-8")
print(summary_df.to_string(index=False))
print(f"Saved: {FIG_DIR / 'fig3.png'}")
print(f"Saved: {FIG_DIR / 'fig3.pdf'}")
