"""
ETTh1 5种子合并与符号翻转率（Sign Flip Rate）分析
--------------------------------------------------
输入：results/反向的四个种子验证/ETT_seed{42,123,2026,456,789}/
      results_refactor_{lstm,dlinear,patchtst}.csv
输出：
  - 控制台：三维统计汇总表
  - 磁盘：merged_5seeds_ett_results.csv
"""

import os
import glob
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
BASE = str(REPO_ROOT / "results" / "反向的四个种子验证")
OUT_CSV = os.path.join(BASE, "merged_5seeds_ett_results.csv")

# ── 1. 递归读取所有 result_refactor_*.csv ──────────────
rows = []
for folder in sorted(glob.glob(os.path.join(BASE, "ETT_seed*"))):
    seed_str = folder.split("_")[-1]
    seed = int(seed_str.replace("seed", ""))
    csv_files = glob.glob(os.path.join(folder, "results_refactor_*.csv"))
    for f in csv_files:
        df = pd.read_csv(f)
        df["seed"] = seed
        rows.append(df)
        print(f"[读取] seed={seed:4d}  {os.path.basename(f):35s}  {len(df)} 行")

raw = pd.concat(rows, ignore_index=True)
print(f"\n合并后原始记录数: {len(raw)}")

# ── 2. 清洗 ─────────────────────────────────────────────
# 只保留成功行
raw = raw[raw["status"] == "success"].copy()

# 统一列名：dataset 固定为 ETTh1（文件夹已隐含）
raw["dataset"] = "ETTh1"

# 标准化 impute_method 大小写
raw["impute_method"] = raw["impute_method"].str.lower()

# 标准化 missing_mode
raw["missing_mode"] = raw["missing_mode"].str.replace("MAR_Block", "MAR_Block", regex=False)

# 只保留需要的列
KEEP_COLS = ["dataset", "model", "missing_mode", "impute_method",
             "missing_rate", "pred_len", "ie_mae", "fe_mae", "seed"]
df = raw[KEEP_COLS].copy()

# ── 3. 计算 ΔFE = fe_phi - fe_mean ─────────────────────
# fe_mean：每个 (model, seed, missing_mode, missing_rate, pred_len) 下 Mean 的 fe_mae
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

# 去掉 mean 自身（不参与翻转率计算）
df_phi = df[df["impute_method"] != "mean"].copy()

print(f"清洗后（不含 Mean）: {len(df_phi)} 行")
print(f"种子覆盖: {sorted(df_phi['seed'].unique().tolist())}")
print(f"模型覆盖: {sorted(df_phi['model'].unique().tolist())}")
print()

# ── 4. 维度一：样本级符号翻转率 ────────────────────────
# 实验单元主键
UNIT_KEYS = ["model", "impute_method", "missing_mode", "missing_rate", "pred_len"]

def sign_flip_rate(sign_series):
    """组内 5 种子间符号不一致对的比例。传入的是 sign 列的 Series。"""
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

# 只保留有 ≥2 种种子的单元（否则翻转率无定义）
unit_valid = unit_stats[unit_stats["n_seeds"] >= 2].copy()

mean_sfr = unit_valid["sign_flip_rate"].mean()
median_sfr = unit_valid["sign_flip_rate"].median()
deterministic_pct = (unit_valid["sign_flip_rate"] == 0).sum() / len(unit_valid) * 100

print("=" * 65)
print("  维度一：样本级符号翻转率 (Sign Flip Rate)")
print("=" * 65)
print(f"  有效实验单元数 (≥2 种子):  {len(unit_valid)}")
print(f"  平均符号翻转率:            {mean_sfr:.4f}  ({mean_sfr*100:.2f}%)")
print(f"  中位数符号翻转率:          {median_sfr:.4f}  ({median_sfr*100:.2f}%)")
print(f"  确定性不变 (翻转率=0%) 占比: {deterministic_pct:.2f}%")
print()

# ── 5. 维度二：按算法组合 (Model × Imputer) 汇总 ──────
algo_stats = (
    df_phi.groupby(["model", "impute_method"])
    .agg(
        评估样本数=("delta_fe", "count"),
        倒挂样本数=("inversion", "sum"),
        纯倒挂率=("inversion", "mean"),
        delta_fe均值=("delta_fe", "mean"),
        delta_fe中位数=("delta_fe", "median"),
    )
    .reset_index()
)

# 合并该组合下的平均符号翻转率
algo_sfr = (
    unit_valid.groupby(["model", "impute_method"])["sign_flip_rate"]
    .mean()
    .rename("平均符号翻转率")
    .reset_index()
)
algo_stats = algo_stats.merge(algo_sfr, on=["model", "impute_method"], how="left")

algo_stats["纯倒挂率"] = algo_stats["纯倒挂率"] * 100

print("=" * 65)
print("  维度二：按算法组合 (Model × Imputer) 汇总")
print("=" * 65)
print(algo_stats.to_string(index=False))
print()

# ── 6. 维度三：按随机种子 (Seed) 汇总 ─────────────────
seed_stats = (
    df_phi.groupby("seed")
    .agg(
        评估样本数=("delta_fe", "count"),
        倒挂样本数=("inversion", "sum"),
        纯倒挂率=("inversion", "mean"),
        delta_fe中位数=("delta_fe", "median"),
        delta_fe均值=("delta_fe", "mean"),
    )
    .reset_index()
)
seed_stats["纯倒挂率"] = seed_stats["纯倒挂率"] * 100
seed_stats["delta_fe中位数"] = seed_stats["delta_fe中位数"].round(4)
seed_stats["delta_fe均值"] = seed_stats["delta_fe均值"].round(4)

print("=" * 65)
print("  维度三：按随机种子 (Seed) 汇总")
print("=" * 65)
print(seed_stats.to_string(index=False))
print()

# ── 7. 导出全量合并 CSV ────────────────────────────────
# seq_len 在所有文件中均为 96，直接从 raw 取一列补入
seq_len_map = raw[["model", "seed", "missing_mode", "impute_method",
                   "missing_rate", "pred_len", "seq_len"]].drop_duplicates()
df = df.merge(seq_len_map,
              on=["model", "seed", "missing_mode", "impute_method",
                  "missing_rate", "pred_len"],
              how="left", suffixes=("", "_dup"))
# 去掉可能的重复列
dup_cols = [c for c in df.columns if c.endswith("_dup")]
df = df.drop(columns=dup_cols)

export_cols = ["dataset", "model", "missing_mode", "impute_method",
               "missing_rate", "pred_len", "seq_len",
               "ie_mae", "fe_mae", "fe_mean", "delta_fe", "sign", "inversion", "seed"]
df[export_cols].sort_values(["seed", "model", "impute_method",
                              "missing_mode", "missing_rate", "pred_len"]
                            ).to_csv(OUT_CSV, index=False)
print(f"[写入] {OUT_CSV}  ({len(df)} 行)")
