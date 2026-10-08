# -*- coding: utf-8 -*-
"""
后处理：同时产出修订项 ③(η 分布)、④(MSE 敏感性)、⑤(混池 IQR 统一)。
纯读 CSV，不修改任何实验数据。
"""
import pandas as pd
import numpy as np
from pathlib import Path

EPSILON = 0.01
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SRC_ALL = REPO_ROOT / "results" / "seed42" / "all_models_result.csv"
SRC_TAU = REPO_ROOT / "results" / "seed42" / "tau_results.csv"
OUT_DIR = REPO_ROOT / "results" / "计算τ并且分层"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# ③ η 分布（直接由 tau_results.csv 的 MAE 版计算）
# ============================================================
tau = pd.read_csv(SRC_TAU)
tau = tau[tau["tau_status"] == "valid"].copy()

# η = (ΔFE/FE_mean) / (ΔIE/IE_mean)  —— 英文稿 eq:eta
tau["eta"] = (tau["delta_fe"] / tau["fe_mae_mean"]) / (tau["delta_ie"] / tau["ie_mae_mean"])

# 符号一致率：η 与 τ 同号的比例
tau["sign_agree"] = np.sign(tau["eta"]) == np.sign(tau["tau"])

print("="*70)
print("③ η 分布")
print("="*70)
print(f"N = {len(tau)}")
print(f"η 中位数   = {tau['eta'].median():.6f}")
print(f"η IQR      = [{tau['eta'].quantile(0.25):.6f}, {tau['eta'].quantile(0.75):.6f}]")
print(f"η 均值     = {tau['eta'].mean():.6f}")
print(f"η <0 比例  = {(tau['eta'] < 0).mean():.1%}")
print("\n按模型:")
mm = tau.groupby("model")["eta"].agg(count="count", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
print(mm.round(6).to_string())
print("\n按插补:")
mi = tau.groupby("impute_method")["eta"].agg(count="count", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
print(mi.round(6).to_string())
print(f"\nη 与 τ 符号一致率 = {tau['sign_agree'].mean():.1%}")

# ============================================================
# ④ MSE 版 τ（复刻 compute_tau.py 逻辑，MAE -> MSE）
# ============================================================
df = pd.read_csv(SRC_ALL)
df = df[df["status"] == "success"].copy()

def build_tau(var_ie, var_fe):
    """复刻 compute_tau.py 聚合口径，改误差列。"""
    ie_table = (
        df.groupby(["dataset", "impute_method", "missing_mode", "missing_rate"])[var_ie]
        .mean().reset_index().rename(columns={var_ie: "ie_phi"})
    )
    ie_mean = (
        ie_table[ie_table["impute_method"] == "mean"]
        .drop(columns=["impute_method"]).rename(columns={"ie_phi": "ie_mean"})
    )
    fe_mean = (
        df[df["impute_method"] == "mean"]
        [["dataset", "model", "missing_mode", "missing_rate", "pred_len", var_fe]]
        .rename(columns={var_fe: "fe_mean"})
    )
    out = df[df["impute_method"] != "mean"].copy()
    out = out.merge(ie_table, on=["dataset", "impute_method", "missing_mode", "missing_rate"], how="left")
    out = out.merge(ie_mean, on=["dataset", "missing_mode", "missing_rate"], how="left")
    out = out.merge(fe_mean, on=["dataset", "model", "missing_mode", "missing_rate", "pred_len"], how="left")
    out["delta_ie"] = out["ie_phi"] - out["ie_mean"]
    out["delta_fe"] = out[var_fe] - out["fe_mean"]
    guard = out["delta_ie"].abs() > EPSILON
    out["tau"] = np.where(guard, out["delta_fe"] / out["delta_ie"], np.nan)
    return out

tau_mse = build_tau("ie_mse", "fe_mse")
tau_mse = tau_mse[tau_mse["tau"].notna()].copy()

print("\n" + "="*70)
print("④ MSE 敏感性：τ 由 MSE 版误差重算")
print("="*70)
print(f"N(valid) = {len(tau_mse)}")
q25m, q75m = tau_mse['tau'].quantile(0.25), tau_mse['tau'].quantile(0.75)
q25a, q75a = tau['tau'].quantile(0.25), tau['tau'].quantile(0.75)
print(f"τ_MSE 中位数   = {tau_mse['tau'].median():.6f}   (MAE 版: {tau['tau'].median():.6f})")
print(f"τ_MSE IQR      = {q75m - q25m:.6f}   (MAE 版: {q75a - q25a:.6f})")
# 纯倒挂率 (MAE vs MSE)
inv_mse = ((tau_mse["delta_ie"]<0)&(tau_mse["delta_fe"]>0)).mean()*100
inv_mae = ((tau["delta_ie"]<0)&(tau["delta_fe"]>0)).mean()*100
print(f"纯倒挂率(MSE)   = {inv_mse:.2f}%")
print(f"纯倒挂率(MAE)   = {inv_mae:.2f}%")
print(f"全局倒挂率 τ<0 (MSE) = {(tau_mse['tau']<0).mean()*100:.2f}%   (MAE: {(tau['tau']<0).mean()*100:.2f}%)")
