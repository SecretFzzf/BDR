# -*- coding: utf-8 -*-
"""
Post-processing: Simultaneously output revision items ③(η distributed)、④(MSE sensitivity)、⑤(mixed pool IQR unified)。
Pure reading CSV，Do not modify any experimental data。
"""
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path

EPSILON = 0.01
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SRC_ALL = REPO_ROOT / "results" / "seed42" / "all_models_result.csv"
SRC_TAU = REPO_ROOT / "results" / "seed42" / "tau_results.csv"
OUT_DIR = REPO_ROOT / "results" / "tau_stratified"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_OUT = OUT_DIR / "eta_mse_extra_report.txt"
REPORT_TEXT = StringIO()


def emit(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=REPORT_TEXT)

# ============================================================
# ③ η Distributed (directly by tau_results.csv of MAE version calculation）
# ============================================================
tau = pd.read_csv(SRC_TAU)
tau = tau[tau["tau_status"] == "valid"].copy()

# η = (ΔFE/FE_mean) / (ΔIE/IE_mean)  —— English version eq:eta
tau["eta"] = (tau["delta_fe"] / tau["fe_mae_mean"]) / (tau["delta_ie"] / tau["ie_mae_mean"])

# Symbol consistency rate：η and τ Ratio of the same number
tau["sign_agree"] = np.sign(tau["eta"]) == np.sign(tau["tau"])

emit("="*70)
emit("③ η distributed")
emit("="*70)
emit(f"N = {len(tau)}")
emit(f"η median   = {tau['eta'].median():.6f}")
emit(f"η IQR      = [{tau['eta'].quantile(0.25):.6f}, {tau['eta'].quantile(0.75):.6f}]")
emit(f"η mean     = {tau['eta'].mean():.6f}")
emit(f"η <0 Proportion  = {(tau['eta'] < 0).mean():.1%}")
emit("\nby model:")
mm = tau.groupby("model")["eta"].agg(count="count", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
emit(mm.round(6).to_string())
emit("\nPress interpolation:")
mi = tau.groupby("impute_method")["eta"].agg(count="count", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
emit(mi.round(6).to_string())
emit(f"\nη and τ Symbol consistency rate = {tau['sign_agree'].mean():.1%}")

# ============================================================
# ④ MSE version τ（replica compute_tau.py logic，MAE -> MSE）
# ============================================================
df = pd.read_csv(SRC_ALL)
df = df[df["status"] == "success"].copy()

def build_tau(var_ie, var_fe):
    """replica compute_tau.py Aggregation caliber, change error column。"""
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

emit("\n" + "="*70)
emit("④ MSE sensitivity：τ Depend on MSE version error recalculation")
emit("="*70)
emit(f"N(valid) = {len(tau_mse)}")
q25m, q75m = tau_mse['tau'].quantile(0.25), tau_mse['tau'].quantile(0.75)
q25a, q75a = tau['tau'].quantile(0.25), tau['tau'].quantile(0.75)
emit(f"τ_MSE median   = {tau_mse['tau'].median():.6f}   (MAE version: {tau['tau'].median():.6f})")
emit(f"τ_MSE IQR      = {q75m - q25m:.6f}   (MAE version: {q75a - q25a:.6f})")
# pure inversion rate (MAE vs MSE)
inv_mse = ((tau_mse["delta_ie"]<0)&(tau_mse["delta_fe"]>0)).mean()*100
inv_mae = ((tau["delta_ie"]<0)&(tau["delta_fe"]>0)).mean()*100
emit(f"pure inversion rate(MSE)   = {inv_mse:.2f}%")
emit(f"pure inversion rate(MAE)   = {inv_mae:.2f}%")
emit(f"Global inversion rate τ<0 (MSE) = {(tau_mse['tau']<0).mean()*100:.2f}%   (MAE: {(tau['tau']<0).mean()*100:.2f}%)")

REPORT_OUT.write_text(REPORT_TEXT.getvalue(), encoding="utf-8")
