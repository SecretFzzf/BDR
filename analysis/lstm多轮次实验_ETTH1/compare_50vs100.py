# -*- coding: utf-8 -*-
"""
50-epoch vs 100-epoch LSTM 结果对比分析（只读，不修改任何 CSV）。
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "lstm多轮次实验_ETTH1"
A = pd.read_csv(RESULTS_DIR / "test1.csv")                # max_epochs=50
B = pd.read_csv(RESULTS_DIR / "results_refactor_lstm.csv")  # max_epochs=100
B['dataset'] = 'ETTh1'
KEY = ['missing_mode', 'impute_method', 'missing_rate', 'pred_len']
GRP = ['missing_mode', 'missing_rate', 'pred_len']

assert (A['status'] == 'success').all() and (B['status'] == 'success').all()
assert len(A) == len(B) == 240
assert not A.duplicated(KEY).any() and not B.duplicated(KEY).any()

out = []

def line(s=''):
    out.append(s)
    print(s)

# ---------- 1. 列结构 ----------
line('=' * 78)
line('1) 列结构对比')
line('-' * 78)
line('test1.csv                   (50ep): ' + ', '.join(A.columns))
line('results_refactor_lstm.csv (100ep): ' + ', '.join(B.columns))
line('唯一差异: 100ep 文件缺少 dataset 列 (已按 ETTh1 补全); 两者其余列完全一致。')

# ---------- 2. 倒挂率 ----------
def inversion_stats(df):
    mean_fe = df[df['impute_method'] == 'mean'].groupby(GRP)['fe_mae'].mean().rename('fe_mean')
    mean_ie = df[df['impute_method'] == 'mean'].groupby(GRP)['ie_mae'].mean().rename('ie_mean')
    df = df.merge(mean_fe, on=GRP, how='left').merge(mean_ie, on=GRP, how='left')
    df['delta_fe'] = df['fe_mae'] - df['fe_mean']
    df['delta_ie'] = df['ie_mae'] - df['ie_mean']
    phi = df[df['impute_method'] != 'mean'].copy()
    return dict(
        global_inv=(phi['delta_fe'] > 0).mean() * 100,
        pure_inv=((phi['delta_ie'] < 0) & (phi['delta_fe'] > 0)).mean() * 100,
        n=len(phi),
    )

sA, sB = inversion_stats(A.copy()), inversion_stats(B.copy())
line('')
line('=' * 78)
line('2) 全样本倒挂率 (ETTh1, 非Mean样本 N=%d)' % sA['n'])
line('-' * 78)
line('                  50-epoch    100-epoch    变化(pp)')
line('全局倒挂率(ΔFE>0): %7.2f%%  %8.2f%%   %+6.2f' % (sA['global_inv'], sB['global_inv'], sB['global_inv'] - sA['global_inv']))
line('纯倒挂率(ΔIE<0&ΔFE>0): %6.2f%%  %7.2f%%   %+6.2f' % (sA['pure_inv'], sB['pure_inv'], sB['pure_inv'] - sA['pure_inv']))

# 按预测模型维度没有, 按缺失率/模式分层
line('')
line('按缺失率分层 全局倒挂率 (%):')
line('  rate   50ep    100ep   变化')
for rate in sorted(A['missing_rate'].unique()):
    ra = inversion_stats(A[A['missing_rate'] == rate])['global_inv']
    rb = inversion_stats(B[B['missing_rate'] == rate])['global_inv']
    line('  %4.1f  %6.2f  %7.2f  %+6.2f' % (rate, ra, rb, rb - ra))
line('按缺失机制分层 全局倒挂率 (%):')
line('  mode        50ep    100ep   变化')
for mode in sorted(A['missing_mode'].unique()):
    ra = inversion_stats(A[A['missing_mode'] == mode])['global_inv']
    rb = inversion_stats(B[B['missing_mode'] == mode])['global_inv']
    line('  %-10s %6.2f  %7.2f  %+6.2f' % (mode, ra, rb, rb - ra))

# ---------- 3. 配对逐条件对比 ----------
MA = A[KEY + ['ie_mae', 'fe_mae', 'fe_mse', 'best_val_loss', 'best_epoch', 'elapsed_sec']].rename(
    columns={c: c + '_50' for c in ['ie_mae', 'fe_mae', 'fe_mse', 'best_val_loss', 'best_epoch', 'elapsed_sec']})
MB = B[KEY + ['ie_mae', 'fe_mae', 'fe_mse', 'best_val_loss', 'best_epoch', 'elapsed_sec']].rename(
    columns={c: c + '_100' for c in ['ie_mae', 'fe_mae', 'fe_mse', 'best_val_loss', 'best_epoch', 'elapsed_sec']})
M = MA.merge(MB, on=KEY, how='inner')

M['d_fe'] = M['fe_mae_100'] - M['fe_mae_50']
M['d_fe_mse'] = M['fe_mse_100'] - M['fe_mse_50']
M['d_best_epoch'] = M['best_epoch_100'] - M['best_epoch_50']
M['d_val'] = M['best_val_loss_100'] - M['best_val_loss_50']
M['d_ie'] = M['ie_mae_100'] - M['ie_mae_50']

line('')
line('=' * 78)
line('3) 逐条件配对对比 (240 条件, 各条件 50ep vs 100ep)')
line('-' * 78)
line('FE_MAE 变化:    mean=%.4f  median=%.4f  std=%.4f  改善条件数=%d/%d' %
     (M['d_fe'].mean(), M['d_fe'].median(), M['d_fe'].std(), (M['d_fe'] < 0).sum(), len(M)))
line('FE_MSE 变化:    mean=%.4f  median=%.4f  std=%.4f  改善条件数=%d/%d' %
     (M['d_fe_mse'].mean(), M['d_fe_mse'].median(), M['d_fe_mse'].std(), (M['d_fe_mse'] < 0).sum(), len(M)))
line('IE_MAE 变化:    mean=%.5f  median=%.5f' % (M['d_ie'].mean(), M['d_ie'].median()))
line('best_val_loss 变化: mean=%.5f  median=%.5f  改善条件数=%d/%d' %
     (M['d_val'].mean(), M['d_val'].median(), (M['d_val'] < 0).sum(), len(M)))
line('best_epoch 变化:  mean=%+.1f  median=%+.1f  max=%d' % (M['d_best_epoch'].mean(), M['d_best_epoch'].median(), M['d_best_epoch'].max()))
line('elapsed_sec 变化:  mean=%+.1f  median=%+.1f' %
     ((M['elapsed_sec_100'] - M['elapsed_sec_50']).mean(), (M['elapsed_sec_100'] - M['elapsed_sec_50']).median()))

line('')
line('best_epoch 分布:')
line('  50ep: min=%d med=%.0f max=%d | best_epoch==50 占比 %.1f%%' %
     (M['best_epoch_50'].min(), M['best_epoch_50'].median(), M['best_epoch_50'].max(), (M['best_epoch_50'] == 50).mean() * 100))
line(' 100ep: min=%d med=%.0f max=%d | best_epoch>50 占比 %.1f%% | best_epoch>=50 占比 %.1f%%' %
     (M['best_epoch_100'].min(), M['best_epoch_100'].median(), M['best_epoch_100'].max(),
      (M['best_epoch_100'] > 50).mean() * 100, (M['best_epoch_100'] >= 50).mean() * 100))

# FE 均值/中位数
line('')
line('FE_MAE 汇总 (非Mean 192 样本):')
line('  50ep:  mean=%.4f  median=%.4f' % (M.loc[M['impute_method'] != 'mean', 'fe_mae_50'].mean(), M.loc[M['impute_method'] != 'mean', 'fe_mae_50'].median()))
line(' 100ep:  mean=%.4f  median=%.4f' % (M.loc[M['impute_method'] != 'mean', 'fe_mae_100'].mean(), M.loc[M['impute_method'] != 'mean', 'fe_mae_100'].median()))
line(' 相对变化: mean %+.2f%%  median %+.2f%%' %
     ((M.loc[M['impute_method'] != 'mean', 'fe_mae_100'].mean() / M.loc[M['impute_method'] != 'mean', 'fe_mae_50'].mean() - 1) * 100,
      (M.loc[M['impute_method'] != 'mean', 'fe_mae_100'].median() / M.loc[M['impute_method'] != 'mean', 'fe_mae_50'].median() - 1) * 100))
line('FE_MSE 汇总 (非Mean 192 样本):')
line('  50ep:  mean=%.4f  median=%.4f' % (M.loc[M['impute_method'] != 'mean', 'fe_mse_50'].mean(), M.loc[M['impute_method'] != 'mean', 'fe_mse_50'].median()))
line(' 100ep:  mean=%.4f  median=%.4f' % (M.loc[M['impute_method'] != 'mean', 'fe_mse_100'].mean(), M.loc[M['impute_method'] != 'mean', 'fe_mse_100'].median()))

# ---------- 4. 汇总表 ----------
line('')
line('=' * 78)
line('4) 对比汇总表')
line('-' * 78)
line('Dataset | 50-epoch Inversion | 100-epoch Inversion | Change(pp) | 50 Best Epoch | 100 Best Epoch')
line('ETTh1   | %6.2f%%             | %6.2f%%              | %+6.2f      | med=%.0f (max %d) | med=%.0f (max %d)' %
     (sA['global_inv'], sB['global_inv'], sB['global_inv'] - sA['global_inv'],
      M['best_epoch_50'].median(), M['best_epoch_50'].max(),
      M['best_epoch_100'].median(), M['best_epoch_100'].max()))

with open(RESULTS_DIR / "compare_50vs100_report.txt", "w", encoding="utf-8") as f:
    f.write('\n'.join(out))
print('\n[written] compare_50vs100_report.txt')
