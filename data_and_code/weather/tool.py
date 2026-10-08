import numpy as np
import pandas as pd
import torch
import inspect
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import KNNImputer
from pypots.imputation import SAITS, BRITS

# 统一硬件设备检测：优先 CUDA，其次 MPS，最后 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def _build_imputer_kwargs(model_cls, kwargs):
    """根据当前 PyPOTS 版本过滤可用参数，避免参数不兼容报错。"""
    signature = inspect.signature(model_cls.__init__)
    valid_keys = set(signature.parameters.keys())
    valid_keys.discard("self")
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def _resolve_dl_impute_train_cfg(method, chunk_len, n_features):
    """按数据规模给 SAITS/BRITS 分配训练轮数与早停参数。"""
    method = str(method).lower()
    # 基线轮数：由特征数与序列长度驱动。
    base_epochs = 50
    if n_features >= 128:
        base_epochs += 20
    if n_features >= 192:
        base_epochs += 20
    if chunk_len >= 128:
        base_epochs += 20

    # SAITS 通常对复杂模式更稳，给略多轮次。
    if method == "saits":
        max_epochs = min(160, base_epochs)
        patience = max(10, max_epochs // 3)
    else:
        # BRITS 迭代代价略高，给稍保守轮数。
        max_epochs = min(120, max(40, base_epochs - 10))
        patience = max(10, max_epochs // 3)

    # BRITS 在超长序列上单样本训练会非常慢，限制轮数并依赖早停。
    if method == "brits":
        if chunk_len >= 4096:
            max_epochs = min(max_epochs, 12)
            patience = min(patience, 4)
        elif chunk_len >= 2048:
            max_epochs = min(max_epochs, 16)
            patience = min(patience, 5)
        elif chunk_len >= 1024:
            max_epochs = min(max_epochs, 20)
            patience = min(patience, 6)
        elif chunk_len >= 384:
            max_epochs = min(max_epochs, 26)
            patience = min(patience, 8)
        elif chunk_len >= 192:
            max_epochs = min(max_epochs, 30)
            patience = min(patience, 10)

    return {
        "epochs": int(max_epochs),
        "patience": int(patience),
    }


def _pack_windows_for_imputation(data_2d, window_len, stride):
    """将长序列打包为批量窗口，提升 BRITS 的并行度。"""
    total_steps = int(data_2d.shape[0])
    starts = list(range(0, max(1, total_steps - window_len + 1), stride))
    if not starts:
        starts = [0]
    last_start = max(0, total_steps - window_len)
    if starts[-1] != last_start:
        starts.append(last_start)

    windows = []
    for s in starts:
        e = s + window_len
        windows.append(data_2d[s:e])

    return np.stack(windows, axis=0), starts


def _merge_window_imputations(original_data, imputed_windows, starts):
    """将窗口插补结果回写到整段序列；重叠区域取平均。"""
    merged = np.zeros_like(original_data, dtype=np.float32)
    counts = np.zeros((original_data.shape[0], 1), dtype=np.float32)
    win_len = int(imputed_windows.shape[1])

    for idx, s in enumerate(starts):
        e = s + win_len
        merged[s:e] += imputed_windows[idx]
        counts[s:e] += 1.0

    counts[counts == 0] = 1.0
    merged = merged / counts
    return merged.astype(original_data.dtype, copy=False)

# ==========================================
# 核心辅助函数：统一处理插补与误差评估 (对外隐藏)
# ==========================================
def _impute_and_evaluate(train_data_clean, mask, method, missing_rate, missing_type_str, seq_len=96, verbose=True, fitted_imputer=None):
    train_data_missing = train_data_clean.copy()
    train_data_missing[mask] = np.nan
    train_data_imputed = train_data_missing.copy()
    n_steps, n_features = train_data_imputed.shape
    fitted = fitted_imputer

    if method == 'mean':
        if fitted is None:
            fitted = np.nanmean(train_data_imputed, axis=0)
        col_means = fitted
        inds = np.where(np.isnan(train_data_imputed))
        train_data_imputed[inds] = np.take(col_means, inds[1])
        method_name_cn = "均值"
    elif method == 'spline':
        df_temp = pd.DataFrame(train_data_imputed)
        valid_counts = df_temp.notna().sum(axis=0)
        min_valid = int(valid_counts.min())
        if min_valid >= 2:
            order = 3 if min_valid >= 4 else max(1, min_valid - 1)
            df_imputed = df_temp.interpolate(method='spline', order=order, limit_direction='both')
        else:
            df_imputed = df_temp.copy()
        df_imputed = df_imputed.bfill().ffill()
        train_data_imputed = df_imputed.values
        method_name_cn = "样条"
    elif method == 'knn':
        if fitted is None:
            fitted = KNNImputer(n_neighbors=5, weights='distance')
            train_data_imputed = fitted.fit_transform(train_data_imputed)
        else:
            train_data_imputed = fitted.transform(train_data_imputed)
        method_name_cn = "KNN"
    elif method == 'saits':
        # 长序列直接使用分块模式，避免 OOM（pypots 会包装异常导致 fallback 失效）。
        use_chunk_mode = n_steps > 2048

        if use_chunk_mode:
            chunk_size = 1024  # 固定分块大小，确保显存安全
            print(f"🚀 正在训练 SAITS 插补模型（分块模式）... chunks={n_steps // chunk_size + 1}, chunk_size={chunk_size}")
            for start in range(0, n_steps, chunk_size):
                end = min(start + chunk_size, n_steps)
                chunk = train_data_imputed[start:end]
                chunk_len = end - start
                if chunk_len <= 1:
                    continue
                chunk_dataset = {"X": chunk.reshape(1, chunk_len, n_features)}
                chunk_cfg = _resolve_dl_impute_train_cfg(method='saits', chunk_len=chunk_len, n_features=n_features)
                print(
                    f"   [SAITS] chunk=({start}:{end}/{n_steps}), epochs={chunk_cfg['epochs']}, "
                    f"patience={chunk_cfg['patience']}"
                )
                chunk_kwargs = _build_imputer_kwargs(SAITS, {
                    "n_steps": chunk_len,
                    "n_features": n_features,
                    "n_layers": 2,
                    "d_model": 256,
                    "n_heads": 4,
                    "d_k": 64,
                    "d_v": 64,
                    "d_ffn": 128,
                    "dropout": 0.1,
                    "epochs": chunk_cfg['epochs'],
                    "patience": chunk_cfg['patience'],
                    "device": str(device),
                })
                saits_chunk = SAITS(**chunk_kwargs)
                saits_chunk.fit(chunk_dataset)
                imputed_chunk = saits_chunk.impute(chunk_dataset)
                train_data_imputed[start:end] = imputed_chunk.reshape(chunk_len, n_features)
        else:
            print("🚀 正在训练 SAITS 插补模型（全量单次模式）...")
            full_dataset = {"X": train_data_imputed.reshape(1, n_steps, n_features)}
            train_cfg = _resolve_dl_impute_train_cfg(method='saits', chunk_len=n_steps, n_features=n_features)
            print(
                f"   [SAITS] full_len={n_steps}, epochs={train_cfg['epochs']}, "
                f"patience={train_cfg['patience']}"
            )
            saits_kwargs = _build_imputer_kwargs(SAITS, {
                "n_steps": n_steps,
                "n_features": n_features,
                "n_layers": 2,
                "d_model": 256,
                "n_heads": 4,
                "d_k": 64,
                "d_v": 64,
                "d_ffn": 128,
                "dropout": 0.1,
                "epochs": train_cfg['epochs'],
                "patience": train_cfg['patience'],
                "device": str(device),
            })
            saits = SAITS(**saits_kwargs)
            saits.fit(full_dataset)
            imputed_result = saits.impute(full_dataset)
            train_data_imputed = imputed_result.reshape(n_steps, n_features)
        method_name_cn = "SAITS"
    elif method == 'brits':
        # 超长序列下，batch=1 的全量训练无法吃满 GPU；改用窗口批训练一次完成。
        use_window_batch_mode = n_steps > 512

        if use_window_batch_mode:
            # RNN 对长序列串行成本极高：缩短窗口并提高并发窗口数量。
            window_len = int(min(max(seq_len * 2, 192), 384))
            # 使用无重叠窗口，避免重复计算。
            stride = int(window_len)
            window_data, starts = _pack_windows_for_imputation(train_data_imputed, window_len=window_len, stride=stride)
            train_cfg = _resolve_dl_impute_train_cfg(method='brits', chunk_len=window_len, n_features=n_features)
            print(
                f"🚀 正在训练 BRITS 插补模型（窗口批加速模式）... windows={len(starts)}, "
                f"window_len={window_len}, epochs={train_cfg['epochs']}, patience={train_cfg['patience']}"
            )
            # 批大小对齐窗口总数，尽量一次并行喂满 GPU。
            batch_size = max(1, int(len(starts)))
            dataset_for_pypots = {"X": window_data}
            brits_kwargs = _build_imputer_kwargs(BRITS, {
                "n_steps": window_len,
                "n_features": n_features,
                "rnn_hidden_size": 128,
                "batch_size": batch_size,
                "epochs": train_cfg['epochs'],
                "patience": train_cfg['patience'],
                "device": str(device),
            })
            brits = BRITS(**brits_kwargs)
            brits.fit(dataset_for_pypots)
            imputed_windows = brits.impute(dataset_for_pypots)
            train_data_imputed = _merge_window_imputations(train_data_imputed, imputed_windows, starts)
        else:
            print("🚀 正在训练 BRITS 插补模型（全量单次模式）...")
            full_dataset = {"X": train_data_imputed.reshape(1, n_steps, n_features)}
            train_cfg = _resolve_dl_impute_train_cfg(method='brits', chunk_len=n_steps, n_features=n_features)
            print(
                f"   [BRITS] full_len={n_steps}, epochs={train_cfg['epochs']}, "
                f"patience={train_cfg['patience']}"
            )
            brits_kwargs = _build_imputer_kwargs(BRITS, {
                "n_steps": n_steps,
                "n_features": n_features,
                "rnn_hidden_size": 96,
                "epochs": train_cfg['epochs'],
                "patience": train_cfg['patience'],
                "device": str(device),
            })
            brits = BRITS(**brits_kwargs)
            brits.fit(full_dataset)
            imputed_result = brits.impute(full_dataset)
            train_data_imputed = imputed_result.reshape(n_steps, n_features)
        method_name_cn = "BRITS"
    else:
        raise ValueError("不支持的插补方法")

    channel_maes = []
    channel_mses = []
    for col in range(n_features):
        channel_mask = mask[:, col]
        if not np.any(channel_mask):
            continue

        true_col = train_data_clean[channel_mask, col]
        imputed_col = train_data_imputed[channel_mask, col]
        channel_maes.append(mean_absolute_error(true_col, imputed_col))
        channel_mses.append(mean_squared_error(true_col, imputed_col))

    if channel_maes:
        ie_mae = float(np.mean(channel_maes))
        ie_mse = float(np.mean(channel_mses))
    else:
        ie_mae, ie_mse = 0.0, 0.0

    print(f"--- 第一阶段：{missing_type_str} 插补质量 (IE) ---")
    print(f"{int(missing_rate*100)}% {missing_type_str} + {method_name_cn}插补 MAE: {ie_mae:.4f}")
    print(f"{int(missing_rate*100)}% {missing_type_str} + {method_name_cn}插补 MSE: {ie_mse:.4f}\n")

    return train_data_imputed, ie_mae, ie_mse, fitted

# ==========================================
# 机制 1：MCAR 完全随机缺失
# ==========================================
def inject_mcar_missing_and_impute(train_data_clean, missing_rate=0.30, method='mean', seed=42, verbose=True, fitted_imputer=None):
    np.random.seed(seed)
    rows, cols = train_data_clean.shape
    valid_mask = ~np.isnan(train_data_clean)
    mask = np.zeros((rows, cols), dtype=bool)

    target_missing_count = int(valid_mask.sum() * missing_rate)
    if target_missing_count > 0:
        valid_coords = np.argwhere(valid_mask)
        chosen = np.random.choice(
            len(valid_coords),
            size=min(target_missing_count, len(valid_coords)),
            replace=False,
        )
        coords = valid_coords[chosen]
        mask[coords[:, 0], coords[:, 1]] = True

    return _impute_and_evaluate(train_data_clean, mask, method, missing_rate, "MCAR", verbose=verbose, fitted_imputer=fitted_imputer)

# ==========================================
# 机制 2：MAR - 全局块状缺失 (Block Masking across all variables)
# ==========================================
def inject_mar_block_missing_and_impute(train_data_clean, missing_rate=0.30, block_size=24, method='mean', seed=42, verbose=True, fitted_imputer=None):
    np.random.seed(seed)
    rows, cols = train_data_clean.shape
    valid_mask = ~np.isnan(train_data_clean)
    mask = np.zeros((rows, cols), dtype=bool)
    target_missing_count = int(valid_mask.sum() * missing_rate)

    if target_missing_count == 0:
        return _impute_and_evaluate(train_data_clean, mask, method, missing_rate, "MAR(Block)", verbose=verbose, fitted_imputer=fitted_imputer)

    attempts = 0
    max_attempts = max(rows * 20, 1000)

    while mask.sum() < target_missing_count and attempts < max_attempts:
        attempts += 1
        start_row = np.random.randint(0, rows)
        end_row = min(start_row + block_size, rows)

        block_candidates = valid_mask[start_row:end_row, :] & (~mask[start_row:end_row, :])
        candidate_coords = np.argwhere(block_candidates)
        if len(candidate_coords) == 0:
            continue

        remain = target_missing_count - int(mask.sum())
        if len(candidate_coords) > remain:
            pick = np.random.choice(len(candidate_coords), size=remain, replace=False)
            candidate_coords = candidate_coords[pick]

        candidate_coords[:, 0] += start_row
        mask[candidate_coords[:, 0], candidate_coords[:, 1]] = True

    if mask.sum() < target_missing_count:
        remain = target_missing_count - int(mask.sum())
        fallback_candidates = np.argwhere(valid_mask & (~mask))
        if len(fallback_candidates) > 0:
            pick = np.random.choice(len(fallback_candidates), size=min(remain, len(fallback_candidates)), replace=False)
            coords = fallback_candidates[pick]
            mask[coords[:, 0], coords[:, 1]] = True

    return _impute_and_evaluate(train_data_clean, mask, method, missing_rate, "MAR(Block)", verbose=verbose, fitted_imputer=fitted_imputer)


# ==========================================
# 机制 4：MNAR - 极值截断缺失 (Value-dependent Masking)
# ==========================================
def inject_mnar_value_missing_and_impute(train_data_clean, missing_rate=0.30, method='mean', seed=42, verbose=True, fitted_imputer=None):
    np.random.seed(seed)
    rows, cols = train_data_clean.shape
    valid_mask = ~np.isnan(train_data_clean)
    mask = np.zeros((rows, cols), dtype=bool)
    lower_q = missing_rate / 2.0
    upper_q = 1.0 - (missing_rate / 2.0)

    for col in range(cols):
        valid_rows = np.where(valid_mask[:, col])[0]
        if len(valid_rows) == 0:
            continue
        col_data = train_data_clean[valid_rows, col]
        lower_thresh = np.nanquantile(col_data, lower_q)
        upper_thresh = np.nanquantile(col_data, upper_q)
        col_mask = (col_data <= lower_thresh) | (col_data >= upper_thresh)
        selected_rows = valid_rows[col_mask]
        mask[selected_rows, col] = True

    mask &= valid_mask
    target_missing_count = int(valid_mask.sum() * missing_rate)
    current_missing = int(mask.sum())

    if current_missing > target_missing_count:
        excess = current_missing - target_missing_count
        missing_coords = np.argwhere(mask)
        restore_indices = np.random.choice(len(missing_coords), size=excess, replace=False)
        coords_to_restore = missing_coords[restore_indices]
        mask[coords_to_restore[:, 0], coords_to_restore[:, 1]] = False
    elif current_missing < target_missing_count:
        shortage = target_missing_count - current_missing
        add_candidates = np.argwhere(valid_mask & (~mask))
        if len(add_candidates) > 0:
            add_indices = np.random.choice(len(add_candidates), size=min(shortage, len(add_candidates)), replace=False)
            coords_to_add = add_candidates[add_indices]
            mask[coords_to_add[:, 0], coords_to_add[:, 1]] = True

    return _impute_and_evaluate(train_data_clean, mask, method, missing_rate, "MNAR(Value)", verbose=verbose, fitted_imputer=fitted_imputer)

# ==========================================
# 完全对齐深度学习 7:3 评估标准
# ==========================================
