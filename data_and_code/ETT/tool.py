import numpy as np
import pandas as pd
import torch
import inspect
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import KNNImputer
from pypots.imputation import SAITS, BRITS

# Unified hardware device detection: prefer CUDA, then MPS, finally CPU.
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# SAITS OOM fallback sentinel: marks that imputation has been completed via segmented mode, subsequent calls need not retrain.
_SAITS_IMPUTED_VIA_FALLBACK = object()


def _build_imputer_kwargs(model_cls, kwargs):
    """Filter available parameters according to the current PyPOTS version to avoid parameter incompatibility errors."""
    signature = inspect.signature(model_cls.__init__)
    valid_keys = set(signature.parameters.keys())
    valid_keys.discard("self")
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def _resolve_dl_impute_train_cfg(method, chunk_len, n_features):
    """Allocate SAITS/BRITS training epochs and early-stopping parameters according to ETT data scale."""
    method = str(method).lower()

    # ETT has only 7 feature dimensions, overall convergence is faster, lower the baseline epochs.
    base_epochs = 24
    if n_features >= 32:
        base_epochs += 8
    if n_features >= 64:
        base_epochs += 8
    if chunk_len >= 128:
        base_epochs += 8

    # SAITS is more stable, can use slightly higher epoch cap.
    if method == "saits":
        max_epochs = min(72, max(16, base_epochs))
        patience = max(6, max_epochs // 3)
    else:
        # BRITS serial cost is higher, epochs more conservative.
        max_epochs = min(56, max(14, base_epochs - 4))
        patience = max(5, max_epochs // 3)

    # BRITS further limits epochs on long sequences, relying on early stopping to improve efficiency.
    if method == "brits":
        if chunk_len >= 4096:
            max_epochs = min(max_epochs, 10)
            patience = min(patience, 4)
        elif chunk_len >= 2048:
            max_epochs = min(max_epochs, 12)
            patience = min(patience, 4)
        elif chunk_len >= 1024:
            max_epochs = min(max_epochs, 16)
            patience = min(patience, 5)
        elif chunk_len >= 384:
            max_epochs = min(max_epochs, 20)
            patience = min(patience, 6)
        elif chunk_len >= 192:
            max_epochs = min(max_epochs, 24)
            patience = min(patience, 8)

    return {
        "epochs": int(max_epochs),
        "patience": int(patience),
    }


def _pack_windows_for_imputation(data_2d, window_len, stride):
    """Pack long sequences into batched windows to improve BRITS parallelism."""
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
    """Write window imputation results back to the full sequence; overlapping regions are averaged."""
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
# Core helper: unified imputation and error evaluation (hidden from external use)
# ==========================================
def _impute_and_evaluate(
    train_data_clean,
    mask,
    method,
    missing_rate,
    missing_type_str,
    seq_len=96,
    verbose=True,
    fitted_imputer=None,
):
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
        method_name_cn = "Mean"
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
        method_name_cn = "Spline"
    elif method == 'knn':
        if fitted is None:
            fitted = KNNImputer(n_neighbors=5, weights='distance')
            train_data_imputed = fitted.fit_transform(train_data_imputed)
        else:
            train_data_imputed = fitted.transform(train_data_imputed)
        method_name_cn = "KNN"
    elif method == 'saits':
        full_dataset = {"X": train_data_imputed.reshape(1, n_steps, n_features)}
        if fitted is None:
            print("🚀 Training SAITS imputation model (full single-pass mode)...")
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
                "epochs": train_cfg["epochs"],
                "patience": train_cfg["patience"],
                "device": str(device),
            })
            try:
                fitted = SAITS(**saits_kwargs)
                fitted.fit(full_dataset)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("⚠️ SAITS full-mode out of memory, automatically switching to large-chunk segmented mode to continue training...")
                chunk_size = int(min(max(seq_len * 8, 512), n_steps))
                for start in range(0, n_steps, chunk_size):
                    end = min(start + chunk_size, n_steps)
                    chunk = train_data_imputed[start:end]
                    chunk_len = end - start
                    if chunk_len <= 1:
                        continue
                    chunk_dataset = {"X": chunk.reshape(1, chunk_len, n_features)}
                    chunk_cfg = _resolve_dl_impute_train_cfg(method='saits', chunk_len=chunk_len, n_features=n_features)
                    print(
                        f"   [SAITS-Fallback] chunk=({start}:{end}), epochs={chunk_cfg['epochs']}, "
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
                        "epochs": chunk_cfg["epochs"],
                        "patience": chunk_cfg["patience"],
                        "device": str(device),
                    })
                    saits_chunk = SAITS(**chunk_kwargs)
                    saits_chunk.fit(chunk_dataset)
                    imputed_chunk = saits_chunk.impute(chunk_dataset)
                    train_data_imputed[start:end] = imputed_chunk.reshape(chunk_len, n_features)
                fitted = _SAITS_IMPUTED_VIA_FALLBACK
        if fitted is not None and fitted is not _SAITS_IMPUTED_VIA_FALLBACK:
            imputed_result = fitted.impute(full_dataset)
            train_data_imputed = imputed_result.reshape(n_steps, n_features)
        method_name_cn = "SAITS"
    elif method == 'brits':
        # ETT keeps the windowed-batch acceleration threshold at 512, covering roughly 1700 rows of validation segment.
        use_window_batch_mode = n_steps > 512

        if use_window_batch_mode:
            # RNN serial cost on long sequences is high: shorten the window and increase the number of concurrent windows.
            window_len = int(min(max(seq_len * 2, 192), 384))
            stride = int(window_len)
            window_data, starts = _pack_windows_for_imputation(train_data_imputed, window_len=window_len, stride=stride)
            batch_size = max(1, int(len(starts)))
            dataset_for_pypots = {"X": window_data}
            if fitted is None:
                train_cfg = _resolve_dl_impute_train_cfg(method='brits', chunk_len=window_len, n_features=n_features)
                print(
                    f"🚀 Training BRITS imputation model (windowed-batch acceleration mode)... windows={len(starts)}, "
                    f"window_len={window_len}, epochs={train_cfg['epochs']}, patience={train_cfg['patience']}"
                )
                brits_kwargs = _build_imputer_kwargs(BRITS, {
                    "n_steps": window_len,
                    "n_features": n_features,
                    "rnn_hidden_size": 128,
                    "batch_size": batch_size,
                    "epochs": train_cfg["epochs"],
                    "patience": train_cfg["patience"],
                    "device": str(device),
                })
                fitted = BRITS(**brits_kwargs)
                fitted.fit(dataset_for_pypots)
            imputed_windows = fitted.impute(dataset_for_pypots)
            train_data_imputed = _merge_window_imputations(train_data_imputed, imputed_windows, starts)
        else:
            full_dataset = {"X": train_data_imputed.reshape(1, n_steps, n_features)}
            if fitted is None:
                print("🚀 Training BRITS imputation model (full single-pass mode)...")
                train_cfg = _resolve_dl_impute_train_cfg(method='brits', chunk_len=n_steps, n_features=n_features)
                print(
                    f"   [BRITS] full_len={n_steps}, epochs={train_cfg['epochs']}, "
                    f"patience={train_cfg['patience']}"
                )
                brits_kwargs = _build_imputer_kwargs(BRITS, {
                    "n_steps": n_steps,
                    "n_features": n_features,
                    "rnn_hidden_size": 96,
                    "epochs": train_cfg["epochs"],
                    "patience": train_cfg["patience"],
                    "device": str(device),
                })
                fitted = BRITS(**brits_kwargs)
                fitted.fit(full_dataset)
            imputed_result = fitted.impute(full_dataset)
            train_data_imputed = imputed_result.reshape(n_steps, n_features)
        method_name_cn = "BRITS"
    else:
        raise ValueError("Unsupported imputation method")

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

    if verbose:
        print(f"--- Phase 1: {missing_type_str} imputation quality (IE) ---")
        print(f"{int(missing_rate*100)}% {missing_type_str} + {method_name_cn} imputation MAE: {ie_mae:.4f}")
        print(f"{int(missing_rate*100)}% {missing_type_str} + {method_name_cn} imputation MSE: {ie_mse:.4f}\n")

    return train_data_imputed, ie_mae, ie_mse, fitted

# ==========================================
# Mechanism 1: MCAR completely random missing
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

    return _impute_and_evaluate(
        train_data_clean,
        mask,
        method,
        missing_rate,
        "MCAR",
        verbose=verbose,
        fitted_imputer=fitted_imputer,
    )

# ==========================================
# Mechanism 2: MAR - global block missing (Block Masking across all variables)
# ==========================================
def inject_mar_block_missing_and_impute(train_data_clean, missing_rate=0.30, block_size=24, method='mean', seed=42, verbose=True, fitted_imputer=None):
    np.random.seed(seed)
    rows, cols = train_data_clean.shape
    valid_mask = ~np.isnan(train_data_clean)
    mask = np.zeros((rows, cols), dtype=bool)
    target_missing_count = int(valid_mask.sum() * missing_rate)

    if target_missing_count == 0:
        return _impute_and_evaluate(
            train_data_clean,
            mask,
            method,
            missing_rate,
            "MAR(Block)",
            verbose=verbose,
            fitted_imputer=fitted_imputer,
        )

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

    return _impute_and_evaluate(
        train_data_clean,
        mask,
        method,
        missing_rate,
        "MAR(Block)",
        verbose=verbose,
        fitted_imputer=fitted_imputer,
    )

# ==========================================
# Mechanism 3: MAR - variance-weighted variable-wise missing (Variable-wise Masking)
# ==========================================
# NOTE: MAR(Var-wise) injection removed — not referenced by engine files.

# ==========================================
# Mechanism 4: MNAR - extreme-value truncation missing (Value-dependent Masking)
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

    return _impute_and_evaluate(
        train_data_clean,
        mask,
        method,
        missing_rate,
        "MNAR(Value)",
        verbose=verbose,
        fitted_imputer=fitted_imputer,
    )

# ==========================================
# Core overhaul: VAR rolling forecast evaluation (Sliding Window Rolling Forecast)
# Fully aligned with the deep-learning 7:3 evaluation protocol
# ==========================================
def run_ridge_rolling_forecast(full_imputed_data, full_clean_data, value_cols, train_end, val_end, missing_rate=0.30, method='mean', max_lags=96, forecast_steps=96, stride=24, alphas=None):
    """
    Unified evaluation protocol (7:1:2), using Ridge regression to replace the original VAR:
    1. The training segment (first 70%) is used to construct sliding-window samples and fit RidgeCV
       (internally uses leave-one-out cross-validation to select alpha, no extra use of the
       validation segment; the validation-segment split logic is kept consistent with the original
       VAR flow and is only used for boundary delimitation).
    2. The validation segment (middle 10%) is only used for splitting, not for fitting in this function.
    3. The test segment (last 20%) is used for sliding-window FE evaluation; the rolling protocol
       (start/end points, stride) is fully consistent with the original VAR version.
    4. The multivariate history of the past max_lags steps is flattened into one feature vector,
       then we directly regress the target vector of the next forecast_steps steps flattened across
       all variables. L2 regularization guarantees the design matrix is invertible, naturally
       avoiding the problem that VAR's covariance matrix becomes nearly singular and predictions
       diverge under high missing rates + block missing (the 10 records excluded by data-hygiene rule 1).
    """
    method_name_en = str(method).capitalize()
    rate_str = f"{int(missing_rate*100)}%"
    total_len = len(full_imputed_data)
    n_features = full_imputed_data.shape[1]

    # Defensive boundary protection (consistent with the VAR version).
    train_end = int(max(1, min(train_end, total_len - 2)))
    val_end = int(max(train_end + 1, min(val_end, total_len - 1)))

    # Defensive: residual NaN after imputation is filled with training-segment column means (Ridge cannot accept NaN input).
    col_means = np.nanmean(full_imputed_data[:train_end], axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    nan_locs = np.isnan(full_imputed_data)
    if nan_locs.any():
        full_imputed_data = full_imputed_data.copy()
        for j in range(full_imputed_data.shape[1]):
            mask = nan_locs[:, j]
            if mask.any():
                full_imputed_data[mask, j] = col_means[j]

    # lag_order directly reuses max_lags as Ridge's history-window length;
    # unlike VAR, we no longer dynamically shrink the order based on training-segment length / number of features —
    # Ridge's L2 regularization itself handles high feature dimensions with relatively few samples,
    # free of VAR's degrees-of-freedom constraint.
    lag_order = int(max_lags)
    if lag_order < 1:
        raise ValueError("max_lags (here used as Ridge history-window length) must be >= 1.")
    if lag_order + forecast_steps >= train_end:
        raise ValueError("History-window length + forecast-steps exceeds training-segment length, please check max_lags/forecast_steps or split boundaries.")

    # 1. Strict separation: only the first 70% (training segment) is used to construct sliding-window samples to fit Ridge.
    X_train, Y_train = [], []
    for i in range(lag_order, train_end - forecast_steps + 1):
        x_window = full_imputed_data[i - lag_order : i]           # [lag_order, C], input uses imputed data
        y_window = full_clean_data[i : i + forecast_steps]         # [forecast_steps, C], labels use clean data (consistent with VAR)
        X_train.append(x_window.reshape(-1))
        Y_train.append(y_window.reshape(-1))
    X_train = np.asarray(X_train, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)

    # alpha candidate grid: RidgeCV performs efficient leave-one-out cross-validation internally and auto-selects, no need for hand-written validation logic.
    if alphas is None:
        alphas = np.logspace(-3, 3, 13)

    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train, Y_train)

    predictions = []
    actuals = []

    # 2. Sliding-window rolling forecast on the test segment (start/end points and stride fully consistent with the original VAR version).
    start_idx = val_end
    end_idx = total_len - forecast_steps

    print(f"🔄 Starting Ridge test-set sliding-window forecast (window range: {start_idx} -> {end_idx}, stride: {stride})...")

    # Error if the test segment is insufficient to form a window.
    if start_idx > end_idx:
        raise ValueError("Test-segment length is insufficient for Ridge sliding-window forecasting, please check pred_len or split boundaries.")

    for i in range(start_idx, end_idx + 1, stride):
        # Extract the past lag_order steps of history as input and flatten into a single-row feature vector.
        x_input = full_imputed_data[i - lag_order : i].reshape(1, -1)
        # Forecast the next forecast_steps steps, then reshape back to [forecast_steps, C].
        pred = model.predict(x_input).reshape(forecast_steps, n_features)
        # Obtain the true clean data for these forecast_steps steps.
        y_true = full_clean_data[i : i + forecast_steps]

        predictions.append(pred)
        actuals.append(y_true)

    # 3. Flatten into big arrays and compute the overall mean error.
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    c = predictions.shape[-1]
    predictions_reshaped = predictions.reshape(-1, c)
    actuals_reshaped = actuals.reshape(-1, c)
    per_channel_abs = np.nanmean(np.abs(actuals_reshaped - predictions_reshaped), axis=0)
    per_channel_sq = np.nanmean(np.square(actuals_reshaped - predictions_reshaped), axis=0)
    fe_mae = float(np.nanmean(per_channel_abs))
    fe_mse = float(np.nanmean(per_channel_sq))

    print(f"--- Phase 2: forecast quality (FE - Test Set sliding window) ---")
    print(f"{rate_str} missing + {method_name_en} imputation -> Ridge(lag={lag_order}, alpha={model.alpha_:.4g}) rolling forecast MAE: {fe_mae:.4f}")
    print(f"{rate_str} missing + {method_name_en} imputation -> Ridge(lag={lag_order}, alpha={model.alpha_:.4g}) rolling forecast MSE: {fe_mse:.4f}")

    return predictions, fe_mae, fe_mse
