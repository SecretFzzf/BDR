# -*- coding: utf-8 -*-

# Import system libraries for handling paths, time, exceptions, checkpoint files, and garbage collection.
import os
import sys
import json
import time
import gc
import traceback
import warnings

# Import numerical and tabular processing libraries.
import numpy as np
import pandas as pd

# Import standardization and error evaluation functions.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import PyTorch ecosystem for deep learning training.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Unified hardware device detection: prefer CUDA, then MPS, finally CPU.
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Suppress warnings to reduce terminal log noise.
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Get the absolute directory of the current script to keep file paths stable.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the module search path so sibling tool files can be imported.
sys.path.insert(0, BASE_DIR)

# Reuse missing data injection, imputation, and Ridge rolling forecast evaluation logic.
from tool import (
    inject_mcar_missing_and_impute,
    inject_mar_block_missing_and_impute,
    inject_mnar_value_missing_and_impute,
    run_ridge_rolling_forecast,
)

def masked_mse_loss(pred, target):
    """NaN-aware MSE loss: NaN positions in target are excluded from loss computation."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return (pred * 0.0).sum()
    diff = pred[mask] - target[mask]
    return (diff ** 2).mean()


class MaskedErrorAccumulator:
    """Accumulate valid errors across batches and divide at the end, avoiding weighting bias caused by varying NaN proportions across batches."""
    def __init__(self):
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.count = 0

    def update(self, pred, target):
        mask = ~torch.isnan(target)
        n_valid = int(mask.sum().item())
        if n_valid == 0:
            return
        diff = (pred[mask] - target[mask]).detach()
        self.sum_abs += float(diff.abs().sum().item())
        self.sum_sq += float((diff ** 2).sum().item())
        self.count += n_valid

    def compute(self):
        if self.count == 0:
            return float("nan"), float("nan"), 0
        mae = self.sum_abs / self.count
        mse = self.sum_sq / self.count
        return mae, mse, self.count


# =============================
# 1. Global Configuration
# =============================

# SWaT data file path (normal operating condition data).
DATA_FILE = os.path.join(BASE_DIR, "normal.csv")

# SWaT non-feature columns.
EXCLUDE_COLS = ["Timestamp", "Normal/Attack", "Unnamed: 0"]

# Missing data mechanisms (3 types).
MISSING_MODES = ["MCAR", "MAR_Block", "MNAR"]

# Imputation methods (5 types).
IMPUTE_METHODS = ["mean", "spline", "knn", "brits", "saits"]

# Prediction horizon list.
PRED_LEN_LIST = [24, 96]

# Missing rates (4 levels).
MISSING_RATES = [0.10, 0.30, 0.50, 0.70]

# Fixed train/validation/test split ratios (continuous sequence 7:1:2).
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# Sliding window parameters.
SEQ_LEN = 96
STRIDE = 24

# ---------------------------------------------------------
# SWaT Specific Hyperparameters (Optimized for RTX 3090 24GB,
# approximately 8000 rows x 51 dimensions, medium-small sample size)
# ---------------------------------------------------------
DEFAULT_DL_CONFIG = {
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "seq_len": 96,
    "max_epochs": 40,
    "patience": 8,
    "min_delta": 1e-6,
    "num_workers": 12,
}

MODEL_SPECIFIC_CONFIG = {
    "PatchTST": {
        "batch_size": 256,
        "learning_rate": 5e-4,
        "num_workers": 12,
    },
    "DLinear": {
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "num_workers": 12,
    },
    "LSTM": {"batch_size": 1024, "num_workers": 12},
    "GRU": {"batch_size": 1024, "num_workers": 12},
    "Ridge": {
        "alphas": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    },
}

# Random seed.
SEED = 2026


# =================================
# 2. Dataset Definition (Continuous Sliding Window)
# =================================

class ContinuousTimeSeriesDataset(Dataset):
    """Continuous time series sliding window dataset."""

    def __init__(self, x_array, y_array, seq_len, pred_len, stride=1):
        self.x_array = x_array
        self.y_array = y_array
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.indices = np.arange(0, len(x_array) - seq_len - pred_len + 1, stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x_window = self.x_array[start:start + self.seq_len]
        y_window = self.y_array[start + self.seq_len:start + self.seq_len + self.pred_len]
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_window, dtype=torch.float32)


class SWaTTemporalSplitDataset(Dataset):
    """SWaT strict temporal split dataset (supports 7:1:2 segmented window extraction)."""

    def __init__(self, x_array, y_array, split_start, split_end, seq_len, pred_len, stride=1):
        self.x_array = x_array
        self.y_array = y_array
        self.split_start = int(split_start)
        self.split_end = int(split_end)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)

        if self.split_start < 0 or self.split_end > len(x_array) or self.split_start >= self.split_end:
            raise ValueError("Invalid temporal split range, please check split_start/split_end.")

        min_start = max(0, self.split_start - self.seq_len)
        max_start = self.split_end - self.seq_len - self.pred_len

        indices = []
        if max_start >= min_start:
            for start in range(min_start, max_start + 1, self.stride):
                y_start = start + self.seq_len
                y_end = y_start + self.pred_len
                if y_start >= self.split_start and y_end <= self.split_end:
                    indices.append(start)

        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = int(self.indices[idx])
        x_window = self.x_array[start:start + self.seq_len]
        y_window = self.y_array[start + self.seq_len:start + self.seq_len + self.pred_len]
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_window, dtype=torch.float32)


# =========================
# 3. Model Definitions (Unified Maintenance)
# =========================

class LSTMForecaster(nn.Module):
    """LSTM multivariate forecaster."""

    def __init__(self, num_features, hidden_size, num_layers, pred_len):
        super().__init__()
        self.num_features = num_features
        self.pred_len = pred_len
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, pred_len * num_features)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred_flat = self.proj(last_hidden)
        pred = pred_flat.view(-1, self.pred_len, self.num_features)
        return pred


class GRUForecaster(nn.Module):
    """GRU multivariate forecaster."""

    def __init__(self, num_features, hidden_size, num_layers, pred_len):
        super().__init__()
        self.num_features = num_features
        self.pred_len = pred_len
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, pred_len * num_features)

    def forward(self, x):
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        pred_flat = self.proj(last_hidden)
        pred = pred_flat.view(-1, self.pred_len, self.num_features)
        return pred


class DLinearForecaster(nn.Module):
    """DLinear forecaster (per-channel linear mapping)."""

    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.linears = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(enc_in)])

    def forward(self, x):
        outputs = []
        for i in range(self.enc_in):
            x_i = x[:, :, i]
            y_i = self.linears[i](x_i)
            outputs.append(y_i)
        pred = torch.stack(outputs, dim=2)
        return pred


class PatchTSTForecaster(nn.Module):
    """Simplified PatchTST forecaster."""

    def __init__(self, seq_len, pred_len, enc_in, patch_len=16, stride=8, e_layers=2, d_model=128):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride

        self.n_patches = (seq_len - patch_len) // stride + 1
        self.patch_embedding = nn.ModuleList([nn.Linear(patch_len, d_model) for _ in range(enc_in)])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        self.pred_heads = nn.ModuleList([
            nn.Linear(d_model * self.n_patches, pred_len) for _ in range(enc_in)
        ])

    def forward(self, x):
        batch_size, _, n_features = x.shape
        outputs = []
        for i in range(n_features):
            x_i = x[:, :, i]
            patches = []
            for j in range(self.n_patches):
                start = j * self.stride
                end = start + self.patch_len
                patch = x_i[:, start:end]
                patches.append(patch)

            patches = torch.stack(patches, dim=1)
            emb = self.patch_embedding[i](patches)
            enc = self.encoder(emb)
            enc_flat = enc.reshape(batch_size, -1)
            y_i = self.pred_heads[i](enc_flat)
            outputs.append(y_i)

        pred = torch.stack(outputs, dim=2)
        return pred


# =====================
# 4. Data Loading and Preprocessing
# =====================

def load_and_preprocess_data():
    """Load SWaT data and perform strict leakage-free standardization."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    # Explicitly identify and remove timestamp/datetime columns (column name matching + datetime type).
    timestamp_like_cols = []
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if (
            "timestamp" in col_lower
            or "datetime" in col_lower
            or col_lower == "time"
            or col_lower == "date"
        ):
            timestamp_like_cols.append(col)
            continue

        # For object columns, try parsing as dates; if the vast majority parse successfully, treat it as a time column.
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.98:
                timestamp_like_cols.append(col)

    # Build feature columns: first remove explicitly excluded columns, then remove time columns.
    drop_cols = set(EXCLUDE_COLS) | set(timestamp_like_cols)
    raw_value_cols = [c for c in df.columns if c not in drop_cols]
    # Force remaining columns to numeric, setting unconvertible values to NaN for unified handling.
    numeric_df = df[raw_value_cols].apply(pd.to_numeric, errors="coerce")

    # Robustly handle NaN/Inf to ensure the downstream scaler is usable.
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").ffill().bfill()

    # Convert to float32 matrix for high-throughput training on small data.
    raw_data = numeric_df.values.astype(np.float32)

    variances = np.nanvar(raw_data, axis=0)
    non_constant_mask = variances > 1e-5

    data = raw_data[:, non_constant_mask]
    value_cols = [raw_value_cols[i] for i in range(len(raw_value_cols)) if non_constant_mask[i]]

    total_len = len(data)
    train_end = int(total_len * TRAIN_RATIO)
    val_end = train_end + int(total_len * VAL_RATIO)

    train_end = max(1, min(train_end, total_len - 2))
    val_end = max(train_end + 1, min(val_end, total_len - 1))

    scaler = StandardScaler()
    scaler.fit(data[:train_end])
    full_clean_data = scaler.transform(data).astype(np.float32)

    train_data_full = full_clean_data[:train_end]
    val_data_full = full_clean_data[train_end:val_end]
    test_data_full = full_clean_data[val_end:]

    print("=" * 90)
    print("SWaT data loading complete")
    print(f"Total time steps: {total_len}")
    print(f"Training time steps: {len(train_data_full)}")
    print(f"Validation time steps: {len(val_data_full)}")
    print(f"Test time steps: {len(test_data_full)}")
    print(f"Time columns removed: {len(timestamp_like_cols)}")
    print(f"Raw variable count: {len(raw_value_cols)}")
    print(f"Valid variable count: {len(value_cols)}")
    print("=" * 90)

    split_info = {
        "train_end": int(train_end),
        "val_end": int(val_end),
        "total_len": int(total_len),
    }

    return train_data_full, val_data_full, test_data_full, full_clean_data, value_cols, split_info


def _run_eval_loss(model, data_loader, criterion=None):
    """Compute the average MSE loss on the given dataset using MaskedErrorAccumulator."""
    model.eval()
    acc = MaskedErrorAccumulator()
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            acc.update(pred, batch_y)
    _, mse, _ = acc.compute()
    return mse if not np.isnan(mse) else float("inf")


# ====================
# 5. Missing Data Injection Routing Function
# ====================

def inject_and_impute(train_clean, missing_mode, missing_rate, impute_method, verbose=True, fitted_imputer=None):
    """Route to the corresponding injection + imputation function based on the missing mechanism, and return IE metrics."""
    if missing_mode == "MCAR":
        return inject_mcar_missing_and_impute(train_clean, missing_rate, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)
    if missing_mode == "MAR_Block":
        return inject_mar_block_missing_and_impute(train_clean, missing_rate, 24, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)
    if missing_mode == "MNAR":
        return inject_mnar_value_missing_and_impute(train_clean, missing_rate, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)
    raise ValueError(f"Unknown missing mode: {missing_mode}")


# =====================
# 6. State Persistence Utilities
# =====================

def _load_state(checkpoint_file, results_file):
    """Load checkpoint and historical results."""
    results = []
    completed_keys = set()

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            completed_keys = set(json.load(f))

    if os.path.exists(results_file):
        results = pd.read_csv(results_file).to_dict("records")

    return results, completed_keys


def _save_state(checkpoint_file, results_file, results, completed_keys):
    """Save checkpoint and results."""
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, ensure_ascii=False)

    if results:
        pd.DataFrame(results).to_csv(results_file, index=False)


# =====================
# 7. Experiment Combinations and Model Construction
# =====================

def build_experiments():
    """Build the experiment grid and sort by imputation method complexity."""
    experiments = []
    for pred_len in PRED_LEN_LIST:
        for missing_mode in MISSING_MODES:
            for missing_rate in MISSING_RATES:
                for impute_method in IMPUTE_METHODS:
                    experiments.append(
                        {
                            "pred_len": pred_len,
                            "missing_mode": missing_mode,
                            "missing_rate": missing_rate,
                            "impute_method": impute_method,
                        }
                    )

    method_order = {"mean": 0, "spline": 1, "knn": 2, "brits": 3, "saits": 4}
    experiments.sort(key=lambda x: (method_order[x["impute_method"]], x["pred_len"]))
    return experiments


def build_model(model_name, num_features, pred_len, seq_len=SEQ_LEN):
    """Build the corresponding network based on the model name."""
    if model_name == "LSTM":
        return LSTMForecaster(num_features=num_features, hidden_size=64, num_layers=2, pred_len=pred_len)
    if model_name == "GRU":
        return GRUForecaster(num_features=num_features, hidden_size=64, num_layers=2, pred_len=pred_len)
    if model_name == "DLinear":
        return DLinearForecaster(seq_len=seq_len, pred_len=pred_len, enc_in=num_features)
    if model_name == "PatchTST":
        return PatchTSTForecaster(seq_len=seq_len, pred_len=pred_len, enc_in=num_features)
    raise ValueError(f"Unknown model name: {model_name}")


# ===========================
# 8. Deep Learning Model Training and Evaluation
# ===========================

def train_and_eval_dl_model(model_name, full_imputed_data, full_clean_data, split_info, pred_len, missing_mode, missing_rate):
    """Train and evaluate a single deep learning model, returning FE metrics and window statistics."""
    model_cfg = {**DEFAULT_DL_CONFIG, **MODEL_SPECIFIC_CONFIG.get(model_name, {})}
    batch_size = int(model_cfg["batch_size"])
    learning_rate = float(model_cfg["learning_rate"])
    max_epochs = int(model_cfg["max_epochs"])
    patience = int(model_cfg["patience"])
    min_delta = float(model_cfg["min_delta"])
    seq_len = int(model_cfg["seq_len"])
    num_workers = int(model_cfg.get("num_workers", 12))

    train_end = int(split_info["train_end"])
    val_end = int(split_info["val_end"])
    total_len = int(split_info["total_len"])

    train_dataset = SWaTTemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=0,
        split_end=train_end,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=1,
    )
    val_dataset = SWaTTemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=train_end,
        split_end=val_end,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=1,
    )
    test_dataset = SWaTTemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=val_end,
        split_end=total_len,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=STRIDE,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Window count is 0, please check seq_len/pred_len or the 7:1:2 temporal split boundaries.")

    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=bool(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=bool(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=bool(num_workers > 0),
    )

    model = build_model(model_name=model_name, num_features=full_imputed_data.shape[1], pred_len=pred_len, seq_len=seq_len)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = masked_mse_loss

    missing_mode_tag = str(missing_mode).replace("/", "_")
    weight_path = os.path.join(
        BASE_DIR,
        f"best_model_{model_name}_{missing_mode_tag}_{int(round(float(missing_rate) * 100))}_{pred_len}.pth",
    )
    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    try:
        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / max(1, len(train_loader))
            val_loss = _run_eval_loss(model=model, data_loader=val_loader, criterion=criterion)

            print(
                f"   [{model_name}] Epoch {epoch + 1}/{max_epochs}, "
                f"TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}"
            )

            if val_loss < (best_val_loss - min_delta):
                best_val_loss = float(val_loss)
                best_epoch = int(epoch + 1)
                bad_epochs = 0
                torch.save(model.state_dict(), weight_path)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(
                        f"   [{model_name}] EarlyStopping triggered: "
                        f"validation loss did not improve for {patience} consecutive epochs."
                    )
                    break

        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))

        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_pred = model(batch_x).cpu().numpy()
                preds.append(batch_pred)
                trues.append(batch_y.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        c = preds.shape[-1]
        trues_reshaped = trues.reshape(-1, c)
        preds_reshaped = preds.reshape(-1, c)
        per_channel_abs = np.nanmean(np.abs(trues_reshaped - preds_reshaped), axis=0)
        per_channel_sq = np.nanmean(np.square(trues_reshaped - preds_reshaped), axis=0)
        fe_mae = float(np.nanmean(per_channel_abs))
        fe_mse = float(np.nanmean(per_channel_sq))

        print(f"   [{model_name} Test] FE_MAE={fe_mae:.4f}, FE_MSE={fe_mse:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return (
            fe_mae,
            fe_mse,
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
            float(best_val_loss),
            int(best_epoch),
            weight_path,
        )
    finally:
        if os.path.exists(weight_path):
            try:
                os.remove(weight_path)
            except OSError:
                pass


# =========================
# 9. Ridge Regression Model Evaluation (Reuses utility, replaces original VAR)
# =========================

def eval_ridge_model(
    full_imputed_data,
    full_clean_data,
    value_cols,
    split_info,
    pred_len,
    impute_method,
    missing_rate,
):
    """
    Execute Ridge regression rolling forecast evaluation (reuses run_ridge_rolling_forecast from tool.py).

    Replaces the original VAR with L2-regularized linear regression: the history window length uniformly uses
    the global SEQ_LEN (consistent lookback window with deep learning models such as LSTM/DLinear/PatchTST for
    fair horizontal comparison), and no longer dynamically shrinks the order based on training segment length /
    feature count as VAR did — Ridge's L2 regularization inherently handles high-dimensional, strongly correlated
    design matrices, naturally avoiding the issue of VAR's covariance matrix becoming nearly singular and
    predictions diverging under high missing rates + block missing (the 10 records excluded in data hygiene rule 1).

    Returns:
        fe_mae: forecast error MAE
        fe_mse: forecast error MSE
        train_windows: number of training windows (sliding window samples used for Ridge fitting)
        test_windows: estimated number of test windows
    """
    # Read strict temporal split points.
    train_end = int(split_info["train_end"])
    val_end = int(split_info["val_end"])
    total_len = int(split_info["total_len"])

    # History window length uniformly uses the global SEQ_LEN, comparable to deep learning models, no longer computed dynamically.
    lag_order = SEQ_LEN

    # Read this model's alpha candidate grid (automatically selected by RidgeCV's internal cross-validation).
    ridge_cfg = MODEL_SPECIFIC_CONFIG.get({})
    alphas = ridge_cfg.get("alphas", None)

    # Call the utility function to execute rolling forecast.
    _, fe_mae, fe_mse = run_ridge_rolling_forecast(
        full_imputed_data=full_imputed_data,
        full_clean_data=full_clean_data,
        value_cols=value_cols,
        train_end=train_end,
        val_end=val_end,
        missing_rate=missing_rate,
        method=impute_method,
        max_lags=lag_order,
        forecast_steps=pred_len,
        stride=STRIDE,
        alphas=alphas,
    )

    # Training window count: the number of sliding window samples actually used for Ridge fitting.
    train_windows = max(1, train_end - lag_order - pred_len + 1)
    # Roughly estimate the number of test windows (for logging purposes).
    test_windows = max(1, (total_len - val_end - pred_len) // STRIDE + 1)

    # Return Ridge metrics.
    return float(fe_mae), float(fe_mse), int(train_windows), int(test_windows)


# =======================
# 10. Single-Model Unified Execution Entry
# =======================

def run_single_model(model_name):
    """Execute training and evaluation of a single model across the full experiment grid."""
    results_file = os.path.join(BASE_DIR, f"results_refactor_swat_{model_name.lower()}.csv")
    checkpoint_file = os.path.join(BASE_DIR, f"checkpoint_refactor_swat_{model_name.lower()}.json")

    train_data_full, val_data_full, test_data_full, full_clean_data, value_cols, split_info = load_and_preprocess_data()

    results, completed_keys = _load_state(checkpoint_file=checkpoint_file, results_file=results_file)
    experiments = build_experiments()

    print(f"Model: {model_name}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Results file: {results_file}")
    print(f"Checkpoint file: {checkpoint_file}")

    for exp_idx, exp in enumerate(experiments, 1):
        pred_len = exp["pred_len"]
        missing_mode = exp["missing_mode"]
        missing_rate = exp["missing_rate"]
        impute_method = exp["impute_method"]

        exp_key = f"{model_name}_{missing_mode}_{impute_method}_{pred_len}_{int(missing_rate * 100)}"

        if exp_key in completed_keys:
            print(f"[{exp_idx}/{len(experiments)}] Skip completed: {exp_key}")
            continue

        print("\n" + "=" * 90)
        print(
            f"[{exp_idx}/{len(experiments)}] Model={model_name}, Missing mechanism={missing_mode}, "
            f"Imputation={impute_method}, pred_len={pred_len}, Missing rate={int(missing_rate * 100)}%"
        )
        print("=" * 90)

        start_time = time.time()

        try:
            # ----------------------------------------------------------
            # Cache-First imputation flow: reuse cache first, compute and write to cache on miss.
            # Note: the cache key does not include pred_len, because imputation depends only on missing mechanism/method/rate.
            # ----------------------------------------------------------
            cache_dir = os.path.join(BASE_DIR, "npz_file")
            os.makedirs(cache_dir, exist_ok=True)
            cache_filename = f"impute_{missing_mode}_{impute_method}_rate{int(missing_rate*100)}.npz"
            cache_path = os.path.join(cache_dir, cache_filename)

            if os.path.exists(cache_path):
                data = np.load(cache_path)
                full_imputed_data = data["full_data"]
                ie_mae = float(data["ie_mae"])
                ie_mse = float(data["ie_mse"])
                print(f"⚡ Fast loading imputation cache: {cache_path}")
            else:
                # Inject missing data and impute the training segment (silent).
                train_imputed, _, _, fitted_imputer = inject_and_impute(
                    train_clean=train_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    verbose=False,
                    fitted_imputer=None,
                )

                # SAITS/BRITS models are bound to n_steps; fitted cannot be reused across segments of different lengths.
                reuse = None if impute_method in ("saits", "brits") else fitted_imputer

                # Inject missing data and impute the validation segment (silent).
                val_imputed, _, _, fitted_imputer = inject_and_impute(
                    train_clean=val_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    verbose=False,
                    fitted_imputer=reuse,
                )

                reuse = None if impute_method in ("saits", "brits") else fitted_imputer

                # Inject missing data and impute the test segment (output IE).
                test_imputed, ie_mae, ie_mse, fitted_imputer = inject_and_impute(
                    train_clean=test_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    verbose=True,
                    fitted_imputer=reuse,
                )

                # Concatenate the three imputed segments.
                full_imputed_data = np.vstack((train_imputed, val_imputed, test_imputed))

                # Cache the full imputed result and IE.
                np.savez(cache_path, full_data=full_imputed_data, ie_mae=ie_mae, ie_mse=ie_mse)
                print(f"💾 Imputation computed and cached: {cache_path}")

            if model_name == "Ridge":
                fe_mae, fe_mse, train_windows, test_windows = eval_ridge_model(
                    full_imputed_data=full_imputed_data,
                    full_clean_data=full_clean_data,
                    value_cols=value_cols,
                    split_info=split_info,
                    pred_len=pred_len,
                    impute_method=impute_method,
                    missing_rate=missing_rate,
                )
                val_windows = 0
                best_val_loss = None
                best_epoch = None
                best_model_path = None
            else:
                (
                    fe_mae,
                    fe_mse,
                    train_windows,
                    val_windows,
                    test_windows,
                    best_val_loss,
                    best_epoch,
                    best_model_path,
                ) = train_and_eval_dl_model(
                    model_name=model_name,
                    full_imputed_data=full_imputed_data,
                    full_clean_data=full_clean_data,
                    split_info=split_info,
                    pred_len=pred_len,
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()

            elapsed_sec = round(time.time() - start_time, 2)

            result_row = {
                "model": model_name,
                "missing_mode": missing_mode,
                "impute_method": impute_method,
                "missing_rate": missing_rate,
                "pred_len": pred_len,
                "seq_len": SEQ_LEN,
                "ie_mae": round(float(ie_mae), 6),
                "ie_mse": round(float(ie_mse), 6),
                "fe_mae": round(float(fe_mae), 6),
                "fe_mse": round(float(fe_mse), 6),
                "train_windows": int(train_windows),
                "val_windows": int(val_windows),
                "test_windows": int(test_windows),
                "best_val_loss": None if best_val_loss is None else round(float(best_val_loss), 6),
                "best_epoch": best_epoch,
                "best_model_path": best_model_path,
                "status": "success",
                "elapsed_sec": elapsed_sec,
            }

            results.append(result_row)
            completed_keys.add(exp_key)

            print(
                f"✅ Completed: {exp_key} | IE_MSE(Test)={ie_mse:.4f} | FE_MSE(Test)={fe_mse:.4f} | "
                f"BestVal={best_val_loss if best_val_loss is not None else 'N/A'} | Elapsed={elapsed_sec:.2f}s"
            )

        except Exception as exc:
            elapsed_sec = round(time.time() - start_time, 2)
            error_msg = str(exc)[:200]

            print(f"❌ Failed: {exp_key} | error={error_msg}")
            traceback.print_exc()

            result_row = {
                "model": model_name,
                "missing_mode": missing_mode,
                "impute_method": impute_method,
                "missing_rate": missing_rate,
                "pred_len": pred_len,
                "seq_len": SEQ_LEN,
                "ie_mae": None,
                "ie_mse": None,
                "fe_mae": None,
                "fe_mse": None,
                "train_windows": None,
                "val_windows": None,
                "test_windows": None,
                "best_val_loss": None,
                "best_epoch": None,
                "best_model_path": None,
                "status": f"error: {error_msg}",
                "elapsed_sec": elapsed_sec,
            }

            results.append(result_row)
            completed_keys.add(exp_key)

        _save_state(
            checkpoint_file=checkpoint_file,
            results_file=results_file,
            results=results,
            completed_keys=completed_keys,
        )

        gc.collect()

    print("\n" + "=" * 90)
    print(f"All experiments completed for model {model_name}")
    print(f"Results written to: {results_file}")
    print("=" * 90)


# =======================
# 11. Main Dispatch Entry (Optional)
# =======================

def run_all_models(model_list=None):
    """Run multiple models sequentially."""
    if model_list is None:
        model_list = ["LSTM", "DLinear", "PatchTST"]

    for model_name in model_list:
        run_single_model(model_name)
