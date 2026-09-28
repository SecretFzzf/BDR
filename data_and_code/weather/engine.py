# -*- coding: utf-8 -*-

# Import standard library modules for path, time, checkpoint, exception, and garbage collection management.
import os
import sys
import json
import time
import gc
import traceback
import warnings

# Import numerical computation and tabular processing libraries.
import numpy as np
import pandas as pd

# Import standardization and error metrics.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import PyTorch ecosystem for deep learning training.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Unified hardware device detection: prefer CUDA, then MPS, finally CPU.
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Enable cuDNN auto-tuning for fixed input shape tasks to boost 3090 throughput.
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Uniformly suppress warnings to reduce log noise.
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Get the current script directory to ensure relative paths are stable.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the module search path for importing tool.py in the same directory.
sys.path.insert(0, BASE_DIR)

# Reuse missing data injection and Ridge regression rolling evaluation logic from the historical tool (replaces the original VAR to avoid numerical divergence under high missing rates + block missing).
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
    """Accumulate valid errors across batches and divide uniformly at the end to avoid weighting bias caused by different NaN ratios across batches."""
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
# Section 1: Global Configuration
# =============================

# Weather main data file path.
DATA_FILE = os.path.join(BASE_DIR, "weather.csv")

# Missing mechanism set (3 types).
MISSING_MODES = ["MCAR", "MAR_Block", "MNAR"]

# Imputation method set (5 types).
IMPUTE_METHODS = ["mean", "spline", "knn", "brits", "saits"]

# Prediction horizon set.
PRED_LEN_LIST = [24, 48, 96, 192]

# Missing rate set (4 levels).
MISSING_RATES = [0.10, 0.30, 0.50, 0.70]

# Fixed train/validation/test split ratios (continuous sequence 7:1:2).
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# ---------------------------------------------------------
# METR-LA Specific Hyperparameters (Optimized for RTX 3090 24GB)
# ---------------------------------------------------------
DEFAULT_DL_CONFIG = {
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "seq_len": 96,
    "max_epochs": 50,
    "patience": 12,
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
        "batch_size": 2048,
        "learning_rate": 1e-3,
        "num_workers": 12,
    },
    "LSTM": {"batch_size": 1024, "num_workers": 12},
    "GRU": {"batch_size": 1024, "num_workers": 12},
    "Ridge": {
        "alphas": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    },
}

# Default input history window length (read from layered config).
SEQ_LEN = int(DEFAULT_DL_CONFIG["seq_len"])

# Rolling evaluation stride: Ridge slides by day (288*5min=1 day), DL uses denser stride to obtain sufficient test windows.
STRIDE = 24

# Random seed configuration.
SEED = 42


# =================================
# Section 2: Dataset Definition (Continuous Sliding Window)
# =================================

class ContinuousTimeSeriesDataset(Dataset):
    """Continuous time series sliding window dataset (non-segmented mode)."""

    def __init__(self, x_array, y_array, seq_len, pred_len, stride=1):
        # Save input feature matrix.
        self.x_array = x_array
        # Save supervised target matrix.
        self.y_array = y_array
        # Save history window length.
        self.seq_len = seq_len
        # Save prediction window length.
        self.pred_len = pred_len
        # Save sliding window stride.
        self.stride = stride
        # Construct all valid start indices to ensure complete windows.
        self.indices = np.arange(0, len(x_array) - seq_len - pred_len + 1, stride)

    def __len__(self):
        # Return total number of samples.
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the start index of the current sample.
        start = self.indices[idx]
        # Slice the input window.
        x_window = self.x_array[start:start + self.seq_len]
        # Slice the label window.
        y_window = self.y_array[start + self.seq_len:start + self.seq_len + self.pred_len]
        # Convert to float32 Tensor.
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_window, dtype=torch.float32)


class METRLATemporalSplitDataset(Dataset):
    """METR-LA strict temporal split dataset (supports 7:1:2 segmented window extraction)."""

    def __init__(self, x_array, y_array, split_start, split_end, seq_len, pred_len, stride=1):
        # Save input feature array.
        self.x_array = x_array
        # Save supervised target array.
        self.y_array = y_array
        # Save split start position.
        self.split_start = int(split_start)
        # Save split end position.
        self.split_end = int(split_end)
        # Save history window length.
        self.seq_len = int(seq_len)
        # Save prediction window length.
        self.pred_len = int(pred_len)
        # Save sliding window stride.
        self.stride = int(stride)

        # Validate split boundary legality.
        if self.split_start < 0 or self.split_end > len(x_array) or self.split_start >= self.split_end:
            raise ValueError("Invalid temporal split interval, please check split_start/split_end.")

        # Compute the minimum allowed window start.
        min_start = max(0, self.split_start - self.seq_len)
        # Compute the maximum allowed window start.
        max_start = self.split_end - self.seq_len - self.pred_len

        # Collect all valid window starts.
        indices = []
        if max_start >= min_start:
            for start in range(min_start, max_start + 1, self.stride):
                # Compute label window start.
                y_start = start + self.seq_len
                # Compute label window end.
                y_end = y_start + self.pred_len
                # Only keep windows whose labels fall entirely within the current segment.
                if y_start >= self.split_start and y_end <= self.split_end:
                    indices.append(start)

        # Save window start index array.
        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self):
        # Return number of available windows.
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the start position corresponding to the current sample.
        start = int(self.indices[idx])
        # Slice the input history window.
        x_window = self.x_array[start:start + self.seq_len]
        # Slice the future label window.
        y_window = self.y_array[start + self.seq_len:start + self.seq_len + self.pred_len]
        # Return float32 tensors.
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_window, dtype=torch.float32)


# =========================
# Section 3: Model Definitions (Unified Maintenance)
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
    """DLinear forecaster (channel-independent linear mapping)."""

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
# Section 4: Data Loading and Preprocessing
# =====================

def load_and_preprocess_data():
    """Load Weather data and perform strict leakage-proof standardization."""
    # Read weather.csv.
    candidate_files = [
        DATA_FILE,
        os.path.join(BASE_DIR, "weather.csv"),
    ]
    existing_files = [p for p in candidate_files if os.path.exists(p)]
    if not existing_files:
        raise FileNotFoundError(f"No available data file found, please check: {candidate_files}")
    data_file = existing_files[0]

    # Read CSV data.
    df = pd.read_csv(data_file)
    # Keep only numeric columns (automatically exclude timestamp/string index columns).
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    # Raise error if no numeric columns.
    if numeric_df.shape[1] == 0:
        raise ValueError(f"No available numeric columns in {os.path.basename(data_file)}.")

    # METR-LA: zero = sensor offline (0 mph is not a real driving speed), treated as structural missing; replace with NaN then interpolate to fill.
    numeric_df = numeric_df.replace(0, np.nan)
    numeric_df = numeric_df.interpolate(limit_direction='both').ffill().bfill()

    # Get original column names and numeric matrix.
    value_cols = list(numeric_df.columns)
    data = numeric_df.values.astype(np.float64)

    # Compute total length.
    total_len = len(data)
    # Compute 7:1:2 temporal split points.
    train_end = int(total_len * TRAIN_RATIO)
    val_end = train_end + int(total_len * VAL_RATIO)

    # Defensive correction: avoid split points going out of bounds.
    train_end = max(1, min(train_end, total_len - 2))
    val_end = max(train_end + 1, min(val_end, total_len - 1))

    # Fit scaler on training segment to prevent data leakage.
    scaler = StandardScaler()
    scaler.fit(data[:train_end])

    # Uniformly transform full data (using only training segment statistics).
    full_clean_data = scaler.transform(data)

    # Split into three segments by time.
    train_data_full = full_clean_data[:train_end]
    val_data_full = full_clean_data[train_end:val_end]
    test_data_full = full_clean_data[val_end:]

    # Print data summary.
    print("=" * 80)
    print("METR-LA data loading completed")
    print(f"Data file: {os.path.basename(data_file)}")
    print(f"Total length: {total_len}")
    print(f"Training length: {len(train_data_full)}")
    print(f"Validation length: {len(val_data_full)}")
    print(f"Test length: {len(test_data_full)}")
    print(f"Number of valid variables: {len(value_cols)}")
    print("=" * 80)

    # Build temporal split info dictionary.
    split_info = {
        "train_end": int(train_end),
        "val_end": int(val_end),
        "total_len": int(total_len),
    }

    # Return preprocessing results.
    return train_data_full, val_data_full, test_data_full, full_clean_data, value_cols, split_info


def _run_eval_loss(model, data_loader, criterion=None):
    """Compute average MSE loss on the given dataset using MaskedErrorAccumulator."""
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
# Section 5: Missing Data Injection Routing Function
# ====================

def inject_and_impute(train_clean, missing_mode, missing_rate, impute_method, verbose=True, fitted_imputer=None):
    """Route to the corresponding injection + imputation function by missing mechanism and return IE metrics."""
    if missing_mode == "MCAR":
        return inject_mcar_missing_and_impute(train_clean, missing_rate, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)
    if missing_mode == "MAR_Block":
        return inject_mar_block_missing_and_impute(train_clean, missing_rate, 24, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)
    if missing_mode == "MNAR":
        return inject_mnar_value_missing_and_impute(train_clean, missing_rate, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)
    raise ValueError(f"Unknown missing mode: {missing_mode}")


# =====================
# Section 6: Experiment State Persistence Utilities
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
# Section 7: Experiment Combination and Model Building
# =====================

def build_experiments():
    """Build experiment grid and sort by imputation method complexity."""
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
    """Build the corresponding network by model name."""
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
# Section 8: Deep Learning Model Training and Evaluation
# ===========================

def train_and_eval_dl_model(model_name, full_imputed_data, full_clean_data, split_info, pred_len, missing_mode, missing_rate):
    """Train and evaluate a single deep learning model, returning FE metrics and window statistics."""
    # Merge default config with model-specific config.
    model_cfg = {**DEFAULT_DL_CONFIG, **MODEL_SPECIFIC_CONFIG.get(model_name, {})}
    # Read current model batch size.
    batch_size = int(model_cfg["batch_size"])
    # Read current model learning rate.
    learning_rate = float(model_cfg["learning_rate"])
    # Read current model max epochs.
    max_epochs = int(model_cfg["max_epochs"])
    # Read current model early stopping patience.
    patience = int(model_cfg["patience"])
    # Read current model minimum improvement threshold.
    min_delta = float(model_cfg["min_delta"])
    # Read current model sequence length.
    seq_len = int(model_cfg["seq_len"])
    # Read DataLoader worker process count.
    num_workers = int(model_cfg.get("num_workers", 12))

    # Read strict temporal split points.
    train_end = int(split_info["train_end"])
    val_end = int(split_info["val_end"])
    total_len = int(split_info["total_len"])

    # Extract train/val/test windows by time boundaries on the full sequence.
    train_dataset = METRLATemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=0,
        split_end=train_end,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=1,
    )
    val_dataset = METRLATemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=train_end,
        split_end=val_end,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=1,
    )
    test_dataset = METRLATemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=val_end,
        split_end=total_len,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=STRIDE,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Window count is 0, please check seq_len/pred_len or 7:1:2 temporal split boundaries.")

    # Enable pin_memory under CUDA to accelerate CPU-to-GPU copy.
    use_pin_memory = device.type == "cuda"
    # Enable persistent workers for multi-threaded loading to reduce per-epoch rebuild overhead.
    persistent_workers = bool(num_workers > 0)

    # Build train/validation/test DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    # Create and migrate model.
    model = build_model(model_name=model_name, num_features=full_imputed_data.shape[1], pred_len=pred_len, seq_len=seq_len)
    model = model.to(device)

    # Create optimizer and loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = masked_mse_loss

    # Generate a unique weight filename for the current experiment to avoid parallel experiments overwriting each other.
    missing_mode_tag = str(missing_mode).replace("/", "_")
    weight_path = os.path.join(
        BASE_DIR,
        f"best_model_{model_name}_{missing_mode_tag}_{int(round(float(missing_rate) * 100))}_{pred_len}.pth",
    )
    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    try:
        # Train for multiple epochs and perform validation.
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
                        f"validation set did not improve for {patience} consecutive epochs."
                    )
                    break

        # Reload best weights for testing.
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))

        # Perform test evaluation.
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
# Section 9: Ridge Regression Model Evaluation (Reuses Tool, Replaces Original VAR)
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
    Execute Ridge regression rolling forecast evaluation (reuses run_ridge_rolling_forecast in tool.py).

    Replaces the original VAR with L2-regularized linear regression: the history window length
    uniformly uses the global SEQ_LEN (consistent look-back window with LSTM/DLinear/PatchTST
    and other deep learning models for easy horizontal comparison), no longer dynamically
    shrinking the order by training segment length/feature count as VAR did — Ridge's L2
    regularization itself can handle high-dimensional, strongly correlated design matrices,
    naturally avoiding the problem that VAR's covariance matrix becomes nearly singular
    and predictions diverge under high missing rates + block missing (i.e., the 10 records
    excluded by data hygiene rule 1).

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

    # History window length uniformly uses the global SEQ_LEN, comparable with deep learning models; no longer dynamically computed separately.
    lag_order = SEQ_LEN

    # Call the tool function to execute rolling forecast (alphas is fixed inside tool.py, passing None is fine).
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
        alphas=None,
    )

    # Number of training windows: the number of sliding window samples actually used to fit Ridge.
    train_windows = max(1, train_end - lag_order - pred_len + 1)
    # Roughly estimate the number of test windows (for logging purposes).
    test_windows = max(1, (total_len - val_end - pred_len) // STRIDE + 1)

    # Return Ridge metrics.
    return float(fe_mae), float(fe_mse), int(train_windows), int(test_windows)


# =======================
# Section 10: Single Model Unified Execution Entry
# =======================

def run_single_model(model_name):
    """Execute training and evaluation of a single model on the full experiment grid."""
    results_file = os.path.join(BASE_DIR, f"results_refactor_{model_name.lower()}.csv")
    checkpoint_file = os.path.join(BASE_DIR, f"checkpoint_refactor_{model_name.lower()}.json")

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
            print(f"[{exp_idx}/{len(experiments)}] Skipped completed: {exp_key}")
            continue

        print("\n" + "=" * 90)
        print(
            f"[{exp_idx}/{len(experiments)}] Model={model_name}, Missing Mechanism={missing_mode}, "
            f"Imputation={impute_method}, pred_len={pred_len}, Missing Rate={int(missing_rate * 100)}%"
        )
        print("=" * 90)

        start_time = time.time()

        try:
            # ----------------------------------------------------------
            # Cache-First imputation flow: reuse cache first, compute and write to cache on miss.
            # Note: the cache key does not include pred_len because imputation depends only on missing mechanism/method/rate.
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
                # Inject missing data and impute only on the training segment to construct training input.
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

                # Inject missing data and impute on the validation segment to avoid validation leakage (input consistent with training distribution).
                val_imputed, _, _, fitted_imputer = inject_and_impute(
                    train_clean=val_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    verbose=False,
                    fitted_imputer=reuse,
                )

                reuse = None if impute_method in ("saits", "brits") else fitted_imputer

                # Inject missing data and impute on the test segment; IE is computed only on the test segment.
                test_imputed, ie_mae, ie_mse, fitted_imputer = inject_and_impute(
                    train_clean=test_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    verbose=True,
                    fitted_imputer=reuse,
                )

                # Reconstruct full input by 7:1:2: training segment imputed, validation segment imputed, test segment imputed.
                full_imputed_data = np.vstack((train_imputed, val_imputed, test_imputed))

                # Persist the full imputed data and IE metrics for direct reuse by subsequent models.
                np.savez(cache_path, full_data=full_imputed_data, ie_mae=ie_mae, ie_mse=ie_mse)
                print(f"💾 Imputation computed and cached: {cache_path}")

            # Ridge branch.
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

            # Try to release CUDA cache after each evaluation round.
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
    print(f"Model {model_name} all experiments completed")
    print(f"Results written to: {results_file}")
    print("=" * 90)


# =======================
# Section 11: Overall Scheduling Entry (Optional)
# =======================

def run_all_models(model_list=None):
    """Run multiple models sequentially; by default executes five baseline models."""
    if model_list is None:
        model_list = ["LSTM", "DLinear", "PatchTST"]

    for model_name in model_list:
        run_single_model(model_name)
