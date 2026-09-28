# Import system standard libraries for paths, checkpoints, timing, exception tracing, and garbage collection.
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

# Import standardization and error evaluation metrics.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import PyTorch ecosystem components for deep learning training.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Unified hardware device detection: prefer CUDA, then MPS, finally CPU.
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Enable cuDNN auto-tuning for fixed input shape tasks to reduce GPU idle time.
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Suppress irrelevant warnings to keep logs concise.
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Get the directory of the current script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the module search path to ensure same-directory modules are importable.
sys.path.insert(0, BASE_DIR)

# Reuse missing data injection and VAR evaluation logic from historical utilities to keep the mathematical flow unchanged.
from tool import (
    inject_mcar_missing_and_impute,
    inject_mar_block_missing_and_impute,
    inject_mnar_value_missing_and_impute,
    run_ridge_rolling_forecast,
)

def masked_mse_loss(pred, target):
    """NaN-aware MSE loss: positions where target is NaN do not participate in the loss calculation."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return (pred * 0.0).sum()
    diff = pred[mask] - target[mask]
    return (diff ** 2).mean()


class MaskedErrorAccumulator:
    """Accumulate valid errors across batches and divide uniformly at the end, avoiding weighting bias from differing NaN ratios across batches."""
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
# 1. Global configuration (preserve original AQI-36 logic)
# =============================

# KDD-Beijing data file path (CSV format).
DATA_FILE = os.path.join(BASE_DIR, "kdd_beijing_raw.csv")

# Missing data mechanism set (3 types).
MISSING_MODES = ["MCAR", "MAR_Block", "MNAR"]

# Imputation method set (5 types, aligned with other datasets).
IMPUTE_METHODS = ["mean", "spline", "knn", "brits", "saits"]

# Prediction horizon list (preserve original AQI-36 logic).
PRED_LEN_LIST = [24, 48, 96, 192]

# Missing rate set (4 levels).
MISSING_RATES = [0.10, 0.30, 0.50, 0.70]

# Train/val/test split ratios (7:1:2, strictly split by chronological order).
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# ---------------------------------------------------------
# AQI-36 Specific Hyperparameters (Optimized for RTX 3090 24GB)
# ---------------------------------------------------------
DEFAULT_DL_CONFIG = {
    "batch_size": 128,
    "learning_rate": 1e-3,
    "seq_len": 96,
    "max_epochs": 50,
    "patience": 12,
    "min_delta": 1e-6,
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
    "LSTM": {
        "batch_size": 1024,
        "num_workers": 12,
    },
    "GRU": {
        "batch_size": 1024,
        "num_workers": 12,
    },
    "Ridge": {
        "alphas": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    },
}

# Default input history window length (read from hierarchical configuration).
SEQ_LEN = int(DEFAULT_DL_CONFIG["seq_len"])

# Rolling evaluation stride (slide once per day).
STRIDE = 24

# Random seed.
SEED = 42


# =================================
# 2. Dataset definition (continuous sliding window)
# =================================

class ContinuousTimeSeriesDataset(Dataset):
    """Continuous time series sliding window dataset (no stay_id segmentation involved)."""

    def __init__(self, x_array, y_array, seq_len, pred_len, stride=1):
        # Pre-convert to contiguous float32 tensors to avoid repeated tensor construction in __getitem__ causing CPU overhead.
        self.x_tensor = torch.from_numpy(np.ascontiguousarray(x_array, dtype=np.float32))
        self.y_tensor = torch.from_numpy(np.ascontiguousarray(y_array, dtype=np.float32))
        # Save the history window length.
        self.seq_len = seq_len
        # Save the prediction window length.
        self.pred_len = pred_len
        # Save the sliding window stride.
        self.stride = stride
        # Construct all valid window start points.
        self.indices = np.arange(0, len(x_array) - seq_len - pred_len + 1, stride)

    def __len__(self):
        # Return the number of samples.
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the start point of the current sample.
        start = self.indices[idx]
        # Slice the input window.
        x_window = self.x_tensor[start:start + self.seq_len]
        # Slice the label window.
        y_window = self.y_tensor[start + self.seq_len:start + self.seq_len + self.pred_len]
        # Return the pre-constructed tensor slice.
        return x_window, y_window


class KDDBeijingTemporalSplitDataset(Dataset):
    """KDD Beijing strict temporal split dataset (supports 7:1:2 segmented window extraction)."""

    def __init__(self, x_array, y_array, split_start, split_end, seq_len, pred_len, stride=1):
        # Pre-convert to contiguous float32 tensors to reduce per-sample conversion cost in DataLoader.
        self.x_tensor = torch.from_numpy(np.ascontiguousarray(x_array, dtype=np.float32))
        self.y_tensor = torch.from_numpy(np.ascontiguousarray(y_array, dtype=np.float32))
        self.split_start = int(split_start)
        self.split_end = int(split_end)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)

        if self.split_start < 0 or self.split_end > len(x_array) or self.split_start >= self.split_end:
            raise ValueError("Invalid temporal split interval, please check split_start/split_end.")

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
        x_window = self.x_tensor[start:start + self.seq_len]
        y_window = self.y_tensor[start + self.seq_len:start + self.seq_len + self.pred_len]
        return x_window, y_window


# =========================
# 3. Model definitions (centrally maintained)
# =========================

class LSTMForecaster(nn.Module):
    """LSTM multivariate forecaster."""

    def __init__(self, num_features, hidden_size, num_layers, pred_len):
        # Call parent class constructor.
        super().__init__()
        # Save feature dimension.
        self.num_features = num_features
        # Save prediction horizon length.
        self.pred_len = pred_len
        # Define the LSTM backbone.
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Define the output projection layer.
        self.proj = nn.Linear(hidden_size, pred_len * num_features)

    def forward(self, x):
        # Execute LSTM forward propagation.
        out, _ = self.lstm(x)
        # Take the last time step hidden state.
        last_hidden = out[:, -1, :]
        # Linearly project to prediction vector.
        pred_flat = self.proj(last_hidden)
        # Reshape to [B, pred_len, C].
        pred = pred_flat.view(-1, self.pred_len, self.num_features)
        # Return prediction result.
        return pred


class GRUForecaster(nn.Module):
    """GRU multivariate forecaster."""

    def __init__(self, num_features, hidden_size, num_layers, pred_len):
        # Call parent class constructor.
        super().__init__()
        # Save feature dimension.
        self.num_features = num_features
        # Save prediction horizon length.
        self.pred_len = pred_len
        # Define the GRU backbone.
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Define the output projection layer.
        self.proj = nn.Linear(hidden_size, pred_len * num_features)

    def forward(self, x):
        # Execute GRU forward propagation.
        out, _ = self.gru(x)
        # Take the last time step hidden state.
        last_hidden = out[:, -1, :]
        # Linearly project to prediction vector.
        pred_flat = self.proj(last_hidden)
        # Reshape to [B, pred_len, C].
        pred = pred_flat.view(-1, self.pred_len, self.num_features)
        # Return prediction result.
        return pred


class DLinearForecaster(nn.Module):
    """DLinear forecaster (per-channel linear mapping)."""

    def __init__(self, seq_len, pred_len, enc_in):
        # Call parent class constructor.
        super().__init__()
        # Save input length.
        self.seq_len = seq_len
        # Save prediction length.
        self.pred_len = pred_len
        # Save number of feature channels.
        self.enc_in = enc_in
        # One independent linear head per channel.
        self.linears = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(enc_in)])

    def forward(self, x):
        # Initialize output list.
        outputs = []
        # Compute prediction per channel.
        for i in range(self.enc_in):
            # Take single-channel series.
            x_i = x[:, :, i]
            # Linearly map to prediction length.
            y_i = self.linears[i](x_i)
            # Save single-channel result.
            outputs.append(y_i)
        # Merge all channel outputs.
        pred = torch.stack(outputs, dim=2)
        # Return prediction result.
        return pred


class PatchTSTForecaster(nn.Module):
    """Simplified PatchTST forecaster."""

    def __init__(self, seq_len, pred_len, enc_in, patch_len=16, stride=8, e_layers=2, d_model=128):
        # Call parent class constructor.
        super().__init__()
        # Save input length.
        self.seq_len = seq_len
        # Save prediction length.
        self.pred_len = pred_len
        # Save number of channels.
        self.enc_in = enc_in
        # Save patch length.
        self.patch_len = patch_len
        # Save patch stride.
        self.stride = stride

        # Compute number of patches.
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Per-channel independent patch embedding layer.
        self.patch_embedding = nn.ModuleList([nn.Linear(patch_len, d_model) for _ in range(enc_in)])

        # Define Transformer encoder layer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )

        # Stack Transformer encoder layers.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Per-channel independent prediction head.
        self.pred_heads = nn.ModuleList([
            nn.Linear(d_model * self.n_patches, pred_len) for _ in range(enc_in)
        ])

    def forward(self, x):
        # Get input batch size and number of channels.
        batch_size, _, n_features = x.shape
        # Initialize multi-channel output list.
        outputs = []

        # Per-channel patch modeling and prediction.
        for i in range(n_features):
            # Take the time series of the i-th channel.
            x_i = x[:, :, i]
            # Initialize patch list.
            patches = []
            # Loop to slice patches.
            for j in range(self.n_patches):
                # Compute current patch start point.
                start = j * self.stride
                # Compute current patch end point.
                end = start + self.patch_len
                # Extract the patch.
                patch = x_i[:, start:end]
                # Add to list.
                patches.append(patch)

            # Stack patch tensors.
            patches = torch.stack(patches, dim=1)
            # Perform patch embedding.
            emb = self.patch_embedding[i](patches)
            # Perform Transformer encoding.
            enc = self.encoder(emb)
            # Flatten the encoded output.
            enc_flat = enc.reshape(batch_size, -1)
            # Predict the future series.
            y_i = self.pred_heads[i](enc_flat)
            # Save current channel prediction.
            outputs.append(y_i)

        # Merge all channel outputs.
        pred = torch.stack(outputs, dim=2)
        # Return prediction result.
        return pred


# =====================
# 4. Data loading and preprocessing
# =====================




def load_and_preprocess_data():
    """
    Load AQI-36 data and perform strict leakage-proof standardization.

    Returns:
        train_data_full: standardized training segment [train_T, C]
        val_data_full: standardized validation segment [val_T, C]
        test_data_full: standardized test segment [test_T, C]
        full_clean_data: standardized full series [T, C]
        value_cols: list of valid feature column names
        split_info: temporal split index info
    """
    # Validate data file existence.
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    # Directly read KDD_Beijing_Clean.csv (no longer going through TSF parsing logic).
    df = pd.read_csv(DATA_FILE)
    if df.empty:
        raise ValueError("CSV file is empty, cannot continue training.")

    # Keep only numeric columns, automatically drop non-numeric fields such as timestamps.
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] == 0:
        raise ValueError("No usable numeric columns found in the CSV.")

    # Get raw column names and numeric matrix.
    raw_value_cols = list(numeric_df.columns)
    raw_data = numeric_df.values.astype(np.float64)

    # Compute variance for each column.
    variances = np.nanvar(raw_data, axis=0)
    # Filter out constant columns.
    non_constant_mask = variances > 1e-5

    # Apply valid column mask.
    data = raw_data[:, non_constant_mask]
    # Generate list of valid column names.
    value_cols = [raw_value_cols[i] for i in range(len(raw_value_cols)) if non_constant_mask[i]]

    # Compute total length.
    total_len = len(data)
    # Compute 7:1:2 temporal split points.
    train_end = int(total_len * TRAIN_RATIO)
    val_end = train_end + int(total_len * VAL_RATIO)

    # Defensive correction: prevent split points from going out of bounds.
    train_end = max(1, min(train_end, total_len - 2))
    val_end = max(train_end + 1, min(val_end, total_len - 1))

    # Fit standardizer on the training segment to prevent data leakage.
    scaler = StandardScaler()
    scaler.fit(data[:train_end])

    # Uniformly transform full data (using only training segment statistics).
    full_clean_data = scaler.transform(data)

    # Split three segments by time.
    train_data_full = full_clean_data[:train_end]
    val_data_full = full_clean_data[train_end:val_end]
    test_data_full = full_clean_data[val_end:]

    # Print data summary.
    print("=" * 80)
    print("AQI-36 data loading complete")
    print(f"Total length: {total_len}")
    print(f"Training length: {len(train_data_full)}")
    print(f"Validation length: {len(val_data_full)}")
    print(f"Test length: {len(test_data_full)}")
    print(f"Original variable count: {len(raw_value_cols)}")
    print(f"Valid variable count: {len(value_cols)}")
    print("=" * 80)

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
# 5. Missing data injection routing function
# ====================

def inject_and_impute(train_clean, missing_mode, missing_rate, impute_method, verbose=True, fitted_imputer=None):
    """Route to the corresponding injection + imputation function by missing data mechanism, and return IE metrics."""
    # MCAR (completely at random) missing branch.
    if missing_mode == "MCAR":
        return inject_mcar_missing_and_impute(train_clean, missing_rate, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)

    # MAR (block missing) branch.
    if missing_mode == "MAR_Block":
        return inject_mar_block_missing_and_impute(train_clean, missing_rate, 24, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)

    # MNAR (value-dependent missing) branch.
    if missing_mode == "MNAR":
        return inject_mnar_value_missing_and_impute(train_clean, missing_rate, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)

    # Unknown mechanism raises an error directly.
    raise ValueError(f"Unknown missing mode: {missing_mode}")


# =====================
# 6. Experiment state persistence utilities
# =====================

def _load_state(checkpoint_file, results_file):
    """Load checkpoint file and historical results file."""
    # Initialize results list.
    results = []
    # Initialize completed keys set.
    completed_keys = set()

    # Read checkpoint file if it exists.
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            completed_keys = set(json.load(f))

    # Read results file if it exists.
    if os.path.exists(results_file):
        results = pd.read_csv(results_file).to_dict("records")

    # Return state.
    return results, completed_keys


def _save_state(checkpoint_file, results_file, results, completed_keys):
    """Save checkpoint and results to disk."""
    # Write checkpoint keys set.
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, ensure_ascii=False)

    # Write CSV if results exist.
    if results:
        pd.DataFrame(results).to_csv(results_file, index=False)


# =====================
# 7. Experiment combinations and model construction
# =====================

def build_experiments():
    """Build the full experiment grid and sort by imputation complexity."""
    # Initialize experiment list.
    experiments = []

    # Iterate over prediction horizons.
    for pred_len in PRED_LEN_LIST:
        # Iterate over missing data mechanisms.
        for missing_mode in MISSING_MODES:
            # Iterate over missing rates.
            for missing_rate in MISSING_RATES:
                # Iterate over imputation methods.
                for impute_method in IMPUTE_METHODS:
                    # Add current combination.
                    experiments.append(
                        {
                            "pred_len": pred_len,
                            "missing_mode": missing_mode,
                            "missing_rate": missing_rate,
                            "impute_method": impute_method,
                        }
                    )

    # Define imputation complexity priority.
    method_order = {"mean": 0, "spline": 1, "knn": 2, "brits": 3, "saits": 4}
    # Perform sorting.
    experiments.sort(key=lambda x: (method_order[x["impute_method"]], x["pred_len"]))

    # Return experiment list.
    return experiments


def build_model(model_name, num_features, pred_len, seq_len=SEQ_LEN):
    """Create a deep learning model instance based on the model name."""
    # LSTM branch.
    if model_name == "LSTM":
        return LSTMForecaster(num_features=num_features, hidden_size=64, num_layers=2, pred_len=pred_len)
    # GRU branch.
    if model_name == "GRU":
        return GRUForecaster(num_features=num_features, hidden_size=64, num_layers=2, pred_len=pred_len)
    # DLinear branch.
    if model_name == "DLinear":
        return DLinearForecaster(seq_len=seq_len, pred_len=pred_len, enc_in=num_features)
    # PatchTST branch.
    if model_name == "PatchTST":
        return PatchTSTForecaster(seq_len=seq_len, pred_len=pred_len, enc_in=num_features)
    # Fallback error branch.
    raise ValueError(f"Unknown model name: {model_name}")


# ===========================
# 8. Deep learning model training and evaluation
# ===========================

def train_and_eval_dl_model(model_name, full_imputed_data, full_clean_data, split_info, pred_len, missing_mode, missing_rate):
    """Train and evaluate a single deep learning model, returning FE metrics and window statistics."""
    # Merge default config with model-specific config.
    model_cfg = {**DEFAULT_DL_CONFIG, **MODEL_SPECIFIC_CONFIG.get(model_name, {})}
    # Read current model batch size.
    batch_size = int(model_cfg["batch_size"])
    # Read current model learning rate.
    learning_rate = float(model_cfg["learning_rate"])
    # Read current model maximum epochs.
    max_epochs = int(model_cfg["max_epochs"])
    # Read current model early stopping patience.
    patience = int(model_cfg["patience"])
    # Read current model minimum improvement threshold.
    min_delta = float(model_cfg["min_delta"])
    # Read current model sequence length.
    seq_len = int(model_cfg["seq_len"])

    # Read strict temporal split points.
    train_end = int(split_info["train_end"])
    val_end = int(split_info["val_end"])
    total_len = int(split_info["total_len"])

    # Extract train/val/test windows on the full series by temporal boundaries.
    train_dataset = KDDBeijingTemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=0,
        split_end=train_end,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=1,
    )
    val_dataset = KDDBeijingTemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=train_end,
        split_end=val_end,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=1,
    )
    test_dataset = KDDBeijingTemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=val_end,
        split_end=total_len,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=STRIDE,
    )

    # Defensive check on window count.
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Window count is 0, please check seq_len/pred_len or the 7:1:2 temporal split boundaries.")

    # Enable pin_memory under CUDA to accelerate CPU-to-GPU data copy.
    use_pin_memory = device.type == "cuda"

    # Under CUDA training, prefer multi-process prefetching to reduce the chance of the main process CPU becoming a bottleneck.
    cpu_count = os.cpu_count() or 1
    default_workers = min(16, max(4, cpu_count // 2)) if device.type == "cuda" else 0
    num_workers = max(0, int(model_cfg.get("num_workers", default_workers)))

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "pin_memory": use_pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    # Build training data loader.
    train_loader = DataLoader(train_dataset, **{**loader_kwargs, "shuffle": True})
    # Build validation data loader.
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    # Build test data loader.
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    print(
        f"   [{model_name}] device={device.type}, batch_size={batch_size}, "
        f"num_workers={num_workers}, pin_memory={use_pin_memory}"
    )

    # Create model instance.
    model = build_model(model_name=model_name, num_features=full_imputed_data.shape[1], pred_len=pred_len, seq_len=seq_len)
    # Move model to the target device.
    model = model.to(device)

    # Create Adam optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Create MSE loss function.
    criterion = masked_mse_loss
    # Generate a unique weight file name for the current experiment to avoid parallel experiments overwriting each other.
    missing_mode_tag = str(missing_mode).replace("/", "_")
    weight_path = os.path.join(
        BASE_DIR,
        f"best_model_{model_name}_{missing_mode_tag}_{int(round(float(missing_rate) * 100))}_{pred_len}.pth",
    )
    # Initialize early stopping state.
    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    try:
        # Iterate training over multiple epochs (perform full validation after each round).
        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device, non_blocking=use_pin_memory)
                batch_y = batch_y.to(device, non_blocking=use_pin_memory)

                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / max(1, len(train_loader))
            val_loss = _run_eval_loss(model=model, data_loader=val_loader)

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
                        f"validation loss did not improve for {patience} consecutive rounds."
                    )
                    break

        # Load the best weights selected during validation and compute FE on the test set.
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))

        # Enter evaluation mode.
        model.eval()
        # Initialize prediction container.
        preds = []
        # Initialize ground-truth container.
        trues = []

        # Disable gradient computation.
        with torch.no_grad():
            # Iterate over test batches.
            for batch_x, batch_y in test_loader:
                # Move input tensors.
                batch_x = batch_x.to(device, non_blocking=use_pin_memory)
                # Forward predict and convert to numpy.
                batch_pred = model(batch_x).cpu().numpy()
                # Save predictions.
                preds.append(batch_pred)
                # Save ground truth.
                trues.append(batch_y.numpy())

        # Concatenate all predictions.
        preds = np.concatenate(preds, axis=0)
        # Concatenate all ground truth.
        trues = np.concatenate(trues, axis=0)

        # Compute per-channel errors then average across channels to avoid high-variance variables dominating the overall metric.
        c = preds.shape[-1]
        trues_reshaped = trues.reshape(-1, c)
        preds_reshaped = preds.reshape(-1, c)
        per_channel_abs = np.nanmean(np.abs(trues_reshaped - preds_reshaped), axis=0)
        per_channel_sq = np.nanmean(np.square(trues_reshaped - preds_reshaped), axis=0)
        fe_mae = float(np.nanmean(per_channel_abs))
        fe_mse = float(np.nanmean(per_channel_sq))

        # Print test metrics.
        print(f"   [{model_name} Test] FE_MAE={fe_mae:.4f}, FE_MSE={fe_mse:.4f}")

        # Actively release CUDA memory cache after evaluation to reduce fragmentation risk.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Return evaluation results and window count.
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
# 9. Ridge regression model evaluation (reuses utilities, replaces original VAR)
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

    Uses L2-regularized linear regression to replace the original VAR: the history window length uniformly
    uses the global SEQ_LEN (keeping the same lookback window as deep learning models such as
    LSTM/DLinear/PatchTST for horizontal comparability), and no longer dynamically shrinks the order
    based on training segment length / number of features as VAR did -- Ridge's L2 regularization
    can itself handle high-dimensional, strongly correlated design matrices, naturally avoiding the
    problem where VAR's covariance matrix becomes nearly singular and forecasts diverge under high
    missing rate + block missing (i.e. the 10 records excluded in data hygiene rule 1).

    Returns:
        fe_mae: forecast error MAE
        fe_mse: forecast error MSE
        train_windows: number of training windows (sliding-window samples used to fit Ridge)
        test_windows: estimated number of test windows
    """
    # Read strict temporal split points.
    train_end = int(split_info["train_end"])
    val_end = int(split_info["val_end"])
    total_len = int(split_info["total_len"])

    # History window length uniformly uses the global SEQ_LEN, comparable to deep learning models, no longer dynamically computed.
    lag_order = SEQ_LEN

    # Read this model's alpha candidate grid (automatically selected by RidgeCV via internal cross-validation).
    ridge_cfg = MODEL_SPECIFIC_CONFIG.get({})
    alphas = ridge_cfg.get("alphas", None)

    # Call utility function to execute rolling forecast.
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

    # Training window count: number of sliding-window samples actually used to fit Ridge.
    train_windows = max(1, train_end - lag_order - pred_len + 1)
    # Roughly estimate the number of test windows (for record-keeping).
    test_windows = max(1, (total_len - val_end - pred_len) // STRIDE + 1)

    # Return Ridge metrics.
    return float(fe_mae), float(fe_mse), int(train_windows), int(test_windows)


# =======================
# 10. Unified single-model execution entry
# =======================

def run_single_model(model_name):
    """Execute training and evaluation of a single model on the full experiment grid."""
    # Build model-specific results file path.
    results_file = os.path.join(BASE_DIR, f"results_refactor_{model_name.lower()}.csv")
    # Build model-specific checkpoint file path.
    checkpoint_file = os.path.join(BASE_DIR, f"checkpoint_refactor_{model_name.lower()}.json")

    # Load and preprocess data.
    train_data_full, val_data_full, test_data_full, full_clean_data, value_cols, split_info = load_and_preprocess_data()

    # Load historical state.
    results, completed_keys = _load_state(checkpoint_file=checkpoint_file, results_file=results_file)

    # Build experiment grid.
    experiments = build_experiments()

    # Print execution summary.
    print(f"Model: {model_name}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Results file: {results_file}")
    print(f"Checkpoint file: {checkpoint_file}")

    # Iterate over all experiment combinations.
    for exp_idx, exp in enumerate(experiments, 1):
        # Read prediction horizon length.
        pred_len = exp["pred_len"]
        # Read missing data mechanism.
        missing_mode = exp["missing_mode"]
        # Read missing rate.
        missing_rate = exp["missing_rate"]
        # Read imputation method.
        impute_method = exp["impute_method"]

        # Assemble unique experiment key.
        exp_key = f"{model_name}_{missing_mode}_{impute_method}_{pred_len}_{int(missing_rate * 100)}"

        # Skip if already completed.
        if exp_key in completed_keys:
            print(f"[{exp_idx}/{len(experiments)}] Skipping completed: {exp_key}")
            continue

        # Print current experiment header.
        print("\n" + "=" * 90)
        print(
            f"[{exp_idx}/{len(experiments)}] model={model_name}, missing_mode={missing_mode}, "
            f"imputation={impute_method}, pred_len={pred_len}, missing_rate={int(missing_rate * 100)}%"
        )
        print("=" * 90)

        # Record start time.
        start_time = time.time()

        try:
            # ----------------------------------------------------------
            # Cache-first imputation flow: reuse cache first, then compute and write cache on miss.
            # Note: the cache key does not include pred_len because imputation depends only on missing mechanism / method / missing rate.
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
                print(f"⚡ Fast-loading imputation cache: {cache_path}")
            else:
                # Inject missing data and impute only on the training segment to construct training input; also obtain the fitted imputer.
                train_imputed, _, _, fitted_imp = inject_and_impute(
                    train_clean=train_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                )

                # DL imputation methods (SAITS/BRITS) train independently per split, do not reuse the fitted imputer.
                reuse = None if impute_method in ("saits", "brits") else fitted_imp

                # Inject missing data and impute on the validation segment, reusing the imputer fitted on the training segment to avoid data leakage.
                val_imputed, _, _, _ = inject_and_impute(
                    train_clean=val_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    fitted_imputer=reuse,
                )

                # Inject missing data and impute on the test segment, reusing the imputer fitted on the training segment; IE is only computed on the test segment.
                test_imputed, ie_mae, ie_mse, _ = inject_and_impute(
                    train_clean=test_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    fitted_imputer=reuse,
                )

                # Reconstruct full input by 7:1:2: training segment imputed, validation segment imputed, test segment imputed.
                full_imputed_data = np.vstack((train_imputed, val_imputed, test_imputed))

                # Persist the full imputed data and IE metrics so subsequent models can directly reuse them.
                np.savez(cache_path, full_data=full_imputed_data, ie_mae=ie_mae, ie_mse=ie_mse)
                print(f"💾 Imputation computed and cached: {cache_path}")

            # Ridge branch calls the statistical model evaluation logic.
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
                # Deep learning branch calls the unified training-evaluation logic.
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

            # Attempt to release CUDA cache after each round of model evaluation.
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Compute elapsed time.
            elapsed_sec = round(time.time() - start_time, 2)

            # Organize success record.
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

            # Save success record.
            results.append(result_row)
            # Mark as completed.
            completed_keys.add(exp_key)

            # Print success summary.
            print(
                f"✅ Done: {exp_key} | IE_MSE(Test)={ie_mse:.4f} | FE_MSE(Test)={fe_mse:.4f} | "
                f"BestVal={best_val_loss if best_val_loss is not None else 'N/A'} | elapsed={elapsed_sec:.2f}s"
            )

        except Exception as exc:
            # Compute exception elapsed time.
            elapsed_sec = round(time.time() - start_time, 2)
            # Truncate error message.
            error_msg = str(exc)[:200]

            # Print error summary.
            print(f"❌ Failed: {exp_key} | error={error_msg}")
            # Print full stack trace for troubleshooting.
            traceback.print_exc()

            # Organize failure record.
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

            # Save failure record.
            results.append(result_row)
            # Mark failed as completed too, to avoid infinite retry loops.
            completed_keys.add(exp_key)

        # Save checkpoint and results after each round of experiments.
        _save_state(
            checkpoint_file=checkpoint_file,
            results_file=results_file,
            results=results,
            completed_keys=completed_keys,
        )

        # Actively trigger garbage collection.
        gc.collect()

    # Print model completion prompt.
    print("\n" + "=" * 90)
    print(f"Model {model_name} all experiments completed")
    print(f"Results written to: {results_file}")
    print("=" * 90)


# =======================
# 11. Overall scheduling entry (optional)
# =======================

def run_all_models(model_list=None):
    """Sequentially execute multiple models, by default running five baseline models."""
    # Use default model order.
    if model_list is None:
        model_list = ["LSTM", "DLinear", "PatchTST"]

    # Execute each model in turn.
    for model_name in model_list:
        run_single_model(model_name)
