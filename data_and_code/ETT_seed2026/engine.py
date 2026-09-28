# -*- coding: utf-8 -*-

# Import system-related libraries for path handling, exception tracing, timing, garbage collection, and log storage.
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

# Import classic machine learning evaluation and standardization utilities.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import PyTorch ecosystem for deep learning model construction and training.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Unified hardware device detection: prefer CUDA, then MPS, finally CPU.
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Enable cuDNN auto-tuning for fixed input shape tasks to boost 3090 throughput.
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Suppress a large number of irrelevant warnings to avoid overly noisy terminal output.
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Get the directory of the current script, so it can locate files correctly regardless of the working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the Python module search path so same-directory tool modules are importable.
sys.path.insert(0, BASE_DIR)

# Reuse missing data injection and Ridge regression rolling evaluation logic from historical utilities (replaces the original VAR, avoiding numerical divergence under high missing rate + block missing).
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
# 1. Global configuration (preserve continuous-data logic)
# =============================

# Data file path (ETT main dataset, using the standard default naming).
DATA_FILE = os.path.join(BASE_DIR, "ETTh1.csv")

# Variable columns used by ETT, kept consistent with the original script.
VALUE_COLS = ["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

# Missing data mechanism set (3 types).
MISSING_MODES = ["MCAR", "MAR_Block", "MNAR"]

# Imputation method set (5 types).
IMPUTE_METHODS = ["mean", "spline", "knn", "brits", "saits"]

# Prediction horizon list (ETT original logic).
PRED_LEN_LIST = [24, 96]

# Missing rate set (4 levels).
MISSING_RATES = [0.10, 0.30, 0.50, 0.70]

# Fixed train/val/test split ratios (continuous series 7:1:2).
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# Deep learning input history window length (kept at 96 per the ETT original logic).
SEQ_LEN = 96

# Rolling evaluation stride (kept at 24 per the original logic).
STRIDE = 24

# Deep learning default training configuration (base config).
DEFAULT_DL_CONFIG = {
    "batch_size": 2048,
    "learning_rate": 1e-3,
    "max_epochs": 50,
    "num_workers": 12,
}

# Model-specific override configuration (only overrides the differing items).
MODEL_SPECIFIC_CONFIG = {
    "DLinear": {"batch_size": 4096, "learning_rate": 5e-4, "num_workers": 12},
    "PatchTST": {"batch_size": 512, "learning_rate": 1e-4, "num_workers": 12},
    "LSTM": {"batch_size": 2048, "num_workers": 12},
    "GRU": {"batch_size": 2048, "num_workers": 12},
    "Ridge": {
        "alphas": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    },  # RidgeCV candidate alpha grid; the internal leave-one-out cross-validation selects automatically
}

# Early stopping patience (stop when validation does not improve for this many consecutive rounds).
EARLY_STOPPING_PATIENCE = 12

# Minimum improvement threshold for validation loss.
EARLY_STOPPING_MIN_DELTA = 1e-6

# Random seed to guarantee reproducible missing injection and experiments.
SEED = 2026


# =================================
# 2. Dataset definition (continuous sliding window, no boundary crossing)
# =================================


class ContinuousTimeSeriesDataset(Dataset):
    """
    Continuous time series sliding window dataset.

    Input tensor shapes:
        x_window: [seq_len, n_features]
        y_window: [pred_len, n_features]

    Note:
        This is the standard sliding window for "continuous physical data" and does not use any stay_id or patient segment boundaries.
    """

    def __init__(self, x_array, y_array, seq_len, pred_len, stride=1):
        # Save the input feature array.
        self.x_array = x_array
        # Save the supervision target array.
        self.y_array = y_array
        # Save the history window length.
        self.seq_len = seq_len
        # Save the prediction window length.
        self.pred_len = pred_len
        # Save the sliding window stride.
        self.stride = stride
        # Build all valid start indices to ensure every window has a complete length.
        self.indices = np.arange(0, len(x_array) - seq_len - pred_len + 1, stride)

    def __len__(self):
        # Return the number of available windows.
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the start position for the current sample.
        start = self.indices[idx]
        # Slice the input history window of shape [seq_len, n_features].
        x_window = self.x_array[start : start + self.seq_len]
        # Slice the future label window of shape [pred_len, n_features].
        y_window = self.y_array[
            start + self.seq_len : start + self.seq_len + self.pred_len
        ]
        # Convert to float32 tensors for model training.
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(
            y_window, dtype=torch.float32
        )


class ETTh1TemporalSplitDataset(Dataset):
    """ETTh1 strict temporal split dataset (supports 7:1:2 segmented window extraction)."""

    def __init__(
        self, x_array, y_array, split_start, split_end, seq_len, pred_len, stride=1
    ):
        # Save the input feature array.
        self.x_array = x_array
        # Save the supervision target array.
        self.y_array = y_array
        # Save the split start position.
        self.split_start = int(split_start)
        # Save the split end position.
        self.split_end = int(split_end)
        # Save the history window length.
        self.seq_len = int(seq_len)
        # Save the prediction window length.
        self.pred_len = int(pred_len)
        # Save the sliding window stride.
        self.stride = int(stride)

        # Validate the split boundary legality.
        if (
            self.split_start < 0
            or self.split_end > len(x_array)
            or self.split_start >= self.split_end
        ):
            raise ValueError("Invalid temporal split interval; please check split_start/split_end.")

        # Compute the minimum allowed window start position.
        min_start = max(0, self.split_start - self.seq_len)
        # Compute the maximum allowed window start position.
        max_start = self.split_end - self.seq_len - self.pred_len

        # Collect all valid window start positions.
        indices = []
        if max_start >= min_start:
            for start in range(min_start, max_start + 1, self.stride):
                # Compute the label window start position.
                y_start = start + self.seq_len
                # Compute the label window end position.
                y_end = y_start + self.pred_len
                # Keep only windows whose labels fall entirely inside the current split segment.
                if y_start >= self.split_start and y_end <= self.split_end:
                    indices.append(start)

        # Save the window start index array.
        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self):
        # Return the number of available windows.
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the start position for the current sample.
        start = int(self.indices[idx])
        # Slice the input history window.
        x_window = self.x_array[start : start + self.seq_len]
        # Slice the future label window.
        y_window = self.y_array[
            start + self.seq_len : start + self.seq_len + self.pred_len
        ]
        # Return float32 tensors.
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(
            y_window, dtype=torch.float32
        )


# =========================
# 3. Model definitions (centrally kept in the engine)
# =========================


class LSTMForecaster(nn.Module):
    """LSTM multivariate-to-multivariate forecaster."""

    def __init__(self, num_features, hidden_size, num_layers, pred_len):
        # Call the parent class constructor.
        super().__init__()
        # Save the feature dimension.
        self.num_features = num_features
        # Save the prediction length.
        self.pred_len = pred_len
        # Build the LSTM encoder with input shape [B, T, C].
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Linear layer that maps the last hidden state to pred_len * num_features.
        self.proj = nn.Linear(hidden_size, pred_len * num_features)

    def forward(self, x):
        # Input x has shape [B, seq_len, C].
        out, _ = self.lstm(x)
        # Take the last time step hidden representation, shape [B, hidden_size].
        last_hidden = out[:, -1, :]
        # Linearly map to [B, pred_len * C].
        pred_flat = self.proj(last_hidden)
        # Reshape to [B, pred_len, C] to align with the label shape.
        pred = pred_flat.view(-1, self.pred_len, self.num_features)
        # Return the prediction tensor.
        return pred


class GRUForecaster(nn.Module):
    """GRU multivariate-to-multivariate forecaster."""

    def __init__(self, num_features, hidden_size, num_layers, pred_len):
        # Call the parent class constructor.
        super().__init__()
        # Save the feature dimension.
        self.num_features = num_features
        # Save the prediction length.
        self.pred_len = pred_len
        # Build the GRU encoder with input shape [B, T, C].
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Linear layer that maps the last hidden state to pred_len * num_features.
        self.proj = nn.Linear(hidden_size, pred_len * num_features)

    def forward(self, x):
        # Input x has shape [B, seq_len, C].
        out, _ = self.gru(x)
        # Take the last time step hidden representation, shape [B, hidden_size].
        last_hidden = out[:, -1, :]
        # Linearly map to [B, pred_len * C].
        pred_flat = self.proj(last_hidden)
        # Reshape to [B, pred_len, C].
        pred = pred_flat.view(-1, self.pred_len, self.num_features)
        # Return the prediction tensor.
        return pred


class DLinearForecaster(nn.Module):
    """DLinear forecaster (per-channel linear mapping)."""

    def __init__(self, seq_len, pred_len, enc_in):
        # Call the parent class constructor.
        super().__init__()
        # Save the input length.
        self.seq_len = seq_len
        # Save the prediction length.
        self.pred_len = pred_len
        # Save the number of channels.
        self.enc_in = enc_in
        # Create an independent linear layer per channel to perform [seq_len] -> [pred_len].
        self.linears = nn.ModuleList(
            [nn.Linear(seq_len, pred_len) for _ in range(enc_in)]
        )

    def forward(self, x):
        # Input x has shape [B, seq_len, C].
        outputs = []
        # Linearly forecast per channel.
        for i in range(self.enc_in):
            # Take the i-th channel, resulting in [B, seq_len].
            x_i = x[:, :, i]
            # Linear mapping yields [B, pred_len].
            y_i = self.linears[i](x_i)
            # Save the current channel prediction.
            outputs.append(y_i)
        # Stack to get [B, pred_len, C].
        pred = torch.stack(outputs, dim=2)
        # Return the prediction tensor.
        return pred


class PatchTSTForecaster(nn.Module):
    """Simplified PatchTST forecaster."""

    def __init__(
        self, seq_len, pred_len, enc_in, patch_len=16, stride=8, e_layers=2, d_model=128
    ):
        # Call the parent class constructor.
        super().__init__()
        # Save the input length.
        self.seq_len = seq_len
        # Save the prediction length.
        self.pred_len = pred_len
        # Save the number of channels.
        self.enc_in = enc_in
        # Save the patch length.
        self.patch_len = patch_len
        # Save the patch stride.
        self.stride = stride

        # Compute the number of patches; shape derivation: n_patches = floor((seq_len - patch_len)/stride) + 1.
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Independent per-channel patch embedding layers mapping [patch_len] to [d_model].
        self.patch_embedding = nn.ModuleList(
            [nn.Linear(patch_len, d_model) for _ in range(enc_in)]
        )

        # Define the Transformer encoder layer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )

        # Stack multiple Transformer encoder layers.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Independent per-channel prediction heads mapping [n_patches * d_model] to [pred_len].
        self.pred_heads = nn.ModuleList(
            [nn.Linear(d_model * self.n_patches, pred_len) for _ in range(enc_in)]
        )

    def forward(self, x):
        # Input x has shape [B, seq_len, C].
        batch_size, _, n_features = x.shape
        # Used to save predictions from all channels.
        out_all = []

        # Independently perform patch slicing, encoding, and forecasting per channel.
        for i in range(n_features):
            # Take the i-th channel; shape becomes [B, seq_len].
            x_i = x[:, :, i]
            # Store all patches of this channel.
            patches = []
            # Slice patches one at a time.
            for j in range(self.n_patches):
                # Current patch start position.
                start = j * self.stride
                # Current patch end position.
                end = start + self.patch_len
                # Slice the patch of shape [B, patch_len].
                patch = x_i[:, start:end]
                # Append to the patch list.
                patches.append(patch)

            # Stack patches to get [B, n_patches, patch_len].
            patches = torch.stack(patches, dim=1)
            # After patch embedding we get [B, n_patches, d_model].
            emb = self.patch_embedding[i](patches)
            # After Transformer encoding, the shape remains [B, n_patches, d_model].
            enc = self.encoder(emb)
            # Flatten the patch and feature dimensions to get [B, n_patches * d_model].
            enc_flat = enc.reshape(batch_size, -1)
            # Prediction head outputs [B, pred_len].
            y_i = self.pred_heads[i](enc_flat)
            # Save the current channel prediction.
            out_all.append(y_i)

        # Merge all channels to get [B, pred_len, C].
        pred = torch.stack(out_all, dim=2)
        # Return the prediction tensor.
        return pred


# =====================
# 4. Data loading and preprocessing
# =====================


def load_and_preprocess_data():
    """
    Load ETT data and perform strict leakage-proof standardization.

    Returns:
        train_data_full: standardized training segment, shape [train_T, C]
        val_data_full: standardized validation segment, shape [val_T, C]
        test_data_full: standardized test segment, shape [test_T, C]
        full_clean_data: standardized full series, shape [T, C]
        value_cols: list of valid feature column names
        split_info: temporal split index information
    """
    # Prefer the standard file name; if missing, fall back to the common ETTh1 file.
    candidate_files = [
        DATA_FILE,
        os.path.join(BASE_DIR, "ETTh1.csv"),
    ]
    existing_files = [p for p in candidate_files if os.path.exists(p)]
    if not existing_files:
        raise FileNotFoundError(f"No usable data file found; please check: {candidate_files}")
    data_file = existing_files[0]

    # Read the CSV data.
    df = pd.read_csv(data_file)
    # If a date column exists, drop it first to avoid non-numeric columns affecting standardization.
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # Prefer the pre-defined variable columns; if any are missing, fall back to all numeric columns.
    missing_cols = [c for c in VALUE_COLS if c not in df.columns]
    if len(missing_cols) == 0:
        numeric_df = df[VALUE_COLS].copy()
    else:
        numeric_df = df.select_dtypes(include=[np.number]).copy()

    # If no usable numeric columns exist, raise an error directly.
    if numeric_df.shape[1] == 0:
        raise ValueError(f"No usable numeric columns in {os.path.basename(data_file)}.")

    # Extract the multivariate numeric matrix of shape [T, C].
    data = numeric_df.values.astype(np.float64)
    # Get the list of valid column names.
    value_cols = list(numeric_df.columns)

    # Compute the total length.
    total_len = len(data)
    # Compute the training end index.
    train_end = int(total_len * TRAIN_RATIO)
    # Compute the validation end index.
    val_end = train_end + int(total_len * VAL_RATIO)

    # Defensive correction: prevent split points from going out of bounds.
    train_end = max(1, min(train_end, total_len - 2))
    val_end = max(train_end + 1, min(val_end, total_len - 1))

    # Slice the training segment.
    train_data_raw = data[:train_end]

    # Build the standardizer.
    scaler = StandardScaler()
    # Fit only on the training set to avoid test information leakage.
    scaler.fit(train_data_raw)

    # Transform the full data using the training statistics.
    full_clean_data = scaler.transform(data)

    # Split the standardized data by 7:1:2.
    train_data_full = full_clean_data[:train_end]
    val_data_full = full_clean_data[train_end:val_end]
    test_data_full = full_clean_data[val_end:]

    # Print data overview information.
    print("=" * 80)
    print("ETT data loading complete")
    print(f"Data file: {os.path.basename(data_file)}")
    print(f"Total length: {total_len}")
    print(f"Training length: {len(train_data_full)}")
    print(f"Validation length: {len(val_data_full)}")
    print(f"Test length: {len(test_data_full)}")
    print(f"Feature dimension: {len(value_cols)}")
    print("=" * 80)

    # Build the temporal split info dictionary.
    split_info = {
        "train_end": int(train_end),
        "val_end": int(val_end),
        "total_len": int(total_len),
    }

    # Return the preprocessing results.
    return (
        train_data_full,
        val_data_full,
        test_data_full,
        full_clean_data,
        value_cols,
        split_info,
    )


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
# 5. Missing data injection routing function
# ====================


def inject_and_impute(
    train_clean,
    missing_mode,
    missing_rate,
    impute_method,
    verbose=True,
    fitted_imputer=None,
):
    """
    Route to the corresponding missing mechanism and imputation method.

    Returns:
        train_imputed: training set with missing values injected and then imputed
        ie_mae: imputation error MAE
        ie_mse: imputation error MSE
        fitted_imputer: the imputer or statistic fitted during training
    """
    # MCAR: completely at random missing.
    if missing_mode == "MCAR":
        return inject_mcar_missing_and_impute(
            train_clean,
            missing_rate,
            impute_method,
            SEED,
            verbose=verbose,
            fitted_imputer=fitted_imputer,
        )

    # MAR_Block: block missing across variables.
    if missing_mode == "MAR_Block":
        return inject_mar_block_missing_and_impute(
            train_clean,
            missing_rate,
            24,
            impute_method,
            SEED,
            verbose=verbose,
            fitted_imputer=fitted_imputer,
        )

    # MNAR: missingness that depends on the value magnitude.
    if missing_mode == "MNAR":
        return inject_mnar_value_missing_and_impute(
            train_clean,
            missing_rate,
            impute_method,
            SEED,
            verbose=verbose,
            fitted_imputer=fitted_imputer,
        )

    # Unknown modes raise an error directly.
    raise ValueError(f"Unknown missing mode: {missing_mode}")


# =====================
# 6. Experiment state persistence utilities
# =====================


def _load_state(checkpoint_file, results_file):
    """Load the checkpoint and historical results."""
    # Initialize the results list.
    results = []
    # Initialize the set of completed experiment keys.
    completed_keys = set()

    # Read the checkpoint file if it exists.
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            completed_keys = set(json.load(f))

    # Read the results file if it exists.
    if os.path.exists(results_file):
        results = pd.read_csv(results_file).to_dict("records")

    # Return the results and the completed-key set.
    return results, completed_keys


def _save_state(checkpoint_file, results_file, results, completed_keys):
    """Save the checkpoint and the results."""
    # Write out the completed experiment keys.
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, ensure_ascii=False)

    # Write out the CSV if there are results.
    if results:
        pd.DataFrame(results).to_csv(results_file, index=False)


# =====================
# 7. Experiment combinations and model construction
# =====================


def build_experiments():
    """Build the experiment grid and sort by imputation method complexity."""
    # Initialize the experiment list.
    experiments = []
    # Iterate over all prediction horizons.
    for pred_len in PRED_LEN_LIST:
        # Iterate over all missing mechanisms.
        for missing_mode in MISSING_MODES:
            # Iterate over all missing rates.
            for missing_rate in MISSING_RATES:
                # Iterate over all imputation methods.
                for impute_method in IMPUTE_METHODS:
                    # Record an experiment combination dictionary.
                    experiments.append(
                        {
                            "pred_len": pred_len,
                            "missing_mode": missing_mode,
                            "missing_rate": missing_rate,
                            "impute_method": impute_method,
                        }
                    )

    # Define the imputation method order: fast first, slow last.
    method_order = {"mean": 0, "spline": 1, "knn": 2, "brits": 3, "saits": 4}
    # Sorting rule: first by method complexity, then by prediction horizon.
    experiments.sort(key=lambda x: (method_order[x["impute_method"]], x["pred_len"]))
    # Return the experiment list.
    return experiments


def build_model(model_name, num_features, pred_len):
    """Build the corresponding network based on the model name."""
    # LSTM branch.
    if model_name == "LSTM":
        return LSTMForecaster(
            num_features=num_features, hidden_size=64, num_layers=2, pred_len=pred_len
        )
    # GRU branch.
    if model_name == "GRU":
        return GRUForecaster(
            num_features=num_features, hidden_size=64, num_layers=2, pred_len=pred_len
        )
    # DLinear branch.
    if model_name == "DLinear":
        return DLinearForecaster(
            seq_len=SEQ_LEN, pred_len=pred_len, enc_in=num_features
        )
    # PatchTST branch.
    if model_name == "PatchTST":
        return PatchTSTForecaster(
            seq_len=SEQ_LEN, pred_len=pred_len, enc_in=num_features
        )
    # Raise an error for invalid model names.
    raise ValueError(f"Unknown model name: {model_name}")


# ===========================
# 8. Deep learning model training and evaluation
# ===========================


def train_and_eval_dl_model(
    model_name,
    full_imputed_data,
    full_clean_data,
    split_info,
    pred_len,
    missing_mode,
    missing_rate,
):
    """
    Execute continuous sliding-window training and testing for deep learning models.

    Parameters:
        model_name: model name (LSTM/GRU/DLinear/PatchTST)
        full_imputed_data: imputed full series from the three segments, shape [T, C]
        full_clean_data: standardized clean full series, shape [T, C]
        split_info: temporal split info dictionary
        pred_len: prediction horizon
        missing_mode: name of the missing mechanism
        missing_rate: missing rate

    Returns:
        fe_mae: forecast error MAE
        fe_mse: forecast error MSE
        train_windows: number of training windows
        val_windows: number of validation windows
        test_windows: number of test windows
        best_val_loss: best validation loss
        best_epoch: best epoch
        best_model_path: best model path
    """
    # Read the strict temporal split points.
    train_end = int(split_info["train_end"])
    val_end = int(split_info["val_end"])
    total_len = int(split_info["total_len"])

    # Extract train/val/test windows on the full series by temporal boundaries.
    train_dataset = ETTh1TemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=0,
        split_end=train_end,
        seq_len=SEQ_LEN,
        pred_len=pred_len,
        stride=1,
    )
    val_dataset = ETTh1TemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=train_end,
        split_end=val_end,
        seq_len=SEQ_LEN,
        pred_len=pred_len,
        stride=1,
    )
    test_dataset = ETTh1TemporalSplitDataset(
        x_array=full_imputed_data,
        y_array=full_clean_data,
        split_start=val_end,
        split_end=total_len,
        seq_len=SEQ_LEN,
        pred_len=pred_len,
        stride=STRIDE,
    )

    # Defensive check on window counts.
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError(
            "Window count is 0; please check seq_len/pred_len or the 7:1:2 temporal split boundaries."
        )

    # Merge default configuration with model-specific configuration.
    model_cfg = {**DEFAULT_DL_CONFIG, **MODEL_SPECIFIC_CONFIG.get(model_name, {})}
    # Read the current model's batch size.
    batch_size = int(model_cfg["batch_size"])
    # Read the current model's learning rate.
    learning_rate = float(model_cfg["learning_rate"])
    # Read the current model's maximum epochs.
    max_epochs = int(model_cfg["max_epochs"])
    # Read the DataLoader worker count.
    num_workers = int(model_cfg.get("num_workers", 12))

    # Enable pin_memory under CUDA to speed up CPU-to-GPU copy.
    use_pin_memory = device.type == "cuda"
    # Enable persistent workers for multi-threaded loading to reduce per-round reconstruction overhead.
    persistent_workers = bool(num_workers > 0)

    # Build the training DataLoader.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    # Build the validation DataLoader.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    # Build the test DataLoader.
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    # Create the network based on the model name.
    model = build_model(
        model_name=model_name,
        num_features=full_imputed_data.shape[1],
        pred_len=pred_len,
    )
    # Move the model to the device.
    model = model.to(device)

    # Build the Adam optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = masked_mse_loss
    # Generate a unique weight file name for the current experiment to avoid parallel experiments overwriting each other.
    missing_mode_tag = str(missing_mode).replace("/", "_")
    best_model_path = os.path.join(
        BASE_DIR,
        f"best_model_{model_name}_{missing_mode_tag}_{int(round(float(missing_rate) * 100))}_{pred_len}.pth",
    )
    # Initialize the best validation loss.
    best_val_loss = float("inf")
    # Initialize the best epoch.
    best_epoch = -1
    # Initialize the count of consecutive non-improving rounds.
    bad_epochs = 0

    try:
        # Run multiple epochs.
        for epoch in range(max_epochs):
            # Switch to training mode.
            model.train()
            # Initialize the epoch cumulative loss.
            epoch_loss = 0.0
            # Iterate over all training batches.
            for batch_x, batch_y in train_loader:
                # Move the input to the device, shape [B, seq_len, C].
                batch_x = batch_x.to(device)
                # Move the label to the device, shape [B, pred_len, C].
                batch_y = batch_y.to(device)

                # Zero out gradients.
                optimizer.zero_grad()
                # Forward pass; output shape [B, pred_len, C].
                pred = model(batch_x)
                # Compute the MSE loss.
                loss = criterion(pred, batch_y)
                # Backward pass.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Update parameters.
                optimizer.step()
                # Accumulate the loss.
                epoch_loss += loss.item()

            # Compute the current epoch's training loss.
            train_loss = epoch_loss / max(1, len(train_loader))
            # Compute the current epoch's validation loss.
            val_loss = _run_eval_loss(
                model=model, data_loader=val_loader, criterion=criterion
            )

            # Print the training/validation losses.
            print(
                f"   [{model_name}] Epoch {epoch + 1}/{max_epochs}, "
                f"TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}"
            )

            # Save the best model if the validation loss improves.
            if val_loss < (best_val_loss - EARLY_STOPPING_MIN_DELTA):
                best_val_loss = float(val_loss)
                best_epoch = int(epoch + 1)
                bad_epochs = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                # Increment the consecutive non-improvement count.
                bad_epochs += 1
                # End training early if the early-stopping condition is met.
                if bad_epochs >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"   [{model_name}] EarlyStopping triggered: "
                        f"validation did not improve for {EARLY_STOPPING_PATIENCE} consecutive rounds."
                    )
                    break

        # Load the best weights chosen during validation and compute FE on the test set.
        if os.path.exists(best_model_path):
            model.load_state_dict(
                torch.load(best_model_path, map_location=device, weights_only=True)
            )

        # Switch to evaluation mode.
        model.eval()
        # Initialize the prediction list.
        preds = []
        # Initialize the ground-truth list.
        trues = []

        # Disable gradient computation to reduce memory and speed up inference.
        with torch.no_grad():
            # Iterate over test batches.
            for batch_x, batch_y in test_loader:
                # Move the input to the device.
                batch_x = batch_x.to(device)
                # Forward predict and convert back to numpy.
                batch_pred = model(batch_x).cpu().numpy()
                # Save the predictions.
                preds.append(batch_pred)
                # Save the ground truth.
                trues.append(batch_y.numpy())

        # Concatenate all predictions, shape [N, pred_len, C].
        preds = np.concatenate(preds, axis=0)
        # Concatenate all ground truth, shape [N, pred_len, C].
        trues = np.concatenate(trues, axis=0)

        # Compute error per channel, then average across channels to avoid high-variance variables dominating the overall metric.
        c = preds.shape[-1]
        trues_reshaped = trues.reshape(-1, c)
        preds_reshaped = preds.reshape(-1, c)
        per_channel_abs = np.nanmean(np.abs(trues_reshaped - preds_reshaped), axis=0)
        per_channel_sq = np.nanmean(np.square(trues_reshaped - preds_reshaped), axis=0)
        fe_mae = float(np.nanmean(per_channel_abs))
        fe_mse = float(np.nanmean(per_channel_sq))

        # Print the test metrics.
        print(f"   [{model_name} Test] FE_MAE={fe_mae:.4f}, FE_MSE={fe_mse:.4f}")

        # Actively release CUDA memory cache.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Return the metrics and window counts.
        return (
            fe_mae,
            fe_mse,
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
            float(best_val_loss),
            int(best_epoch),
            best_model_path,
        )
    finally:
        if os.path.exists(best_model_path):
            try:
                os.remove(best_model_path)
            except OSError:
                pass


# =========================
# 9. Ridge regression model evaluation (reuses utilities, replaces the original VAR)
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
    # Read the strict temporal split points.
    train_end = int(split_info["train_end"])
    val_end = int(split_info["val_end"])
    total_len = int(split_info["total_len"])

    # History window length uniformly uses the global SEQ_LEN, comparable to deep learning models, no longer dynamically computed.
    lag_order = SEQ_LEN

    # Read this model's alpha candidate grid (automatically selected by RidgeCV via internal cross-validation).
    ridge_cfg = MODEL_SPECIFIC_CONFIG.get({})
    alphas = ridge_cfg.get("alphas", None)

    # Call the utility function to execute the rolling forecast.
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
    """
    Run the full experiment grid for a single model.

    Grid dimensions:
        prediction horizon x missing mechanism x missing rate x imputation method
    """
    # Define a separate results file per model so multiple-GPU runs do not overwrite each other.
    results_file = os.path.join(BASE_DIR, f"results_refactor_{model_name.lower()}.csv")
    # Define a separate checkpoint file per model to support resume-after-interruption.
    checkpoint_file = os.path.join(
        BASE_DIR, f"checkpoint_refactor_{model_name.lower()}.json"
    )

    # Load and preprocess data.
    (
        train_data_full,
        val_data_full,
        test_data_full,
        full_clean_data,
        value_cols,
        split_info,
    ) = load_and_preprocess_data()

    # Load historical state.
    results, completed_keys = _load_state(
        checkpoint_file=checkpoint_file, results_file=results_file
    )

    # Build the experiment grid.
    experiments = build_experiments()

    # Print the execution overview.
    print(f"Model: {model_name}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Results file: {results_file}")
    print(f"Checkpoint file: {checkpoint_file}")

    # Iterate over all experiment combinations.
    for exp_idx, exp in enumerate(experiments, 1):
        # Read the current prediction length.
        pred_len = exp["pred_len"]
        # Read the current missing mechanism.
        missing_mode = exp["missing_mode"]
        # Read the current missing rate.
        missing_rate = exp["missing_rate"]
        # Read the current imputation method.
        impute_method = exp["impute_method"]

        # Generate the unique experiment key, used for resume-after-interruption checks.
        exp_key = f"{model_name}_{missing_mode}_{impute_method}_{pred_len}_{int(missing_rate * 100)}"

        # Skip if the experiment is already completed.
        if exp_key in completed_keys:
            print(f"[{exp_idx}/{len(experiments)}] Skipping completed: {exp_key}")
            continue

        # Print the current experiment header.
        print("\n" + "=" * 90)
        print(
            f"[{exp_idx}/{len(experiments)}] model={model_name}, missing_mode={missing_mode}, "
            f"imputation={impute_method}, pred_len={pred_len}, missing_rate={int(missing_rate * 100)}%"
        )
        print("=" * 90)

        # Record the experiment start time.
        start_time = time.time()

        try:
            # ----------------------------------------------------------
            # Cache-first imputation flow: reuse the cache first, and compute+write on cache miss.
            # Note: the cache key does not include pred_len, because imputation depends only on the missing mechanism / method / missing rate.
            # ----------------------------------------------------------
            cache_dir = os.path.join(BASE_DIR, "npz_file")
            os.makedirs(cache_dir, exist_ok=True)
            cache_filename = f"impute_v2_{missing_mode}_{impute_method}_rate{int(missing_rate*100)}.npz"
            cache_path = os.path.join(cache_dir, cache_filename)

            if os.path.exists(cache_path):
                data = np.load(cache_path)
                full_imputed_data = data["full_data"]
                ie_mae = float(data["ie_mae"])
                ie_mse = float(data["ie_mse"])
                print(f"⚡ Fast-loading imputation cache: {cache_path}")
            else:
                # Inject missing data and impute only on the training segment to construct the training input.
                train_imputed, _, _, fitted_imputer = inject_and_impute(
                    train_clean=train_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    verbose=False,
                    fitted_imputer=None,
                )

                # The SAITS/BRITS models are bound to n_steps, so a fitted imputer cannot be reused across segments of different lengths.
                reuse = None if impute_method in ("saits", "brits") else fitted_imputer

                # Inject missing data and impute on the validation segment to avoid validation leakage (input distribution matches training).
                val_imputed, _, _, fitted_imputer = inject_and_impute(
                    train_clean=val_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    verbose=False,
                    fitted_imputer=reuse,
                )

                # Handle the test segment similarly.
                reuse = None if impute_method in ("saits", "brits") else fitted_imputer

                # Inject missing data and impute on the test segment; IE is only computed on the test segment.
                test_imputed, ie_mae, ie_mse, fitted_imputer = inject_and_impute(
                    train_clean=test_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    verbose=True,
                    fitted_imputer=reuse,
                )

                # Reconstruct the full input by 7:1:2: training segment imputed, validation segment imputed, test segment imputed.
                full_imputed_data = np.vstack(
                    (train_imputed, val_imputed, test_imputed)
                )

                # Persist the full imputed data and IE metrics so subsequent models can reuse them directly.
                np.savez(
                    cache_path,
                    full_data=full_imputed_data,
                    ie_mae=ie_mae,
                    ie_mse=ie_mse,
                )
                print(f"💾 Imputation computed and cached: {cache_path}")

            # Branch 1: Ridge regression goes through the classic statistical evaluation flow (the original VAR branch replaced with Ridge).
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
            # Branch 2: all other models go through the deep learning sliding-window training/evaluation flow.
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

            # Attempt to release the CUDA cache after each round of evaluation.
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Compute the elapsed time for a single experiment.
            elapsed_sec = round(time.time() - start_time, 2)

            # Assemble the success record row.
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
                "best_val_loss": (
                    None if best_val_loss is None else round(float(best_val_loss), 6)
                ),
                "best_epoch": best_epoch,
                "best_model_path": best_model_path,
                "status": "success",
                "elapsed_sec": elapsed_sec,
            }

            # Append to the results list.
            results.append(result_row)
            # Mark the current experiment as completed.
            completed_keys.add(exp_key)

            # Print the success summary.
            print(
                f"✅ Done: {exp_key} | IE_MSE(Test)={ie_mse:.4f} | FE_MSE(Test)={fe_mse:.4f} | "
                f"BestVal={best_val_loss if best_val_loss is not None else 'N/A'} | elapsed={elapsed_sec:.2f}s"
            )

        except Exception as exc:
            # Compute the elapsed time for the failed experiment.
            elapsed_sec = round(time.time() - start_time, 2)
            # Truncate the error message to avoid overly long CSV content.
            error_msg = str(exc)[:200]

            # Print the failure information.
            print(f"❌ Failed: {exp_key} | error={error_msg}")
            # Print the full stack trace for debugging.
            traceback.print_exc()

            # Assemble the failure record row.
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

            # Append the failure record.
            results.append(result_row)
            # Mark the experiment as processed even on failure to prevent infinite retries of the same error.
            completed_keys.add(exp_key)

        # Save after each experiment so resumption after interruption is always possible.
        _save_state(
            checkpoint_file=checkpoint_file,
            results_file=results_file,
            results=results,
            completed_keys=completed_keys,
        )

        # Trigger garbage collection to reduce memory usage during long runs.
        gc.collect()

    # Print the model completion prompt.
    print("\n" + "=" * 90)
    print(f"All experiments for model {model_name} completed")
    print(f"Results written to: {results_file}")
    print("=" * 90)


# =======================
# 11. Overall scheduling entry (optional)
# =======================


def run_all_models(model_list=None):
    """Run multiple models in sequence to enable single-machine serial scheduling."""
    # If no model list is explicitly passed, use the default set of 5 models.
    if model_list is None:
        model_list = ["LSTM", "DLinear", "PatchTST"]

    # Run each model in turn.
    for model_name in model_list:
        run_single_model(model_name)
