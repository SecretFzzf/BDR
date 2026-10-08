# -*- coding: utf-8 -*-

# 导入系统标准库，用于路径、时间、断点、异常和垃圾回收管理。
import os
import sys
import json
import argparse
import time
import gc
import traceback
import warnings
from pathlib import Path

# 导入数值计算与表格处理库。
import numpy as np
import pandas as pd

# 导入标准化与误差指标。
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 导入 PyTorch 生态用于深度学习训练。
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 统一硬件设备检测：优先 CUDA，其次 MPS，最后 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# 对固定输入形状任务启用 cuDNN 自动调优，提升 3090 吞吐。
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 统一关闭警告，减少日志噪音。
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# 获取当前脚本目录，确保相对路径稳定。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "training"
CHECKPOINT_DIR = REPO_ROOT / ".cache" / "checkpoints"

# 将当前目录加入模块搜索路径，便于导入同目录 tool.py。
sys.path.insert(0, BASE_DIR)

from tool import (
    inject_mcar_missing_and_impute,
    inject_mar_block_missing_and_impute,
    inject_mnar_value_missing_and_impute,
)


def masked_mse_loss(pred, target):
    """NaN感知MSE损失：target中NaN位置不参与loss计算。"""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return (pred * 0.0).sum()
    diff = pred[mask] - target[mask]
    return (diff ** 2).mean()


class MaskedErrorAccumulator:
    """跨batch累加有效误差，最后统一相除，避免不同batch NaN比例不同导致加权偏差。"""
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
# 一、全局配置
# =============================

# Weather 主数据文件路径。
DATA_FILE = DEFAULT_DATA_DIR / "weather.csv"

def _resolve_data_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    args, _ = parser.parse_known_args()
    return args.data_dir / DATA_FILE.name

# 缺失机制集合（3 种）。
MISSING_MODES = ["MCAR", "MAR_Block", "MNAR"]

# 插补方法集合（5 种）。
IMPUTE_METHODS = ["mean", "spline", "knn", "brits", "saits"]

# 预测步长集合。
PRED_LEN_LIST = [24, 48, 96, 192]

# 缺失率集合（4 档）。
MISSING_RATES = [0.10, 0.30, 0.50, 0.70]

# 固定训练/验证/测试划分比例（连续序列 7:1:2）。
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
}

# 默认输入历史窗口长度（从分层配置读取）。
SEQ_LEN = int(DEFAULT_DL_CONFIG["seq_len"])

STRIDE = 24

# 随机种子配置。
SEED = 42


# =================================
# 二、数据集定义（连续滑窗）
# =================================

class ContinuousTimeSeriesDataset(Dataset):
    """连续时间序列滑窗数据集（非分段模式）。"""

    def __init__(self, x_array, y_array, seq_len, pred_len, stride=1):
        # 保存输入特征矩阵。
        self.x_array = x_array
        # 保存监督目标矩阵。
        self.y_array = y_array
        # 保存历史窗口长度。
        self.seq_len = seq_len
        # 保存预测窗口长度。
        self.pred_len = pred_len
        # 保存滑窗步长。
        self.stride = stride
        # 构造所有合法起点索引，确保窗口完整。
        self.indices = np.arange(0, len(x_array) - seq_len - pred_len + 1, stride)

    def __len__(self):
        # 返回总样本数。
        return len(self.indices)

    def __getitem__(self, idx):
        # 取当前样本起点。
        start = self.indices[idx]
        # 切分输入窗口。
        x_window = self.x_array[start:start + self.seq_len]
        # 切分标签窗口。
        y_window = self.y_array[start + self.seq_len:start + self.seq_len + self.pred_len]
        # 转换为 float32 Tensor。
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_window, dtype=torch.float32)


class METRLATemporalSplitDataset(Dataset):
    """METR-LA 严格时间切分数据集（支持 7:1:2 分段窗口抽取）。"""

    def __init__(self, x_array, y_array, split_start, split_end, seq_len, pred_len, stride=1):
        # 保存输入特征数组。
        self.x_array = x_array
        # 保存监督目标数组。
        self.y_array = y_array
        # 保存切分起始位置。
        self.split_start = int(split_start)
        # 保存切分结束位置。
        self.split_end = int(split_end)
        # 保存历史窗口长度。
        self.seq_len = int(seq_len)
        # 保存预测窗口长度。
        self.pred_len = int(pred_len)
        # 保存滑窗步长。
        self.stride = int(stride)

        # 校验切分边界合法性。
        if self.split_start < 0 or self.split_end > len(x_array) or self.split_start >= self.split_end:
            raise ValueError("非法时间切分区间，请检查 split_start/split_end。")

        # 计算允许的最小窗口起点。
        min_start = max(0, self.split_start - self.seq_len)
        # 计算允许的最大窗口起点。
        max_start = self.split_end - self.seq_len - self.pred_len

        # 收集所有合法窗口起点。
        indices = []
        if max_start >= min_start:
            for start in range(min_start, max_start + 1, self.stride):
                # 计算标签窗口起点。
                y_start = start + self.seq_len
                # 计算标签窗口终点。
                y_end = y_start + self.pred_len
                # 仅保留标签完全落在当前切分段内的窗口。
                if y_start >= self.split_start and y_end <= self.split_end:
                    indices.append(start)

        # 保存窗口起点索引数组。
        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self):
        # 返回可用窗口数量。
        return len(self.indices)

    def __getitem__(self, idx):
        # 取得当前样本对应的起点位置。
        start = int(self.indices[idx])
        # 切出输入历史窗口。
        x_window = self.x_array[start:start + self.seq_len]
        # 切出未来标签窗口。
        y_window = self.y_array[start + self.seq_len:start + self.seq_len + self.pred_len]
        # 返回 float32 张量。
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_window, dtype=torch.float32)


# =========================
# 三、模型定义（统一维护）
# =========================

class LSTMForecaster(nn.Module):
    """LSTM 多变量预测器。"""

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




class DLinearForecaster(nn.Module):
    """DLinear 预测器（按通道独立线性映射）。"""

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
    """简化版 PatchTST 预测器。"""

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
# 四、数据加载与预处理
# =====================

def load_and_preprocess_data():
    """加载 Weather 数据并执行严格防泄露标准化。"""
    # 读取 weather.csv。
    candidate_files = [
        DATA_FILE,
        os.path.join(BASE_DIR, "weather.csv"),
    ]
    existing_files = [p for p in candidate_files if os.path.exists(p)]
    if not existing_files:
        raise FileNotFoundError(f"找不到可用数据文件，请检查: {candidate_files}")
    data_file = existing_files[0]

    # 读取 CSV 数据。
    df = pd.read_csv(data_file)
    # 仅保留数值列（自动排除时间戳/字符串索引列）。
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    # 无数值列则报错。
    if numeric_df.shape[1] == 0:
        raise ValueError(f"{os.path.basename(data_file)} 中没有可用的数值列。")

    # METR-LA 零值=传感器离线（0 mph 不是真实行驶速度），视为结构性缺失，替换为 NaN 后插值填充。
    numeric_df = numeric_df.replace(0, np.nan)
    numeric_df = numeric_df.interpolate(limit_direction='both').ffill().bfill()

    # 获取原始列名与数值矩阵。
    value_cols = list(numeric_df.columns)
    data = numeric_df.values.astype(np.float64)

    # 统计总长度。
    total_len = len(data)
    # 计算 7:1:2 时间切分点。
    train_end = int(total_len * TRAIN_RATIO)
    val_end = train_end + int(total_len * VAL_RATIO)

    # 防御式修正：避免切分点越界。
    train_end = max(1, min(train_end, total_len - 2))
    val_end = max(train_end + 1, min(val_end, total_len - 1))

    # 使用训练段拟合标准化器，防止数据泄露。
    scaler = StandardScaler()
    scaler.fit(data[:train_end])

    # 全量数据统一变换（仅使用训练段统计量）。
    full_clean_data = scaler.transform(data)

    # 按时间切分出三段。
    train_data_full = full_clean_data[:train_end]
    val_data_full = full_clean_data[train_end:val_end]
    test_data_full = full_clean_data[val_end:]

    # 打印数据摘要。
    print("=" * 80)
    print("METR-LA 数据加载完成")
    print(f"数据文件: {os.path.basename(data_file)}")
    print(f"总长度: {total_len}")
    print(f"训练长度: {len(train_data_full)}")
    print(f"验证长度: {len(val_data_full)}")
    print(f"测试长度: {len(test_data_full)}")
    print(f"有效变量数: {len(value_cols)}")
    print("=" * 80)

    # 构建时间切分信息字典。
    split_info = {
        "train_end": int(train_end),
        "val_end": int(val_end),
        "total_len": int(total_len),
    }

    # 返回预处理结果。
    return train_data_full, val_data_full, test_data_full, full_clean_data, value_cols, split_info


def _run_eval_loss(model, data_loader, criterion=None):
    """在给定数据集上用 MaskedErrorAccumulator 计算平均 MSE loss。"""
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
# 五、缺失注入路由函数
# ====================

def inject_and_impute(train_clean, missing_mode, missing_rate, impute_method, verbose=True, fitted_imputer=None):
    """按缺失机制路由到对应注入+插补函数，并返回 IE 指标。"""
    if missing_mode == "MCAR":
        return inject_mcar_missing_and_impute(train_clean, missing_rate, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)
    if missing_mode == "MAR_Block":
        return inject_mar_block_missing_and_impute(train_clean, missing_rate, 24, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)
    if missing_mode == "MNAR":
        return inject_mnar_value_missing_and_impute(train_clean, missing_rate, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)
    raise ValueError(f"未知缺失模式: {missing_mode}")


# =====================
# 六、实验状态持久化工具
# =====================

def _load_state(checkpoint_file, results_file):
    """加载断点与历史结果。"""
    results = []
    completed_keys = set()

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            completed_keys = set(json.load(f))

    if os.path.exists(results_file):
        results = pd.read_csv(results_file).to_dict("records")

    return results, completed_keys


def _save_state(checkpoint_file, results_file, results, completed_keys):
    """保存断点与结果。"""
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, ensure_ascii=False)

    if results:
        pd.DataFrame(results).to_csv(results_file, index=False)


# =====================
# 七、实验组合与模型构建
# =====================

def build_experiments():
    """构建实验网格并按插补方法复杂度排序。"""
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
    """根据模型名构建对应网络。"""
    if model_name == "LSTM":
        return LSTMForecaster(num_features=num_features, hidden_size=64, num_layers=2, pred_len=pred_len)
    if model_name == "DLinear":
        return DLinearForecaster(seq_len=seq_len, pred_len=pred_len, enc_in=num_features)
    if model_name == "PatchTST":
        return PatchTSTForecaster(seq_len=seq_len, pred_len=pred_len, enc_in=num_features)
    raise ValueError(f"未知模型名: {model_name}")


# ===========================
# 八、深度学习模型训练与评估
# ===========================

def train_and_eval_dl_model(model_name, full_imputed_data, full_clean_data, split_info, pred_len, missing_mode, missing_rate):
    """训练并评估单个深度学习模型，返回 FE 指标和窗口统计。"""
    # 合并默认配置与模型特定配置。
    model_cfg = {**DEFAULT_DL_CONFIG, **MODEL_SPECIFIC_CONFIG.get(model_name, {})}
    # 读取当前模型批大小。
    batch_size = int(model_cfg["batch_size"])
    # 读取当前模型学习率。
    learning_rate = float(model_cfg["learning_rate"])
    # 读取当前模型最大 epoch。
    max_epochs = int(model_cfg["max_epochs"])
    # 读取当前模型早停耐心轮数。
    patience = int(model_cfg["patience"])
    # 读取当前模型最小提升阈值。
    min_delta = float(model_cfg["min_delta"])
    # 读取当前模型序列长度。
    seq_len = int(model_cfg["seq_len"])
    # 读取 DataLoader 工作进程数。
    num_workers = int(model_cfg.get("num_workers", 12))

    # 读取严格时间切分点。
    train_end = int(split_info["train_end"])
    val_end = int(split_info["val_end"])
    total_len = int(split_info["total_len"])

    # 在完整序列上按时间边界抽取 train/val/test 窗口。
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
        raise ValueError("窗口数量为 0，请检查 seq_len/pred_len 或 7:1:2 时间切分边界。")

    # CUDA 下启用 pin_memory，加速 CPU 到 GPU 拷贝。
    use_pin_memory = device.type == "cuda"
    # 多线程加载时启用持久 worker，降低每轮重建开销。
    persistent_workers = bool(num_workers > 0)

    # 构建训练/验证/测试 DataLoader。
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

    # 创建并迁移模型。
    model = build_model(model_name=model_name, num_features=full_imputed_data.shape[1], pred_len=pred_len, seq_len=seq_len)
    model = model.to(device)

    # 创建优化器和损失函数。
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = masked_mse_loss

    # 为当前实验生成唯一权重文件名，避免并行实验互相覆盖。
    missing_mode_tag = str(missing_mode).replace("/", "_")
    weight_path = os.path.join(
        BASE_DIR,
        f"best_model_{model_name}_{missing_mode_tag}_{int(round(float(missing_rate) * 100))}_{pred_len}.pth",
    )
    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    try:
        # 训练多个 epoch 并执行验证。
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
                        f"   [{model_name}] EarlyStopping 触发: "
                        f"连续 {patience} 轮验证集未提升。"
                    )
                    break

        # 回载最佳权重进行测试。
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))

        # 执行测试评估。
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
# =========================




# =======================
# 十、单模型统一执行入口
# =======================

def run_single_model(model_name):
    global DATA_FILE
    DATA_FILE = _resolve_data_file()
    """执行单个模型在全实验网格上的训练与评估。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / Path(BASE_DIR).name / f"results_refactor_{model_name.lower()}.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / Path(BASE_DIR).name / f"checkpoint_refactor_{model_name.lower()}.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    train_data_full, val_data_full, test_data_full, full_clean_data, value_cols, split_info = load_and_preprocess_data()

    results, completed_keys = _load_state(checkpoint_file=checkpoint_file, results_file=results_file)
    experiments = build_experiments()

    print(f"模型: {model_name}")
    print(f"实验总数: {len(experiments)}")
    print(f"结果文件: {results_file}")
    print(f"断点文件: {checkpoint_file}")

    for exp_idx, exp in enumerate(experiments, 1):
        pred_len = exp["pred_len"]
        missing_mode = exp["missing_mode"]
        missing_rate = exp["missing_rate"]
        impute_method = exp["impute_method"]

        exp_key = f"{model_name}_{missing_mode}_{impute_method}_{pred_len}_{int(missing_rate * 100)}"

        if exp_key in completed_keys:
            print(f"[{exp_idx}/{len(experiments)}] 跳过已完成: {exp_key}")
            continue

        print("\n" + "=" * 90)
        print(
            f"[{exp_idx}/{len(experiments)}] 模型={model_name}, 缺失机制={missing_mode}, "
            f"插补={impute_method}, pred_len={pred_len}, 缺失率={int(missing_rate * 100)}%"
        )
        print("=" * 90)

        start_time = time.time()

        try:
            # ----------------------------------------------------------
            # Cache-First 插补流程：先复用缓存，未命中时再计算并写入缓存。
            # 说明：缓存键不含 pred_len，因为插补仅依赖缺失机制/方法/缺失率。
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
                print(f"⚡ 快速加载插补缓存: {cache_path}")
            else:
                # 仅在训练段注入缺失并插补，构造训练输入。
                train_imputed, _, _, fitted_imputer = inject_and_impute(
                    train_clean=train_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    verbose=False,
                    fitted_imputer=None,
                )

                # SAITS/BRITS 的模型与 n_steps 绑定，不同长度段不能复用 fitted。
                reuse = None if impute_method in ("saits", "brits") else fitted_imputer

                # 在验证段注入缺失并插补，避免验证泄漏（输入与训练分布一致）。
                val_imputed, _, _, fitted_imputer = inject_and_impute(
                    train_clean=val_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    verbose=False,
                    fitted_imputer=reuse,
                )

                reuse = None if impute_method in ("saits", "brits") else fitted_imputer

                # 在测试段注入缺失并插补，IE 只在测试段统计。
                test_imputed, ie_mae, ie_mse, fitted_imputer = inject_and_impute(
                    train_clean=test_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    verbose=True,
                    fitted_imputer=reuse,
                )

                # 按 7:1:2 重建完整输入：训练段插补，验证段插补，测试段插补。
                full_imputed_data = np.vstack((train_imputed, val_imputed, test_imputed))

                # 将整段插补数据和 IE 指标持久化，供后续模型直接复用。
                np.savez(cache_path, full_data=full_imputed_data, ie_mae=ie_mae, ie_mse=ie_mse)
                print(f"💾 插补已计算并缓存: {cache_path}")


            # 每轮评估后尝试释放 CUDA 缓存。
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
                f"✅ 完成: {exp_key} | IE_MSE(Test)={ie_mse:.4f} | FE_MSE(Test)={fe_mse:.4f} | "
                f"BestVal={best_val_loss if best_val_loss is not None else 'N/A'} | 耗时={elapsed_sec:.2f}s"
            )

        except Exception as exc:
            elapsed_sec = round(time.time() - start_time, 2)
            error_msg = str(exc)[:200]

            print(f"❌ 失败: {exp_key} | error={error_msg}")
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
    print(f"模型 {model_name} 全部实验完成")
    print(f"结果已写入: {results_file}")
    print("=" * 90)


# =======================
# 十一、总调度入口（可选）
# =======================

def run_all_models(model_list=None):
    """顺序运行多个模型，默认执行五个基线模型。"""
    if model_list is None:
        model_list = ["LSTM", "DLinear", "PatchTST"]

    for model_name in model_list:
        run_single_model(model_name)
