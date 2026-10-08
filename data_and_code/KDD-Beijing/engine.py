# 导入系统标准库，用于路径、断点、计时、异常追踪和垃圾回收。
import os
import sys
import json
import argparse
import time
import gc
import traceback
import warnings
from pathlib import Path

# 导入数值和表格处理库。
import numpy as np
import pandas as pd

# 导入标准化与误差评估指标。
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 导入 PyTorch 生态组件，用于深度学习训练。
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 统一硬件设备检测：优先 CUDA，其次 MPS，最后 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# 对固定输入形状任务启用 cuDNN 自动调优，降低 GPU 空转。
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 抑制无关警告，保持日志简洁。
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 获取当前脚本所在目录。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "training"
CHECKPOINT_DIR = REPO_ROOT / ".cache" / "checkpoints"

# 把当前目录加入模块搜索路径，确保可导入同目录模块。
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
# 一、全局配置（保持原 AQI-36 逻辑）
# =============================

# KDD-Beijing 数据文件路径（CSV 格式）。
DATA_FILE = DEFAULT_DATA_DIR / "kdd_beijing_raw.csv"

def _resolve_data_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    args, _ = parser.parse_known_args()
    return args.data_dir / DATA_FILE.name

# 缺失机制集合（3 种）。
MISSING_MODES = ["MCAR", "MAR_Block", "MNAR"]

# 插补方法集合（5 种，与其他数据集对齐）。
IMPUTE_METHODS = ["mean", "spline", "knn", "brits", "saits"]

# 预测步长集合（保持原 AQI-36 逻辑）。
PRED_LEN_LIST = [24, 48, 96, 192]

# 缺失率集合（4 档）。
MISSING_RATES = [0.10, 0.30, 0.50, 0.70]

# 训练/验证/测试划分比例（7:1:2，严格按时间先后切分）。
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
}

# 默认输入历史窗口长度（从分层配置读取）。
SEQ_LEN = int(DEFAULT_DL_CONFIG["seq_len"])

# 滚动评估步长（每天滑动一次）。
STRIDE = 24

# 随机种子。
SEED = 42


# =================================
# 二、数据集定义（连续滑窗）
# =================================

class ContinuousTimeSeriesDataset(Dataset):
    """连续时间序列滑窗数据集（不涉及 stay_id 分段）。"""

    def __init__(self, x_array, y_array, seq_len, pred_len, stride=1):
        # 预先转为连续 float32 张量，避免 __getitem__ 里重复构造张量造成 CPU 开销。
        self.x_tensor = torch.from_numpy(np.ascontiguousarray(x_array, dtype=np.float32))
        self.y_tensor = torch.from_numpy(np.ascontiguousarray(y_array, dtype=np.float32))
        # 保存历史窗口长度。
        self.seq_len = seq_len
        # 保存预测窗口长度。
        self.pred_len = pred_len
        # 保存滑窗步长。
        self.stride = stride
        # 构造全部合法窗口起点。
        self.indices = np.arange(0, len(x_array) - seq_len - pred_len + 1, stride)

    def __len__(self):
        # 返回样本数量。
        return len(self.indices)

    def __getitem__(self, idx):
        # 取当前样本起点。
        start = self.indices[idx]
        # 切分输入窗口。
        x_window = self.x_tensor[start:start + self.seq_len]
        # 切分标签窗口。
        y_window = self.y_tensor[start + self.seq_len:start + self.seq_len + self.pred_len]
        # 返回预构造张量切片。
        return x_window, y_window


class KDDBeijingTemporalSplitDataset(Dataset):
    """KDD Beijing 严格时间切分数据集（支持 7:1:2 的分段窗口抽取）。"""

    def __init__(self, x_array, y_array, split_start, split_end, seq_len, pred_len, stride=1):
        # 预先转为连续 float32 张量，降低 DataLoader 单样本转换成本。
        self.x_tensor = torch.from_numpy(np.ascontiguousarray(x_array, dtype=np.float32))
        self.y_tensor = torch.from_numpy(np.ascontiguousarray(y_array, dtype=np.float32))
        self.split_start = int(split_start)
        self.split_end = int(split_end)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)

        if self.split_start < 0 or self.split_end > len(x_array) or self.split_start >= self.split_end:
            raise ValueError("非法时间切分区间，请检查 split_start/split_end。")

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
# 三、模型定义（统一维护）
# =========================

class LSTMForecaster(nn.Module):
    """LSTM 多变量预测器。"""

    def __init__(self, num_features, hidden_size, num_layers, pred_len):
        # 调用父类构造函数。
        super().__init__()
        # 保存特征维度。
        self.num_features = num_features
        # 保存预测步长。
        self.pred_len = pred_len
        # 定义 LSTM 主干。
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # 定义输出映射层。
        self.proj = nn.Linear(hidden_size, pred_len * num_features)

    def forward(self, x):
        # 执行 LSTM 前向传播。
        out, _ = self.lstm(x)
        # 取最后时刻隐状态。
        last_hidden = out[:, -1, :]
        # 线性映射到预测向量。
        pred_flat = self.proj(last_hidden)
        # 重塑到 [B, pred_len, C]。
        pred = pred_flat.view(-1, self.pred_len, self.num_features)
        # 返回预测结果。
        return pred




class DLinearForecaster(nn.Module):
    """DLinear 预测器（逐通道线性映射）。"""

    def __init__(self, seq_len, pred_len, enc_in):
        # 调用父类构造函数。
        super().__init__()
        # 保存输入长度。
        self.seq_len = seq_len
        # 保存预测长度。
        self.pred_len = pred_len
        # 保存特征通道数。
        self.enc_in = enc_in
        # 每个通道独立一个线性头。
        self.linears = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(enc_in)])

    def forward(self, x):
        # 初始化输出列表。
        outputs = []
        # 逐通道计算预测。
        for i in range(self.enc_in):
            # 取单通道序列。
            x_i = x[:, :, i]
            # 线性映射到预测长度。
            y_i = self.linears[i](x_i)
            # 保存单通道结果。
            outputs.append(y_i)
        # 合并所有通道输出。
        pred = torch.stack(outputs, dim=2)
        # 返回预测结果。
        return pred


class PatchTSTForecaster(nn.Module):
    """简化版 PatchTST 预测器。"""

    def __init__(self, seq_len, pred_len, enc_in, patch_len=16, stride=8, e_layers=2, d_model=128):
        # 调用父类构造函数。
        super().__init__()
        # 保存输入长度。
        self.seq_len = seq_len
        # 保存预测长度。
        self.pred_len = pred_len
        # 保存通道数。
        self.enc_in = enc_in
        # 保存 patch 长度。
        self.patch_len = patch_len
        # 保存 patch 步长。
        self.stride = stride

        # 计算 patch 数量。
        self.n_patches = (seq_len - patch_len) // stride + 1

        # 每个通道独立 patch 嵌入层。
        self.patch_embedding = nn.ModuleList([nn.Linear(patch_len, d_model) for _ in range(enc_in)])

        # 定义 Transformer 编码器层。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )

        # 叠加 Transformer 编码层。
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # 每个通道独立预测头。
        self.pred_heads = nn.ModuleList([
            nn.Linear(d_model * self.n_patches, pred_len) for _ in range(enc_in)
        ])

    def forward(self, x):
        # 获取输入批次大小与通道数。
        batch_size, _, n_features = x.shape
        # 初始化多通道输出列表。
        outputs = []

        # 逐通道做 patch 建模与预测。
        for i in range(n_features):
            # 取第 i 个通道的时间序列。
            x_i = x[:, :, i]
            # 初始化 patch 列表。
            patches = []
            # 循环切分 patch。
            for j in range(self.n_patches):
                # 计算当前 patch 起点。
                start = j * self.stride
                # 计算当前 patch 终点。
                end = start + self.patch_len
                # 截取 patch。
                patch = x_i[:, start:end]
                # 添加到列表。
                patches.append(patch)

            # 堆叠 patch 张量。
            patches = torch.stack(patches, dim=1)
            # 执行 patch 嵌入。
            emb = self.patch_embedding[i](patches)
            # 执行 Transformer 编码。
            enc = self.encoder(emb)
            # 展平编码输出。
            enc_flat = enc.reshape(batch_size, -1)
            # 预测未来序列。
            y_i = self.pred_heads[i](enc_flat)
            # 保存当前通道预测。
            outputs.append(y_i)

        # 合并全部通道输出。
        pred = torch.stack(outputs, dim=2)
        # 返回预测结果。
        return pred


# =====================
# 四、数据加载与预处理
# =====================




def load_and_preprocess_data():
    """
    加载 AQI-36 数据并执行严格防泄露标准化。

    返回：
        train_data_full: 标准化训练段 [train_T, C]
        val_data_full: 标准化验证段 [val_T, C]
        test_data_full: 标准化测试段 [test_T, C]
        full_clean_data: 标准化完整序列 [T, C]
        value_cols: 有效特征列名列表
        split_info: 时间切分索引信息
    """
    # 校验数据文件存在性。
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"找不到数据文件: {DATA_FILE}")

    # 直接读取 KDD_Beijing_Clean.csv（不再走 TSF 解析逻辑）。
    df = pd.read_csv(DATA_FILE)
    if df.empty:
        raise ValueError("CSV 文件为空，无法继续训练。")

    # 仅保留数值列，自动剔除时间戳等非数值字段。
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] == 0:
        raise ValueError("CSV 中没有可用的数值列。")

    # 获取原始列名与数值矩阵。
    raw_value_cols = list(numeric_df.columns)
    raw_data = numeric_df.values.astype(np.float64)

    # 计算每一列方差。
    variances = np.nanvar(raw_data, axis=0)
    # 过滤常数列。
    non_constant_mask = variances > 1e-5

    # 应用有效列掩码。
    data = raw_data[:, non_constant_mask]
    # 生成有效列名列表。
    value_cols = [raw_value_cols[i] for i in range(len(raw_value_cols)) if non_constant_mask[i]]

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
    print("AQI-36 数据加载完成")
    print(f"总长度: {total_len}")
    print(f"训练长度: {len(train_data_full)}")
    print(f"验证长度: {len(val_data_full)}")
    print(f"测试长度: {len(test_data_full)}")
    print(f"原始变量数: {len(raw_value_cols)}")
    print(f"有效变量数: {len(value_cols)}")
    print("=" * 80)

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
    # MCAR 随机缺失分支。
    if missing_mode == "MCAR":
        return inject_mcar_missing_and_impute(train_clean, missing_rate, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)

    # MAR 块状缺失分支。
    if missing_mode == "MAR_Block":
        return inject_mar_block_missing_and_impute(train_clean, missing_rate, 24, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)

    # MNAR 值相关缺失分支。
    if missing_mode == "MNAR":
        return inject_mnar_value_missing_and_impute(train_clean, missing_rate, impute_method, SEED, verbose=verbose, fitted_imputer=fitted_imputer)

    # 未知机制直接报错。
    raise ValueError(f"未知缺失模式: {missing_mode}")


# =====================
# 六、实验状态持久化工具
# =====================

def _load_state(checkpoint_file, results_file):
    """加载断点文件与历史结果文件。"""
    # 初始化结果列表。
    results = []
    # 初始化完成键集合。
    completed_keys = set()

    # 若断点文件存在则读取。
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            completed_keys = set(json.load(f))

    # 若结果文件存在则读取。
    if os.path.exists(results_file):
        results = pd.read_csv(results_file).to_dict("records")

    # 返回状态。
    return results, completed_keys


def _save_state(checkpoint_file, results_file, results, completed_keys):
    """保存断点和结果到磁盘。"""
    # 写入断点键集合。
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, ensure_ascii=False)

    # 若存在结果则写出 CSV。
    if results:
        pd.DataFrame(results).to_csv(results_file, index=False)


# =====================
# 七、实验组合与模型构建
# =====================

def build_experiments():
    """构建全实验网格并按插补复杂度排序。"""
    # 初始化实验列表。
    experiments = []

    # 依次遍历预测步长。
    for pred_len in PRED_LEN_LIST:
        # 依次遍历缺失机制。
        for missing_mode in MISSING_MODES:
            # 依次遍历缺失率。
            for missing_rate in MISSING_RATES:
                # 依次遍历插补方法。
                for impute_method in IMPUTE_METHODS:
                    # 添加当前组合。
                    experiments.append(
                        {
                            "pred_len": pred_len,
                            "missing_mode": missing_mode,
                            "missing_rate": missing_rate,
                            "impute_method": impute_method,
                        }
                    )

    # 定义插补复杂度优先级。
    method_order = {"mean": 0, "spline": 1, "knn": 2, "brits": 3, "saits": 4}
    # 执行排序。
    experiments.sort(key=lambda x: (method_order[x["impute_method"]], x["pred_len"]))

    # 返回实验列表。
    return experiments


def build_model(model_name, num_features, pred_len, seq_len=SEQ_LEN):
    """根据模型名称创建深度学习模型实例。"""
    # LSTM 分支。
    if model_name == "LSTM":
        return LSTMForecaster(num_features=num_features, hidden_size=64, num_layers=2, pred_len=pred_len)
    # DLinear 分支。
    if model_name == "DLinear":
        return DLinearForecaster(seq_len=seq_len, pred_len=pred_len, enc_in=num_features)
    # PatchTST 分支。
    if model_name == "PatchTST":
        return PatchTSTForecaster(seq_len=seq_len, pred_len=pred_len, enc_in=num_features)
    # 兜底错误分支。
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

    # 读取严格时间切分点。
    train_end = int(split_info["train_end"])
    val_end = int(split_info["val_end"])
    total_len = int(split_info["total_len"])

    # 在完整序列上按时间边界抽取 train/val/test 窗口。
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

    # 防御式检查窗口数量。
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("窗口数量为 0，请检查 seq_len/pred_len 或 7:1:2 时间切分边界。")

    # CUDA 下启用 pin_memory，加速 CPU 到 GPU 的数据拷贝。
    use_pin_memory = device.type == "cuda"

    # CUDA 训练优先使用多进程预取，降低主进程 CPU 成为瓶颈的概率。
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

    # 构建训练数据加载器。
    train_loader = DataLoader(train_dataset, **{**loader_kwargs, "shuffle": True})
    # 构建验证数据加载器。
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    # 构建测试数据加载器。
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    print(
        f"   [{model_name}] device={device.type}, batch_size={batch_size}, "
        f"num_workers={num_workers}, pin_memory={use_pin_memory}"
    )

    # 创建模型实例。
    model = build_model(model_name=model_name, num_features=full_imputed_data.shape[1], pred_len=pred_len, seq_len=seq_len)
    # 将模型迁移到目标设备。
    model = model.to(device)

    # 创建 Adam 优化器。
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 创建 MSE 损失函数。
    criterion = masked_mse_loss
    # 为当前实验生成唯一权重文件名，避免并行实验互相覆盖。
    missing_mode_tag = str(missing_mode).replace("/", "_")
    weight_path = os.path.join(
        BASE_DIR,
        f"best_model_{model_name}_{missing_mode_tag}_{int(round(float(missing_rate) * 100))}_{pred_len}.pth",
    )
    # 初始化早停状态。
    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    try:
        # 迭代训练多个 epoch（每轮后执行完整验证）。
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
                        f"   [{model_name}] EarlyStopping 触发: "
                        f"连续 {patience} 轮验证集未提升。"
                    )
                    break

        # 加载验证阶段选出的最佳权重，并在测试集计算 FE。
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))

        # 进入评估模式。
        model.eval()
        # 初始化预测容器。
        preds = []
        # 初始化真值容器。
        trues = []

        # 禁用梯度计算。
        with torch.no_grad():
            # 遍历测试批次。
            for batch_x, batch_y in test_loader:
                # 迁移输入张量。
                batch_x = batch_x.to(device, non_blocking=use_pin_memory)
                # 前向预测并转 numpy。
                batch_pred = model(batch_x).cpu().numpy()
                # 保存预测结果。
                preds.append(batch_pred)
                # 保存真值结果。
                trues.append(batch_y.numpy())

        # 拼接全部预测。
        preds = np.concatenate(preds, axis=0)
        # 拼接全部真值。
        trues = np.concatenate(trues, axis=0)

        # 按通道计算误差，再对通道求平均，避免大波动变量主导总指标。
        c = preds.shape[-1]
        trues_reshaped = trues.reshape(-1, c)
        preds_reshaped = preds.reshape(-1, c)
        per_channel_abs = np.nanmean(np.abs(trues_reshaped - preds_reshaped), axis=0)
        per_channel_sq = np.nanmean(np.square(trues_reshaped - preds_reshaped), axis=0)
        fe_mae = float(np.nanmean(per_channel_abs))
        fe_mse = float(np.nanmean(per_channel_sq))

        # 打印测试指标。
        print(f"   [{model_name} Test] FE_MAE={fe_mae:.4f}, FE_MSE={fe_mse:.4f}")

        # CUDA 评估后主动释放显存缓存，降低碎片化风险。
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 返回评估结果与窗口数量。
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
    # 构建该模型专属结果文件路径。
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / Path(BASE_DIR).name / f"results_refactor_{model_name.lower()}.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    # 构建该模型专属断点文件路径。
    checkpoint_file = CHECKPOINT_DIR / Path(BASE_DIR).name / f"checkpoint_refactor_{model_name.lower()}.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    # 加载并预处理数据。
    train_data_full, val_data_full, test_data_full, full_clean_data, value_cols, split_info = load_and_preprocess_data()

    # 加载历史状态。
    results, completed_keys = _load_state(checkpoint_file=checkpoint_file, results_file=results_file)

    # 构建实验网格。
    experiments = build_experiments()

    # 打印执行摘要。
    print(f"模型: {model_name}")
    print(f"实验总数: {len(experiments)}")
    print(f"结果文件: {results_file}")
    print(f"断点文件: {checkpoint_file}")

    # 遍历全部实验组合。
    for exp_idx, exp in enumerate(experiments, 1):
        # 读取预测长度。
        pred_len = exp["pred_len"]
        # 读取缺失机制。
        missing_mode = exp["missing_mode"]
        # 读取缺失率。
        missing_rate = exp["missing_rate"]
        # 读取插补方法。
        impute_method = exp["impute_method"]

        # 拼装唯一实验键。
        exp_key = f"{model_name}_{missing_mode}_{impute_method}_{pred_len}_{int(missing_rate * 100)}"

        # 已完成则跳过。
        if exp_key in completed_keys:
            print(f"[{exp_idx}/{len(experiments)}] 跳过已完成: {exp_key}")
            continue

        # 打印当前实验头。
        print("\n" + "=" * 90)
        print(
            f"[{exp_idx}/{len(experiments)}] 模型={model_name}, 缺失机制={missing_mode}, "
            f"插补={impute_method}, pred_len={pred_len}, 缺失率={int(missing_rate * 100)}%"
        )
        print("=" * 90)

        # 记录开始时间。
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
                # 仅在训练段注入缺失并插补，构造训练输入；同时获取 fitted imputer。
                train_imputed, _, _, fitted_imp = inject_and_impute(
                    train_clean=train_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                )

                # DL 插补方法（SAITS/BRITS）每个 split 独立训练，不复用 fitted imputer。
                reuse = None if impute_method in ("saits", "brits") else fitted_imp

                # 在验证段注入缺失并插补，复用训练段拟合的 imputer，避免数据泄漏。
                val_imputed, _, _, _ = inject_and_impute(
                    train_clean=val_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    fitted_imputer=reuse,
                )

                # 在测试段注入缺失并插补，复用训练段拟合的 imputer；IE 只在测试段统计。
                test_imputed, ie_mae, ie_mse, _ = inject_and_impute(
                    train_clean=test_data_full.copy(),
                    missing_mode=missing_mode,
                    missing_rate=missing_rate,
                    impute_method=impute_method,
                    fitted_imputer=reuse,
                )

                # 按 7:1:2 重建完整输入：训练段插补，验证段插补，测试段插补。
                full_imputed_data = np.vstack((train_imputed, val_imputed, test_imputed))

                # 将整段插补数据和 IE 指标持久化，供后续模型直接复用。
                np.savez(cache_path, full_data=full_imputed_data, ie_mae=ie_mae, ie_mse=ie_mse)
                print(f"💾 插补已计算并缓存: {cache_path}")


            # 每轮模型评估后尝试释放 CUDA 缓存。
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # 计算耗时。
            elapsed_sec = round(time.time() - start_time, 2)

            # 组织成功记录。
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

            # 保存成功记录。
            results.append(result_row)
            # 标记完成。
            completed_keys.add(exp_key)

            # 打印成功摘要。
            print(
                f"✅ 完成: {exp_key} | IE_MSE(Test)={ie_mse:.4f} | FE_MSE(Test)={fe_mse:.4f} | "
                f"BestVal={best_val_loss if best_val_loss is not None else 'N/A'} | 耗时={elapsed_sec:.2f}s"
            )

        except Exception as exc:
            # 计算异常耗时。
            elapsed_sec = round(time.time() - start_time, 2)
            # 截断错误信息。
            error_msg = str(exc)[:200]

            # 打印错误摘要。
            print(f"❌ 失败: {exp_key} | error={error_msg}")
            # 打印完整堆栈以便排查。
            traceback.print_exc()

            # 组织失败记录。
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

            # 保存失败记录。
            results.append(result_row)
            # 失败也标记完成，避免死循环重试。
            completed_keys.add(exp_key)

        # 每轮实验都保存断点与结果。
        _save_state(
            checkpoint_file=checkpoint_file,
            results_file=results_file,
            results=results,
            completed_keys=completed_keys,
        )

        # 主动触发垃圾回收。
        gc.collect()

    # 打印模型完成提示。
    print("\n" + "=" * 90)
    print(f"模型 {model_name} 全部实验完成")
    print(f"结果已写入: {results_file}")
    print("=" * 90)


# =======================
# 十一、总调度入口（可选）
# =======================

def run_all_models(model_list=None):
    """顺序执行多个模型，默认执行五个基线模型。"""
    # 使用默认模型顺序。
    if model_list is None:
        model_list = ["LSTM", "DLinear", "PatchTST"]

    # 逐个执行模型。
    for model_name in model_list:
        run_single_model(model_name)
