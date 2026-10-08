# -*- coding: utf-8 -*-

# 导入系统相关库，用于路径处理、异常追踪、时间统计、垃圾回收和日志存储。
import os
import sys
import json
import argparse
import time
import gc
import traceback
import warnings
from pathlib import Path

# 导入数值与表格处理库。
import numpy as np
import pandas as pd

# 导入经典机器学习评估与标准化工具。
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 导入 PyTorch 生态，用于深度学习模型构建与训练。
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 统一硬件设备检测：优先 CUDA，其次 MPS，最后 CPU。
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# 对固定输入形状任务启用 cuDNN 自动调优，提升 3090 吞吐。
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 关闭大量无关警告，避免终端输出过于嘈杂。
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# 获取当前脚本所在目录，确保脚本从任意工作目录启动都能正确定位文件。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "training"
CHECKPOINT_DIR = REPO_ROOT / ".cache" / "checkpoints"

# 把当前目录加入 Python 模块搜索路径，便于导入同目录下的工具文件。
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
# 一、全局配置（保持连续数据逻辑）
# =============================

# 数据文件路径（ETT 主数据，默认标准命名）。
DATA_FILE = DEFAULT_DATA_DIR / "ETTh1.csv"

def _resolve_data_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    args, _ = parser.parse_known_args()
    return args.data_dir / DATA_FILE.name

# ETT 使用的变量列，与原脚本保持一致。
VALUE_COLS = ["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

# 缺失机制集合（3 种）。
MISSING_MODES = ["MCAR", "MAR_Block", "MNAR"]

# 插补方法集合（5 种）。
IMPUTE_METHODS = ["mean", "spline", "knn", "brits", "saits"]

# 预测步长集合（ETT 原逻辑）。
PRED_LEN_LIST = [24, 48, 96, 192]

# 缺失率集合（4 档）。
MISSING_RATES = [0.10, 0.30, 0.50, 0.70]

# 固定训练/验证/测试划分比例（连续序列 7:1:2）。
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# 深度学习输入历史窗口长度（保持 ETT 原逻辑为 96）。
SEQ_LEN = 96

# 滚动评估步长（保持原逻辑为 24）。
STRIDE = 24

# 深度学习默认训练配置（基础配置）。
DEFAULT_DL_CONFIG = {
    "batch_size": 2048,
    "learning_rate": 1e-3,
    "max_epochs": 50,
    "num_workers": 12,
}

# 模型特定覆盖配置（仅覆盖差异项）。
MODEL_SPECIFIC_CONFIG = {
    "DLinear": {"batch_size": 4096, "learning_rate": 5e-4, "num_workers": 12},
    "PatchTST": {"batch_size": 512, "learning_rate": 1e-4, "num_workers": 12},
    "LSTM": {"batch_size": 2048, "num_workers": 12},  # RidgeCV 候选alpha网格，内部留一法交叉验证自动选择
}

# 早停耐心轮数（连续未提升即停止）。
EARLY_STOPPING_PATIENCE = 12

# 验证损失最小改善阈值。
EARLY_STOPPING_MIN_DELTA = 1e-6

# 随机种子，保证缺失注入和实验可复现。
SEED = 42


# =================================
# 二、数据集定义（连续滑窗，不跨边界）
# =================================


class ContinuousTimeSeriesDataset(Dataset):
    """
    连续时间序列滑窗数据集。

    输入张量形状：
        x_window: [seq_len, n_features]
        y_window: [pred_len, n_features]

    注意：
        这里是“连续物理数据”的标准滑窗，不使用任何 stay_id 或分段病人边界。
    """

    def __init__(self, x_array, y_array, seq_len, pred_len, stride=1):
        # 保存输入特征数组。
        self.x_array = x_array
        # 保存监督目标数组。
        self.y_array = y_array
        # 保存历史窗口长度。
        self.seq_len = seq_len
        # 保存预测窗口长度。
        self.pred_len = pred_len
        # 保存滑窗步长。
        self.stride = stride
        # 构建所有合法起点索引，保证每个窗口长度完整。
        self.indices = np.arange(0, len(x_array) - seq_len - pred_len + 1, stride)

    def __len__(self):
        # 返回可用窗口数量。
        return len(self.indices)

    def __getitem__(self, idx):
        # 取得当前样本对应的起点位置。
        start = self.indices[idx]
        # 切出输入历史窗口，形状为 [seq_len, n_features]。
        x_window = self.x_array[start : start + self.seq_len]
        # 切出未来标签窗口，形状为 [pred_len, n_features]。
        y_window = self.y_array[
            start + self.seq_len : start + self.seq_len + self.pred_len
        ]
        # 转成 float32 Tensor，供模型训练使用。
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(
            y_window, dtype=torch.float32
        )


class ETTh1TemporalSplitDataset(Dataset):
    """ETTh1 严格时间切分数据集（支持 7:1:2 分段窗口抽取）。"""

    def __init__(
        self, x_array, y_array, split_start, split_end, seq_len, pred_len, stride=1
    ):
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
        if (
            self.split_start < 0
            or self.split_end > len(x_array)
            or self.split_start >= self.split_end
        ):
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
        x_window = self.x_array[start : start + self.seq_len]
        # 切出未来标签窗口。
        y_window = self.y_array[
            start + self.seq_len : start + self.seq_len + self.pred_len
        ]
        # 返回 float32 张量。
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(
            y_window, dtype=torch.float32
        )


# =========================
# 三、模型定义（统一放引擎）
# =========================


class LSTMForecaster(nn.Module):
    """LSTM 多变量到多变量预测器。"""

    def __init__(self, num_features, hidden_size, num_layers, pred_len):
        # 调用父类初始化。
        super().__init__()
        # 保存特征维度。
        self.num_features = num_features
        # 保存预测长度。
        self.pred_len = pred_len
        # 构建 LSTM 编码器，输入形状为 [B, T, C]。
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # 线性层把最后时刻隐藏状态映射到 pred_len * num_features。
        self.proj = nn.Linear(hidden_size, pred_len * num_features)

    def forward(self, x):
        # 输入 x 形状为 [B, seq_len, C]。
        out, _ = self.lstm(x)
        # 取最后时间步隐藏表示，形状为 [B, hidden_size]。
        last_hidden = out[:, -1, :]
        # 线性映射到 [B, pred_len * C]。
        pred_flat = self.proj(last_hidden)
        # 重塑为 [B, pred_len, C]，与标签形状对齐。
        pred = pred_flat.view(-1, self.pred_len, self.num_features)
        # 返回预测张量。
        return pred




class DLinearForecaster(nn.Module):
    """DLinear 预测器（逐通道线性映射）。"""

    def __init__(self, seq_len, pred_len, enc_in):
        # 调用父类初始化。
        super().__init__()
        # 保存输入长度。
        self.seq_len = seq_len
        # 保存预测长度。
        self.pred_len = pred_len
        # 保存通道数。
        self.enc_in = enc_in
        # 为每个通道创建独立线性层，完成 [seq_len] -> [pred_len]。
        self.linears = nn.ModuleList(
            [nn.Linear(seq_len, pred_len) for _ in range(enc_in)]
        )

    def forward(self, x):
        # 输入 x 形状为 [B, seq_len, C]。
        outputs = []
        # 逐通道做线性预测。
        for i in range(self.enc_in):
            # 取第 i 个通道，得到 [B, seq_len]。
            x_i = x[:, :, i]
            # 线性映射后得到 [B, pred_len]。
            y_i = self.linears[i](x_i)
            # 保存当前通道预测。
            outputs.append(y_i)
        # 堆叠后得到 [B, pred_len, C]。
        pred = torch.stack(outputs, dim=2)
        # 返回预测张量。
        return pred


class PatchTSTForecaster(nn.Module):
    """简化版 PatchTST 预测器。"""

    def __init__(
        self, seq_len, pred_len, enc_in, patch_len=16, stride=8, e_layers=2, d_model=128
    ):
        # 调用父类初始化。
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

        # 计算 patch 数量，形状推导：n_patches = floor((seq_len - patch_len)/stride) + 1。
        self.n_patches = (seq_len - patch_len) // stride + 1

        # 每个通道独立 patch 嵌入层，把 [patch_len] 映射到 [d_model]。
        self.patch_embedding = nn.ModuleList(
            [nn.Linear(patch_len, d_model) for _ in range(enc_in)]
        )

        # 定义 Transformer 编码层。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )

        # 堆叠多层 Transformer 编码器。
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # 每个通道独立预测头，把 [n_patches * d_model] 映射到 [pred_len]。
        self.pred_heads = nn.ModuleList(
            [nn.Linear(d_model * self.n_patches, pred_len) for _ in range(enc_in)]
        )

    def forward(self, x):
        # 输入 x 形状为 [B, seq_len, C]。
        batch_size, _, n_features = x.shape
        # 用于保存所有通道预测结果。
        out_all = []

        # 按通道独立执行 patch 切分、编码与预测。
        for i in range(n_features):
            # 取第 i 通道，形状变为 [B, seq_len]。
            x_i = x[:, :, i]
            # 存储该通道所有 patch。
            patches = []
            # 逐 patch 切分。
            for j in range(self.n_patches):
                # 当前 patch 起点。
                start = j * self.stride
                # 当前 patch 终点。
                end = start + self.patch_len
                # 切出 patch，形状 [B, patch_len]。
                patch = x_i[:, start:end]
                # 追加到 patch 列表。
                patches.append(patch)

            # 堆叠 patch，得到 [B, n_patches, patch_len]。
            patches = torch.stack(patches, dim=1)
            # patch 嵌入后得到 [B, n_patches, d_model]。
            emb = self.patch_embedding[i](patches)
            # Transformer 编码后形状保持 [B, n_patches, d_model]。
            enc = self.encoder(emb)
            # 展平 patch 维与特征维，得到 [B, n_patches * d_model]。
            enc_flat = enc.reshape(batch_size, -1)
            # 预测头输出 [B, pred_len]。
            y_i = self.pred_heads[i](enc_flat)
            # 保存该通道预测。
            out_all.append(y_i)

        # 合并所有通道，得到 [B, pred_len, C]。
        pred = torch.stack(out_all, dim=2)
        # 返回预测张量。
        return pred


# =====================
# 四、数据加载与预处理
# =====================


def load_and_preprocess_data():
    """
    加载 ETT 数据并执行严格防泄露标准化。

    返回：
        train_data_full: 标准化训练段，形状 [train_T, C]
        val_data_full: 标准化验证段，形状 [val_T, C]
        test_data_full: 标准化测试段，形状 [test_T, C]
        full_clean_data: 标准化完整序列，形状 [T, C]
        value_cols: 有效特征列名列表
        split_info: 时间切分索引信息
    """
    # 优先读取标准文件名，若不存在则回退到常用 ETTh1 文件。
    candidate_files = [
        DATA_FILE,
        os.path.join(BASE_DIR, "ETTh1.csv"),
    ]
    existing_files = [p for p in candidate_files if os.path.exists(p)]
    if not existing_files:
        raise FileNotFoundError(f"找不到可用数据文件，请检查: {candidate_files}")
    data_file = existing_files[0]

    # 读取 CSV 数据。
    df = pd.read_csv(data_file)
    # 若存在日期列则先删除，避免非数值列影响标准化。
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # 优先使用预定义变量列；若列缺失则回退到全部数值列。
    missing_cols = [c for c in VALUE_COLS if c not in df.columns]
    if len(missing_cols) == 0:
        numeric_df = df[VALUE_COLS].copy()
    else:
        numeric_df = df.select_dtypes(include=[np.number]).copy()

    # 若无可用数值列则直接报错。
    if numeric_df.shape[1] == 0:
        raise ValueError(f"{os.path.basename(data_file)} 中没有可用的数值列。")

    # 提取多变量数值矩阵，形状 [T, C]。
    data = numeric_df.values.astype(np.float64)
    # 获取有效列名列表。
    value_cols = list(numeric_df.columns)

    # 计算总长度。
    total_len = len(data)
    # 计算训练结束索引。
    train_end = int(total_len * TRAIN_RATIO)
    # 计算验证结束索引。
    val_end = train_end + int(total_len * VAL_RATIO)

    # 防御式修正：避免切分点越界。
    train_end = max(1, min(train_end, total_len - 2))
    val_end = max(train_end + 1, min(val_end, total_len - 1))

    # 切分训练段。
    train_data_raw = data[:train_end]

    # 构建标准化器。
    scaler = StandardScaler()
    # 仅在训练集拟合，避免测试信息泄露。
    scaler.fit(train_data_raw)

    # 使用训练统计量变换全量数据。
    full_clean_data = scaler.transform(data)

    # 按 7:1:2 切分标准化后数据。
    train_data_full = full_clean_data[:train_end]
    val_data_full = full_clean_data[train_end:val_end]
    test_data_full = full_clean_data[val_end:]

    # 打印数据概览信息。
    print("=" * 80)
    print("ETT 数据加载完成")
    print(f"数据文件: {os.path.basename(data_file)}")
    print(f"总长度: {total_len}")
    print(f"训练长度: {len(train_data_full)}")
    print(f"验证长度: {len(val_data_full)}")
    print(f"测试长度: {len(test_data_full)}")
    print(f"特征维度: {len(value_cols)}")
    print("=" * 80)

    # 构建时间切分信息字典。
    split_info = {
        "train_end": int(train_end),
        "val_end": int(val_end),
        "total_len": int(total_len),
    }

    # 返回预处理结果。
    return (
        train_data_full,
        val_data_full,
        test_data_full,
        full_clean_data,
        value_cols,
        split_info,
    )


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


def inject_and_impute(
    train_clean,
    missing_mode,
    missing_rate,
    impute_method,
    verbose=True,
    fitted_imputer=None,
):
    """
    路由到对应缺失机制与插补方法。

    返回：
        train_imputed: 注入缺失后完成插补的训练集
        ie_mae: 插补误差 MAE
        ie_mse: 插补误差 MSE
        fitted_imputer: 训练阶段拟合好的插补器或统计量
    """
    # MCAR：完全随机缺失。
    if missing_mode == "MCAR":
        return inject_mcar_missing_and_impute(
            train_clean,
            missing_rate,
            impute_method,
            SEED,
            verbose=verbose,
            fitted_imputer=fitted_imputer,
        )

    # MAR_Block：跨变量的块状缺失。
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

    # MNAR：与数值大小相关的缺失。
    if missing_mode == "MNAR":
        return inject_mnar_value_missing_and_impute(
            train_clean,
            missing_rate,
            impute_method,
            SEED,
            verbose=verbose,
            fitted_imputer=fitted_imputer,
        )

    # 未知模式直接报错。
    raise ValueError(f"未知缺失模式: {missing_mode}")


# =====================
# 六、实验状态持久化工具
# =====================


def _load_state(checkpoint_file, results_file):
    """加载断点与历史结果。"""
    # 初始化结果列表。
    results = []
    # 初始化已完成实验键集合。
    completed_keys = set()

    # 若存在断点文件则读取。
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            completed_keys = set(json.load(f))

    # 若存在结果文件则读取。
    if os.path.exists(results_file):
        results = pd.read_csv(results_file).to_dict("records")

    # 返回结果与断点集合。
    return results, completed_keys


def _save_state(checkpoint_file, results_file, results, completed_keys):
    """保存断点与结果。"""
    # 写入已完成实验键。
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, ensure_ascii=False)

    # 若有结果则写出 CSV。
    if results:
        pd.DataFrame(results).to_csv(results_file, index=False)


# =====================
# 七、实验组合与模型构建
# =====================


def build_experiments():
    """构建实验网格并按插补方法复杂度排序。"""
    # 初始化实验列表。
    experiments = []
    # 遍历所有预测步长。
    for pred_len in PRED_LEN_LIST:
        # 遍历所有缺失机制。
        for missing_mode in MISSING_MODES:
            # 遍历所有缺失率。
            for missing_rate in MISSING_RATES:
                # 遍历所有插补方法。
                for impute_method in IMPUTE_METHODS:
                    # 记录一个实验组合字典。
                    experiments.append(
                        {
                            "pred_len": pred_len,
                            "missing_mode": missing_mode,
                            "missing_rate": missing_rate,
                            "impute_method": impute_method,
                        }
                    )

    # 定义插补方法顺序：先快后慢。
    method_order = {"mean": 0, "spline": 1, "knn": 2, "brits": 3, "saits": 4}
    # 排序规则：先按方法复杂度，再按预测步长。
    experiments.sort(key=lambda x: (method_order[x["impute_method"]], x["pred_len"]))
    # 返回实验列表。
    return experiments


def build_model(model_name, num_features, pred_len):
    """根据模型名构建对应网络。"""
    # LSTM 分支。
    if model_name == "LSTM":
        return LSTMForecaster(
            num_features=num_features, hidden_size=64, num_layers=2, pred_len=pred_len
        )
    # DLinear 分支。
    if model_name == "DLinear":
        return DLinearForecaster(
            seq_len=SEQ_LEN, pred_len=pred_len, enc_in=num_features
        )
    # PatchTST 分支。
    if model_name == "PatchTST":
        return PatchTSTForecaster(
            seq_len=SEQ_LEN, pred_len=pred_len, enc_in=num_features
        )
    # 非法模型名报错。
    raise ValueError(f"未知模型名: {model_name}")


# ===========================
# 八、深度学习模型训练与评估
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
    针对深度学习模型执行连续滑窗训练与测试。

    参数：
        model_name: 模型名（LSTM/DLinear/PatchTST）
        full_imputed_data: 三段插补后完整序列，形状 [T, C]
        full_clean_data: 标准化后的干净完整序列，形状 [T, C]
        split_info: 时间切分信息字典
        pred_len: 预测步长
        missing_mode: 缺失机制名称
        missing_rate: 缺失率

    返回：
        fe_mae: 预测误差 MAE
        fe_mse: 预测误差 MSE
        train_windows: 训练窗口数量
        val_windows: 验证窗口数量
        test_windows: 测试窗口数量
        best_val_loss: 最优验证损失
        best_epoch: 最优轮次
        best_model_path: 最优模型路径
    """
    # 读取严格时间切分点。
    train_end = int(split_info["train_end"])
    val_end = int(split_info["val_end"])
    total_len = int(split_info["total_len"])

    # 在完整序列上按时间边界抽取 train/val/test 窗口。
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

    # 防御式检查窗口数量。
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError(
            "窗口数量为 0，请检查 seq_len/pred_len 或 7:1:2 时间切分边界。"
        )

    # 合并默认配置与模型特定配置。
    model_cfg = {**DEFAULT_DL_CONFIG, **MODEL_SPECIFIC_CONFIG.get(model_name, {})}
    # 读取当前模型批大小。
    batch_size = int(model_cfg["batch_size"])
    # 读取当前模型学习率。
    learning_rate = float(model_cfg["learning_rate"])
    # 读取当前模型最大 epoch。
    max_epochs = int(model_cfg["max_epochs"])
    # 读取 DataLoader 工作进程数。
    num_workers = int(model_cfg.get("num_workers", 12))

    # CUDA 下启用 pin_memory，加速 CPU 到 GPU 拷贝。
    use_pin_memory = device.type == "cuda"
    # 多线程加载时启用持久 worker，减少每轮重建开销。
    persistent_workers = bool(num_workers > 0)

    # 构建训练 DataLoader。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    # 构建验证 DataLoader。
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    # 构建测试 DataLoader。
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    # 根据模型名创建网络。
    model = build_model(
        model_name=model_name,
        num_features=full_imputed_data.shape[1],
        pred_len=pred_len,
    )
    # 把模型迁移到设备。
    model = model.to(device)

    # 构建 Adam 优化器。
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = masked_mse_loss
    # 为当前实验生成唯一权重文件名，避免并行实验互相覆盖。
    missing_mode_tag = str(missing_mode).replace("/", "_")
    best_model_path = os.path.join(
        BASE_DIR,
        f"best_model_{model_name}_{missing_mode_tag}_{int(round(float(missing_rate) * 100))}_{pred_len}.pth",
    )
    # 初始化最优验证损失。
    best_val_loss = float("inf")
    # 初始化最优轮次。
    best_epoch = -1
    # 初始化连续未提升轮数。
    bad_epochs = 0

    try:
        # 执行多轮 epoch。
        for epoch in range(max_epochs):
            # 切换训练模式。
            model.train()
            # 初始化 epoch 累计损失。
            epoch_loss = 0.0
            # 遍历所有训练批次。
            for batch_x, batch_y in train_loader:
                # 输入迁移到设备，形状 [B, seq_len, C]。
                batch_x = batch_x.to(device)
                # 标签迁移到设备，形状 [B, pred_len, C]。
                batch_y = batch_y.to(device)

                # 梯度清零。
                optimizer.zero_grad()
                # 前向传播，输出形状 [B, pred_len, C]。
                pred = model(batch_x)
                # 计算 MSE 损失。
                loss = criterion(pred, batch_y)
                # 反向传播。
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # 参数更新。
                optimizer.step()
                # 累计损失。
                epoch_loss += loss.item()

            # 计算当前轮训练损失。
            train_loss = epoch_loss / max(1, len(train_loader))
            # 计算当前轮验证损失。
            val_loss = _run_eval_loss(
                model=model, data_loader=val_loader, criterion=criterion
            )

            # 打印训练/验证损失。
            print(
                f"   [{model_name}] Epoch {epoch + 1}/{max_epochs}, "
                f"TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}"
            )

            # 若验证损失提升则保存最优模型。
            if val_loss < (best_val_loss - EARLY_STOPPING_MIN_DELTA):
                best_val_loss = float(val_loss)
                best_epoch = int(epoch + 1)
                bad_epochs = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                # 连续未提升计数加一。
                bad_epochs += 1
                # 命中早停条件则提前结束训练。
                if bad_epochs >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"   [{model_name}] EarlyStopping 触发: "
                        f"连续 {EARLY_STOPPING_PATIENCE} 轮验证集未提升。"
                    )
                    break

        # 加载验证阶段选出的最佳权重，并在测试集计算 FE。
        if os.path.exists(best_model_path):
            model.load_state_dict(
                torch.load(best_model_path, map_location=device, weights_only=True)
            )

        # 切换评估模式。
        model.eval()
        # 初始化预测列表。
        preds = []
        # 初始化真值列表。
        trues = []

        # 关闭梯度计算，减少显存和加速推理。
        with torch.no_grad():
            # 遍历测试批次。
            for batch_x, batch_y in test_loader:
                # 输入迁移设备。
                batch_x = batch_x.to(device)
                # 前向预测并转回 numpy。
                batch_pred = model(batch_x).cpu().numpy()
                # 保存预测。
                preds.append(batch_pred)
                # 保存真值。
                trues.append(batch_y.numpy())

        # 拼接全部预测，形状 [N, pred_len, C]。
        preds = np.concatenate(preds, axis=0)
        # 拼接全部真值，形状 [N, pred_len, C]。
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

        # CUDA 下主动释放显存缓存。
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 返回指标与窗口数。
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
# =========================




# =======================
# 十、单模型统一执行入口
# =======================


def run_single_model(model_name):
    global DATA_FILE
    DATA_FILE = _resolve_data_file()
    """
    运行单个模型的完整实验网格。

    网格维度：
        预测步长 × 缺失机制 × 缺失率 × 插补方法
    """
    # 为每个模型单独定义结果文件，便于多 GPU 并发运行时互不覆盖。
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / Path(BASE_DIR).name / f"results_refactor_{model_name.lower()}.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    # 为每个模型单独定义 checkpoint 文件，支持中断续跑。
    checkpoint_file = CHECKPOINT_DIR / Path(BASE_DIR).name / f"checkpoint_refactor_{model_name.lower()}.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    # 加载并预处理数据。
    (
        train_data_full,
        val_data_full,
        test_data_full,
        full_clean_data,
        value_cols,
        split_info,
    ) = load_and_preprocess_data()

    # 读取历史状态。
    results, completed_keys = _load_state(
        checkpoint_file=checkpoint_file, results_file=results_file
    )

    # 构建实验网格。
    experiments = build_experiments()

    # 打印运行概览。
    print(f"模型: {model_name}")
    print(f"实验总数: {len(experiments)}")
    print(f"结果文件: {results_file}")
    print(f"断点文件: {checkpoint_file}")

    # 遍历所有实验组合。
    for exp_idx, exp in enumerate(experiments, 1):
        # 读取当前预测长度。
        pred_len = exp["pred_len"]
        # 读取当前缺失机制。
        missing_mode = exp["missing_mode"]
        # 读取当前缺失率。
        missing_rate = exp["missing_rate"]
        # 读取当前插补方法。
        impute_method = exp["impute_method"]

        # 生成实验唯一键，用于断点续跑判断。
        exp_key = f"{model_name}_{missing_mode}_{impute_method}_{pred_len}_{int(missing_rate * 100)}"

        # 若实验已完成则跳过。
        if exp_key in completed_keys:
            print(f"[{exp_idx}/{len(experiments)}] 跳过已完成: {exp_key}")
            continue

        # 打印当前实验头信息。
        print("\n" + "=" * 90)
        print(
            f"[{exp_idx}/{len(experiments)}] 模型={model_name}, 缺失机制={missing_mode}, "
            f"插补={impute_method}, pred_len={pred_len}, 缺失率={int(missing_rate * 100)}%"
        )
        print("=" * 90)

        # 记录实验开始时间。
        start_time = time.time()

        try:
            # ----------------------------------------------------------
            # Cache-First 插补流程：先复用缓存，未命中时再计算并写入缓存。
            # 说明：缓存键不含 pred_len，因为插补仅依赖缺失机制/方法/缺失率。
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

                # 测试段同样处理。
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
                full_imputed_data = np.vstack(
                    (train_imputed, val_imputed, test_imputed)
                )

                # 将整段插补数据和 IE 指标持久化，供后续模型直接复用。
                np.savez(
                    cache_path,
                    full_data=full_imputed_data,
                    ie_mae=ie_mae,
                    ie_mse=ie_mse,
                )
                print(f"💾 插补已计算并缓存: {cache_path}")


            # 每轮评估后尝试释放 CUDA 缓存。
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # 计算单次实验耗时。
            elapsed_sec = round(time.time() - start_time, 2)

            # 整理成功记录行。
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

            # 追加到结果列表。
            results.append(result_row)
            # 标记当前实验已完成。
            completed_keys.add(exp_key)

            # 打印成功摘要。
            print(
                f"✅ 完成: {exp_key} | IE_MSE(Test)={ie_mse:.4f} | FE_MSE(Test)={fe_mse:.4f} | "
                f"BestVal={best_val_loss if best_val_loss is not None else 'N/A'} | 耗时={elapsed_sec:.2f}s"
            )

        except Exception as exc:
            # 计算失败实验耗时。
            elapsed_sec = round(time.time() - start_time, 2)
            # 截断错误信息，避免 CSV 内容过长。
            error_msg = str(exc)[:200]

            # 打印失败信息。
            print(f"❌ 失败: {exp_key} | error={error_msg}")
            # 打印完整堆栈，便于调试。
            traceback.print_exc()

            # 整理失败记录行。
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

            # 追加失败记录。
            results.append(result_row)
            # 即使失败也标记为已处理，防止无限重复同一错误。
            completed_keys.add(exp_key)

        # 每个实验结束即保存一次，确保可断点续跑。
        _save_state(
            checkpoint_file=checkpoint_file,
            results_file=results_file,
            results=results,
            completed_keys=completed_keys,
        )

        # 触发垃圾回收，降低长跑内存占用。
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
    """顺序运行多个模型，便于单机串行调度。"""
    # 若未显式传入模型列表，则使用默认 3 模型。
    if model_list is None:
        model_list = ["LSTM", "DLinear", "PatchTST"]

    # 逐个运行模型。
    for model_name in model_list:
        run_single_model(model_name)
