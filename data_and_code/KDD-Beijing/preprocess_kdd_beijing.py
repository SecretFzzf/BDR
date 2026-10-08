

# 导入路径处理模块，用于构造跨平台文件路径。
from pathlib import Path
# 导入正则模块，用于清洗列名中的特殊字符。
import re
# 导入 pandas，用于构建二维表并导出 CSV。
import pandas as pd
import argparse


# 定义将任意文本清洗为安全列名片段的函数。
def _sanitize_token(text: str) -> str:
    # 去掉首尾空白，避免列名出现无意义空格。
    text = text.strip()
    # 将所有非字母数字/下划线/点/连字符替换为下划线。
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    # 将连续下划线压缩为单个下划线，提升可读性。
    text = re.sub(r"_+", "_", text)
    # 去掉首尾下划线，避免列名边界噪声。
    return text.strip("_")


# 定义主函数，负责解析 TSF、筛选北京序列并导出宽表 CSV。
def preprocess_kdd_beijing(tsf_path: Path, output_csv_path: Path) -> None:
    # 定义目标时间步长度，依据数据集说明应为 10898。
    expected_length = 10898
    # 初始化用于保存“列名 -> 序列值列表”的字典容器。
    beijing_series_dict: dict[str, list[float]] = {}
    # 初始化数据段标志位，只有进入 @data 后才解析序列行。
    in_data_section = False

    # 以 UTF-8 编码打开 TSF 文件并逐行读取，节省内存开销。
    with tsf_path.open("r", encoding="utf-8") as f:
        # 使用 enumerate 获取行号，便于报错时定位。
        for line_no, raw_line in enumerate(f, start=1):
            # 去除当前行首尾空白字符（含换行符）。
            line = raw_line.strip()
            # 跳过空行，避免无效解析。
            if not line:
                # 继续处理下一行。
                continue
            # 跳过注释行，TSF 文件中注释以 # 开头。
            if line.startswith("#"):
                # 继续处理下一行。
                continue
            # 检测到 @data 标记时，开启数据段解析模式。
            if line.lower() == "@data":
                # 标记已进入数据段。
                in_data_section = True
                # 继续处理下一行。
                continue
            # 在未进入数据段前，所有元信息行都直接跳过。
            if not in_data_section:
                # 继续处理下一行。
                continue
            # 仅在数据段中处理真正的时间序列记录行。
            # 先按最后一个冒号分割，右侧应是逗号分隔的序列值。
            try:
                # 将行拆成“元信息部分”和“数值串部分”。
                meta_part, value_part = line.rsplit(":", 1)
            except ValueError as e:
                # 如果连最后一个冒号都找不到，直接抛出格式错误。
                raise ValueError(f"第 {line_no} 行格式错误：无法定位数值段") from e

            # 将元信息部分按冒号拆分，标准应有 5 个字段。
            meta_fields = meta_part.split(":")
            # 检查元信息字段数量是否合法。
            if len(meta_fields) != 5:
                # 字段数异常时抛出错误，防止 silently wrong parsing。
                raise ValueError(
                    f"第 {line_no} 行元信息字段数异常：期望 5，实际 {len(meta_fields)}"
                )

            # 依次解包元信息字段：序列编号、城市、站点、测量项、起始时间。
            series_id, city, station, measurement, start_timestamp = meta_fields
            # 构造用于筛选的“序列名语义串”，并统一转小写。
            series_name_for_filter = (
                f"{series_id}:{city}:{station}:{measurement}:{start_timestamp}".lower()
            )

            # 仅保留“序列名语义串”中包含 beijing 的记录（忽略大小写）。
            if "beijing" not in series_name_for_filter:
                # 非北京序列直接跳过。
                continue

            # 把逗号分隔的数值字符串拆成单个 token。
            value_tokens = value_part.split(",")
            # 将每个 token 转成浮点数，问号按缺失值处理为 NaN。
            values = [float(token) if token != "?" else float("nan") for token in value_tokens]

            # 严格校验每条北京序列长度必须为 10898。
            if len(values) != expected_length:
                # 长度不一致直接报错，避免后续时间步错位。
                raise ValueError(
                    f"第 {line_no} 行北京序列长度异常：期望 {expected_length}，实际 {len(values)}"
                )

            # 生成目标列名：beijing_站点_测量项，便于后续建模解释。
            city_token = _sanitize_token(city.lower())
            # 清洗站点名，避免空格与特殊字符影响列名可用性。
            station_token = _sanitize_token(station.lower())
            # 清洗测量项名（如 PM2.5 会被保留点号）。
            measurement_token = _sanitize_token(measurement)
            # 组合基础列名。
            base_col_name = f"{city_token}_{station_token}_{measurement_token}"

            # 为处理潜在重名，初始化最终列名为基础列名。
            final_col_name = base_col_name
            # 初始化重名后缀计数器。
            duplicate_idx = 1
            # 如果列名已存在，则追加编号直到唯一。
            while final_col_name in beijing_series_dict:
                # 递增后缀计数器。
                duplicate_idx += 1
                # 生成新的唯一候选列名。
                final_col_name = f"{base_col_name}_{duplicate_idx}"

            # 将北京序列写入字典，键为列名、值为该列完整时间序列。
            beijing_series_dict[final_col_name] = values

    # 若最终没有筛到北京序列，则提示并停止。
    if not beijing_series_dict:
        # 抛出空结果异常，提醒检查筛选逻辑或源文件。
        raise RuntimeError("未提取到任何包含 'beijing' 的时间序列，请检查源文件格式。")

    # 用字典直接构建宽表 DataFrame：列是变量、行索引是时间步。
    df_wide = pd.DataFrame(beijing_series_dict)
    # 将索引名显式命名为 TimeStep，表达第 0~10897 个时间步。
    df_wide.index.name = "TimeStep"

    # 再次整体校验行数必须为 10898，保证时间步完整对齐。
    if df_wide.shape[0] != expected_length:
        # 行数异常时抛错，防止导出错误结果。
        raise ValueError(
            f"最终宽表行数异常：期望 {expected_length}，实际 {df_wide.shape[0]}"
        )

    # 将结果导出为 CSV，保留 TimeStep 索引列。
    df_wide.to_csv(output_csv_path, index=True, encoding="utf-8-sig")

    # 打印提取出的北京变量总列数，用于研究记录。
    print(f"北京序列列数（N）: {df_wide.shape[1]}")
    # 打印最终 CSV 形状，理论应为 10898 x N。
    print(f"最终 CSV 形状: {df_wide.shape[0]} x {df_wide.shape[1]}")
    # 打印输出文件绝对路径，便于快速定位产物。
    print(f"输出文件: {output_csv_path.resolve()}")


# 仅当脚本被直接运行时执行以下入口逻辑。
if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=repo_root / "data")
    parser.add_argument("--output", type=Path, default=repo_root / "results" / "KDD_Beijing_Clean.csv")
    args = parser.parse_args()
    default_tsf = args.data_dir / "kdd_cup_2018_dataset_without_missing_values.tsf"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preprocess_kdd_beijing(default_tsf, args.output)
