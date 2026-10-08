# -*- coding: utf-8 -*-

# 导入共享引擎中的单模型执行函数。
from engine import run_single_model


# 直接运行脚本时，仅执行 PatchTST 全实验网格。
if __name__ == "__main__":
    # 启动 PatchTST 评估。
    run_single_model("PatchTST")
