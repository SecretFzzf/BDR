# -*- coding: utf-8 -*-

# 导入共享引擎中的单模型执行函数。
from engine import run_single_model


# 当脚本被直接执行时，启动 PatchTST 模型全网格实验。
if __name__ == "__main__":
    # 指定模型名称为 PatchTST。
    run_single_model("PatchTST")
