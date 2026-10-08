# -*- coding: utf-8 -*-

# 导入共享引擎中的单模型执行函数。
from engine import run_single_model


# 直接运行当前脚本时，执行 LSTM 全实验网格。
if __name__ == "__main__":
    # 传入模型名 LSTM，启动连续数据实验流程。
    run_single_model("LSTM")
