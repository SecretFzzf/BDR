# -*- coding: utf-8 -*-

# 导入共享引擎中的单模型执行入口函数。
from engine import run_single_model


# 当脚本被直接运行时，启动 DLinear 全网格实验。
if __name__ == "__main__":
    # 传入模型名 DLinear，执行连续数据逻辑下的全部组合实验。
    run_single_model("DLinear")
