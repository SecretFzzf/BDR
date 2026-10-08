# -*- coding: utf-8 -*-

# 导入共享引擎中的总调度函数。
from engine import run_all_models


# 当脚本被直接执行时，顺序运行全部模型。
if __name__ == "__main__":
    # 不传参时使用默认模型列表执行。
    run_all_models()
