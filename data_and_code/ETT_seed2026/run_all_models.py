# -*- coding: utf-8 -*-

# 导入共享引擎中的多模型总调度函数。
from engine import run_all_models


# 当脚本被直接运行时，按默认顺序依次运行全部模型。
if __name__ == "__main__":
    run_all_models()
