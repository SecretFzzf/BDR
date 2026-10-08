# -*- coding: utf-8 -*-

# 导入共享引擎中的多模型调度函数。
from engine import run_all_models


# 直接运行脚本时，按默认顺序依次执行全部模型。
if __name__ == "__main__":
    run_all_models()
