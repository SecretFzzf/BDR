# -*- coding: utf-8 -*-

# Import the overall scheduling function from the shared engine.
from engine import run_all_models


# When running this script directly, execute all models sequentially.
if __name__ == "__main__":
    # When no arguments are passed, VAR/LSTM/GRU/DLinear/PatchTST are executed by default.
    run_all_models()
