# -*- coding: utf-8 -*-

# Import the multi-model dispatcher from the shared engine.
from engine import run_all_models


# When the script is executed directly, run all models in the default order.
if __name__ == "__main__":
    # When no argument is passed, defaults to [VAR, LSTM, GRU, DLinear, PatchTST].
    run_all_models()
