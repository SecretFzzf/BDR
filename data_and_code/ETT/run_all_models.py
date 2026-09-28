# -*- coding: utf-8 -*-

# Import the multi-model master scheduling function from the shared engine.
from engine import run_all_models


# When the script is run directly, run all models in order.
if __name__ == "__main__":
    # With no arguments, defaults to executing [VAR, LSTM, GRU, DLinear, PatchTST].
    run_all_models()
