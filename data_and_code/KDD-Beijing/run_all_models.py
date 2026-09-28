# -*- coding: utf-8 -*-

# Import the multi-model scheduling function from the shared engine.
from engine import run_all_models


# When the script is run directly, execute all models in sequence in the default order.
if __name__ == "__main__":
    # Default order is VAR -> LSTM -> GRU -> DLinear -> PatchTST.
    run_all_models()
