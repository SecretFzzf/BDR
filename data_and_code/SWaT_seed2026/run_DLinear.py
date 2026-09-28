# -*- coding: utf-8 -*-

# Import the single-model execution function from the shared engine.
from engine import run_single_model


# When the script is executed directly, launch the full-grid experiment for the DLinear model.
if __name__ == "__main__":
    # Specify the model name as DLinear.
    run_single_model("DLinear")
