# -*- coding: utf-8 -*-

# Import the single-model execution function from the shared engine.
from engine import run_single_model


# When the script is executed directly, launch the LSTM model full-grid experiments.
if __name__ == "__main__":
    # Specify the model name as LSTM.
    run_single_model("LSTM")
