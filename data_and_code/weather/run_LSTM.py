# -*- coding: utf-8 -*-

# Import the single-model execution function from the shared engine.
from engine import run_single_model


# When running this script directly, execute the LSTM full experiment grid.
if __name__ == "__main__":
    # Pass the model name LSTM to launch the continuous data experiment flow.
    run_single_model("LSTM")
