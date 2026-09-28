# -*- coding: utf-8 -*-

# Import the single-model execution function from the shared engine.
from engine import run_single_model


# When running this script directly, execute the PatchTST full experiment grid.
if __name__ == "__main__":
    # Pass the model name PatchTST to launch the continuous data experiment flow.
    run_single_model("PatchTST")
