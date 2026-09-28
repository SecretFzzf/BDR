# -*- coding: utf-8 -*-

# Import the single-model execution entry function from the shared engine.
from engine import run_single_model


# When the script is run directly, launch the PatchTST full-grid experiment.
if __name__ == "__main__":
    # Pass the model name PatchTST to run all combination experiments under the continuous-data logic.
    run_single_model("PatchTST")
