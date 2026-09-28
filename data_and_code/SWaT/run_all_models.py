# -*- coding: utf-8 -*-

# Import the single-model execution entry point from the shared engine.
from engine import run_single_model


# When the script is executed directly, launch the PatchTST full-grid experiments.
if __name__ == "__main__":
    # Pass the model name PatchTST to run the full combination experiments under continuous-data logic.
    run_single_model("PatchTST")
