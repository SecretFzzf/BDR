# -*- coding: utf-8 -*-

# Import the single-model execution function from the shared engine.
from engine import run_single_model


# When the script is run directly, only execute the full PatchTST experiment grid.
if __name__ == "__main__":
    # Launch PatchTST evaluation.
    run_single_model("PatchTST")
