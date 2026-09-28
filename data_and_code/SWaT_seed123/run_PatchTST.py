# -*- coding: utf-8 -*-

# Import the single-model execution function from the shared engine.
from engine import run_single_model


# When this script is executed directly, launch the full grid experiment for the PatchTST model.
if __name__ == "__main__":
    # Specify the model name as PatchTST.
    run_single_model("PatchTST")
