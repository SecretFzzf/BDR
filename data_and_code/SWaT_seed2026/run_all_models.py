# -*- coding: utf-8 -*-

# Import the main dispatch function from the shared engine.
from engine import run_all_models


# When the script is executed directly, run all models sequentially.
if __name__ == "__main__":
    # Execute with the default model list when no arguments are passed.
    run_all_models()
