# -*- coding: utf-8 -*-

# Import the dispatcher from the shared engine.
from engine import run_all_models


# When the script is executed directly, run all models in sequence.
if __name__ == "__main__":
    # When no argument is passed, use the default model list.
    run_all_models()
