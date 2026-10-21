#!/usr/bin/env python3
"""Start the simulator-only ExperimentServer; client disconnect does not stop it."""
from auspex_v2.experiment.service import main

if __name__ == "__main__":
    main()
