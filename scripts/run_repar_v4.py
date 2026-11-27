"""Run the existing immutable-snapshot launcher with the v4 protocol."""
import sys

from run_repar_v3 import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/v4_experiment_r10.json"])
    main()
