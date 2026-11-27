"""Freeze and run the v5 MATH and LongBench comparison on r61's index set."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_r60 import evaluate, lock_condition
from repar.io import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5/experiment_768_42.json"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Use a positive worker count")
    config = read_json(args.config)
    suite = lock_condition(args.config, config, read_json(config["solver_config"]))
    with ThreadPoolExecutor(max_workers=1) as grading_pool:
        _, grades = evaluate(suite, args.workers, grading_pool)
        grades.result()
    suite.report()


if __name__ == "__main__":
    main()
