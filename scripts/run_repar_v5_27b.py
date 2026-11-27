"""Run frozen v5 indexes with the existing 27B deployment and benchmark samplers."""
import argparse
from pathlib import Path

from repar.benchmarks.v3_suite import freeze
from repar.benchmarks.v4_suite import V4Suite
from repar.client import sampling_parameters
from repar.io import read_json
from run_r60 import check_run_index


class FrozenIndexSuite(V4Suite):
    def load_index(self, context, kind):
        index, metadata = super().load_index(context, kind)
        check_run_index(self, index, metadata)
        return index, metadata


def select_dataset(suite, base_solver, dataset):
    sampling = suite.config["sampling_by_dataset"][dataset]
    sampling_parameters({"sampling": sampling})
    suite.solver = {**base_solver, "sampling": dict(sampling)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5/experiment_768_42_direct8k_27b.json"))
    args = parser.parse_args()
    freeze(args.config)
    suite = FrozenIndexSuite(args.config)
    base_solver = dict(suite.solver)
    select_dataset(suite, base_solver, "math")
    suite.generate_math()
    # Solver and judge share the same single-slot 27B deployment. Grade before
    # LongBench so different inference roles do not compete for the slot.
    suite.grade_math()
    select_dataset(suite, base_solver, "longbench")
    suite.generate_longbench()
    suite.solver = base_solver
    suite.report()


if __name__ == "__main__":
    main()
