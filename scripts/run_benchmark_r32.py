"""Run the paired 0.8B r32 benchmark with the established r31 evaluation protocol."""
from argparse import ArgumentParser
from pathlib import Path

from repar.io import read_json, write_json
from repar.v4 import workflow

import run_benchmark_r31 as suite


CONFIG = Path("configs/benchmark_r32_08_20261125.json")


def configuration(seed):
    suite.CONFIG = CONFIG
    return suite.configuration(seed)


def implementation_identity():
    suite.CONFIG = CONFIG
    return {**suite.implementation_identity(), "scripts/run_benchmark_r32.py": suite.runner.sha(Path(__file__))}


def prepare(seed):
    settings = configuration(seed)
    protocol = read_json(suite.runner.OLD[suite.MODEL] / "protocol.json")
    protocol["identity"]["solver"].update(seed=seed, **{
        key: settings[key] for key in ("agent_version", "plan_max_tokens", "action_max_tokens")})
    config = dict(protocol["identity"]["config"], longbench_final_max_tokens=settings["longbench"]["max_tokens"],
                  longbench_answer_grammar=settings["longbench"]["grammar"])
    protocols = {suite.MODEL: protocol}
    suite.runner.preflight(settings, protocols, config)
    manifest = Path(settings["output"]) / "r32_implementation.json"
    identity = implementation_identity()
    if manifest.exists():
        assert read_json(manifest) == identity
    else:
        write_json(manifest, identity)
    return settings, protocols, config


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "generate", "grade"))
    cli.add_argument("--seed", type=int, choices=suite.SEEDS, action="append")
    cli.add_argument("--dataset", choices=suite.DATASETS, action="append")
    cli.add_argument("--watch", action="store_true")
    args = cli.parse_args()
    seeds = args.seed or suite.SEEDS
    if args.phase == "grade":
        suite.judging.CONFIG = CONFIG
        suite.judging.SEEDS = seeds
        suite.judging.configuration = configuration
        suite.judging.grade(suite.MODEL, args.watch)
        return
    suite.runner.prepare = suite.resume.prepare
    suite.runner.case = suite.resume.one_case
    workflow.structured = suite.resume.structured_unicode
    for seed in seeds:
        settings, protocol, config = prepare(seed)
        if args.phase == "generate":
            select = [f"{suite.MODEL}/{dataset}" for dataset in (args.dataset or suite.DATASETS)]
            suite.runner.run(settings, protocol, config, select=select)


if __name__ == "__main__":
    main()
