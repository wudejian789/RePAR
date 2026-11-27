"""Run 0.8B MATH with persistent task objectives and model-written actions."""
from argparse import ArgumentParser
from pathlib import Path

from repar.io import read_json, write_json
from repar.v4 import workflow

import resume_benchmark_recommended as resume
import run_benchmark_recommended as runner
import run_math_r28 as paired


CONFIG = Path("configs/math_r30_08_20261124.json")
MODEL = "08"


def configuration(seed):
    experiment = read_json(CONFIG)
    agent = read_json(experiment["agent_config"])
    settings = paired.recommended_configuration(seed)
    settings.update(output=f"{experiment['output']}/seed_{seed}",
                    experiment_id=f"{experiment['experiment_id']}/seed_{seed}",
                    conditions={"08/math": ["repar"]}, longbench_08_parallel_endpoints=[],
                    agent_version=agent["agent_version"], action_max_tokens=agent["action_max_tokens"])
    settings["sampling"][MODEL]["temperature"] = read_json(experiment["temperature_reference"])["temperature"]
    return settings


def implementation_identity():
    experiment = read_json(CONFIG)
    paths = [CONFIG, Path(__file__), Path(experiment["agent_config"]), Path(experiment["temperature_reference"]),
             Path(paired.__file__), Path(resume.__file__), Path("scripts/recover_benchmark_recommended.py"),
             Path("scripts/repeat_benchmark_recommended.py"), runner.CHINESE / "zh_index.py"]
    return {path.as_posix(): runner.sha(path) for path in paths}


def generate(seeds):
    runner.prepare = resume.prepare
    runner.case = resume.one_case
    workflow.structured = resume.structured_unicode
    for seed in seeds:
        settings = configuration(seed)
        protocol = read_json(runner.OLD[MODEL] / "protocol.json")
        protocol["identity"]["solver"].update(seed=seed, agent_version=settings["agent_version"],
                                              action_max_tokens=settings["action_max_tokens"])
        config = protocol["identity"]["config"]
        protocols = {MODEL: protocol}
        runner.preflight(settings, protocols, config)
        path = Path(settings["output"]) / "r30_implementation.json"
        identity = implementation_identity()
        if path.exists():
            assert read_json(path) == identity
        else:
            write_json(path, identity)
        runner.run(settings, protocols, config, select=None)


def grade(seeds, watch):
    paired.CONFIG = CONFIG
    paired.SEEDS = seeds
    paired.configuration = configuration
    paired.grade(MODEL, watch)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("generate", "grade"))
    cli.add_argument("--seed", type=int, choices=paired.SEEDS, action="append")
    cli.add_argument("--watch", action="store_true")
    args = cli.parse_args()
    seeds = args.seed or paired.SEEDS
    if args.phase == "generate":
        generate(seeds)
    else:
        grade(seeds, args.watch)
