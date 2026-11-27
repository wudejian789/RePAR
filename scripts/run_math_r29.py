"""Run the 0.8B MATH query rewrite comparison on the existing Chinese index."""
from argparse import ArgumentParser
from pathlib import Path

from repar.io import read_json, write_json
from repar.v4 import selection, workflow

import resume_benchmark_recommended as resume
import run_benchmark_recommended as runner
import run_math_r28 as paired


CONFIG = Path("configs/math_r29_08_20261124.json")
MODEL = "08"


def configuration(seed):
    experiment = read_json(CONFIG)
    settings = paired.recommended_configuration(seed)
    settings.update(output=f"{experiment['output']}/seed_{seed}",
                    experiment_id=f"{experiment['experiment_id']}/seed_{seed}",
                    conditions={f"{MODEL}/math": ["repar"]}, longbench_08_parallel_endpoints=[],
                    agent_version=experiment["agent_version"])
    settings["sampling"][MODEL]["temperature"] = read_json(experiment["temperature_reference"])["temperature"]
    return settings


def implementation_identity():
    files = [CONFIG, Path(__file__), Path(resume.__file__), Path(paired.__file__),
             Path("scripts/recover_benchmark_recommended.py"), Path("scripts/repeat_benchmark_recommended.py"),
             Path(read_json(CONFIG)["temperature_reference"]), runner.CHINESE / "zh_index.py"]
    return {path.as_posix(): runner.sha(path) for path in files}


def generate(seeds):
    runner.prepare = resume.prepare
    runner.case = resume.one_case
    workflow.structured = selection.structured = resume.structured_unicode
    for seed in seeds:
        settings = configuration(seed)
        protocol = read_json(runner.OLD[MODEL] / "protocol.json")
        protocol["identity"]["solver"].update(seed=seed, agent_version=settings["agent_version"])
        config = protocol["identity"]["config"]
        protocols = {MODEL: protocol}
        runner.preflight(settings, protocols, config)
        manifest = Path(settings["output"]) / "r29_implementation.json"
        identity = implementation_identity()
        if manifest.exists():
            assert read_json(manifest) == identity
        else:
            write_json(manifest, identity)
        runner.run(settings, protocols, config, select=None)


def grade(seeds, watch):
    paired.CONFIG = CONFIG
    paired.SEEDS = seeds
    paired.configuration = configuration
    paired.grade(MODEL, watch)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("generate", "grade"))
    cli.add_argument("--seed", choices=paired.SEEDS, type=int, action="append")
    cli.add_argument("--watch", action="store_true")
    args = cli.parse_args()
    seeds = args.seed or paired.SEEDS
    if args.phase == "generate":
        generate(seeds)
    else:
        grade(seeds, args.watch)
