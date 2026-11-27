"""Evaluate r31 on MATH and LongBench v2 Multi-Document QA with the 0.8B solver."""
from argparse import ArgumentParser
from pathlib import Path

from repar.io import read_json, write_json
from repar.v4 import workflow

import resume_benchmark_recommended as resume
import run_benchmark_recommended as runner
import run_math_r28 as judging
from repeat_benchmark_recommended import SEEDS, configuration as reference_configuration


CONFIG = Path("configs/benchmark_r31_08_20261124.json")
MODEL = "08"
DATASETS = ("math", "longbench")


def configuration(seed):
    experiment = read_json(CONFIG)
    agent = read_json(experiment["agent_config"])
    settings = reference_configuration(seed)
    settings.update(
        output=f"{experiment['output']}/seed_{seed}", experiment_id=f"{experiment['experiment_id']}/seed_{seed}",
        conditions={f"{MODEL}/{dataset}": ["repar"] for dataset in DATASETS},
        **{key: agent[key] for key in ("agent_version", "plan_max_tokens", "action_max_tokens")})
    settings["sampling"][MODEL]["temperature"] = read_json(experiment["temperature_reference"])["temperature"]
    return settings


def implementation_identity():
    experiment = read_json(CONFIG)
    files = [CONFIG, Path(__file__), Path(experiment["agent_config"]), Path(experiment["temperature_reference"]),
             Path(judging.__file__), Path(resume.__file__), Path("scripts/recover_benchmark_recommended.py"),
             Path("scripts/repeat_benchmark_recommended.py"), runner.CHINESE / "zh_index.py"]
    return {p.as_posix(): runner.sha(p) for p in files}


def prepare_generation(seed):
    settings = configuration(seed)
    protocol = read_json(runner.OLD[MODEL] / "protocol.json")
    protocol["identity"]["solver"].update(seed=seed, **{key: settings[key] for key in (
        "agent_version", "plan_max_tokens", "action_max_tokens")})
    config = dict(protocol["identity"]["config"], longbench_final_max_tokens=settings["longbench"]["max_tokens"],
                  longbench_answer_grammar=settings["longbench"]["grammar"])
    protocols = {MODEL: protocol}
    runner.preflight(settings, protocols, config)
    manifest = Path(settings["output"]) / "r31_implementation.json"
    identity = implementation_identity()
    if manifest.exists():
        assert read_json(manifest) == identity
    else:
        write_json(manifest, identity)
    return settings, protocols, config


def generate(seeds, datasets):
    runner.prepare = resume.prepare
    runner.case = resume.one_case
    workflow.structured = resume.structured_unicode
    for seed in seeds:
        settings, protocol, config = prepare_generation(seed)
        runner.run(settings, protocol, config, select=[f"{MODEL}/{dataset}" for dataset in datasets])


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "generate", "grade"))
    cli.add_argument("--seed", type=int, choices=SEEDS, action="append")
    cli.add_argument("--dataset", choices=DATASETS, action="append")
    cli.add_argument("--watch", action="store_true")
    args = cli.parse_args()
    seeds = args.seed or SEEDS
    if args.phase == "freeze":
        for seed in seeds:
            prepare_generation(seed)
    elif args.phase == "generate":
        generate(seeds, args.dataset or DATASETS)
    else:
        judging.CONFIG = CONFIG
        judging.SEEDS = seeds
        judging.configuration = configuration
        judging.grade(MODEL, args.watch)
