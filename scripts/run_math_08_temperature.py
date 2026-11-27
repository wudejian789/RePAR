"""Generate the paired 0.8B MATH temperature comparison using frozen indexes."""
from argparse import ArgumentParser
from pathlib import Path

from repar.io import read_json, write_json
from repar.v4 import selection, workflow

import resume_benchmark_recommended as scoped_runner
import run_benchmark_recommended as runner
from repeat_benchmark_recommended import SEEDS, configuration as reference_configuration


CONFIG_PATH = Path("configs/math_08_temperature_01_20261124.json")
_archived_request = runner.archived_request


def configuration(seed):
    experiment = read_json(CONFIG_PATH)
    settings = reference_configuration(seed)
    settings.update(experiment_id=f"{experiment['experiment_id']}/seed_{seed}",
                    output=f"{experiment['output']}/seed_{seed}",
                    conditions={"08/math": settings["conditions"]["08/math"]})
    settings["sampling"]["08"]["temperature"] = experiment["temperature"]
    return settings


def generation_identity(settings):
    paths = (CONFIG_PATH, Path(__file__), Path(scoped_runner.__file__), Path("scripts/recover_benchmark_recommended.py"))
    return {"sources": {path.as_posix(): runner.sha(path) for path in paths}, "settings": settings}


def generate(seed):
    settings = configuration(seed)
    protocol = {model: read_json(path / "protocol.json") for model, path in runner.OLD.items()}
    original = protocol["08"]["identity"]
    original["solver"]["seed"] = seed
    config = dict(original["config"], longbench_final_max_tokens=settings["longbench"]["max_tokens"],
                  longbench_answer_grammar=settings["longbench"]["grammar"])
    runner.preflight(settings, protocol, config)
    path = Path(settings["output"]) / "temperature_implementation.json"
    identity = generation_identity(settings)
    if path.exists():
        assert read_json(path) == identity
    else:
        write_json(path, identity)

    def request_for_seed(source, context_hash):
        body, path = _archived_request(source, context_hash)
        return (dict(body, seed=seed) if body is not None else None), path

    runner.prepare = scoped_runner.prepare
    runner.case = scoped_runner.one_case
    runner.archived_request = request_for_seed
    workflow.structured = selection.structured = scoped_runner.structured_unicode
    runner.run(settings, protocol, config, select=None)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = cli.parse_args()
    generate(args.seed)
