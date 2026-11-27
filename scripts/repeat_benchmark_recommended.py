"""Run additional paired seeds against the frozen recommended-parameter benchmark inputs."""
from argparse import ArgumentParser
from pathlib import Path

from repar.io import read_json, write_json
from repar.v4 import selection, workflow

import resume_benchmark_recommended as resume
import run_benchmark_recommended as runner


SEEDS = (42, 43, 44)
REPEAT_ROOT = Path("artifacts/benchmark_recommended_20261123_repeats")
RECOVERED_QID = "66f00f1a821e116aacb252cf"
_archived_request = runner.archived_request


def configuration(seed):
    settings = read_json(runner.CONFIG_PATH)
    if seed != SEEDS[0]:
        settings.update(output=(REPEAT_ROOT / f"seed_{seed}").as_posix(),
                        experiment_id=f"{settings['experiment_id']}/seed_{seed}")
    return dict(settings, answer_seed=seed)


def recovered_request():
    original = configuration(SEEDS[0])
    condition = Path(original["output"]) / "08/longbench/raptor"
    record = read_json(condition / "cases" / f"{RECOVERED_QID}.json")
    assert record["historical_evidence_file"]
    return record, condition / "cache/solver"


def prepare_main(seed):
    settings = configuration(seed)
    protocol = {model: read_json(path / "protocol.json") for model, path in runner.OLD.items()}
    config = dict(protocol["08"]["identity"]["config"],
                  longbench_final_max_tokens=settings["longbench"]["max_tokens"],
                  longbench_answer_grammar=settings["longbench"]["grammar"])
    for entry in protocol.values():
        entry["identity"]["solver"]["seed"] = seed
    runner.preflight(settings, protocol, config)
    record, cache = recovered_request()
    recovery = cache / (record["calls"][-1]["key"] + ".json")
    frozen = {path.as_posix(): runner.sha(path) for path in (
        Path(__file__), Path(resume.__file__), Path("scripts/recover_benchmark_recommended.py"), recovery,
    )}
    manifest = Path(settings["output"]) / "repeat_implementation.json"
    if manifest.exists():
        assert read_json(manifest) == frozen
    else:
        write_json(manifest, frozen)

    def prepare(settings, protocol, config):
        data, indexes = resume.prepare(settings, protocol, config)
        data["sources"][("08", "longbench", "raptor")][RECOVERED_QID] = (record, cache)
        return data, indexes

    def archived_request(source, context_sha):
        body, path = _archived_request(source, context_sha)
        return (dict(body, seed=seed) if body is not None else None), path

    runner.prepare = prepare
    runner.archived_request = archived_request
    runner.case = resume.one_case
    workflow.structured = selection.structured = resume.structured_unicode
    return settings, protocol, config


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("generate", "grade"))
    cli.add_argument("--seed", type=int, choices=SEEDS[1:], required=True)
    cli.add_argument("--group", action="append", choices=("08/math", "08/longbench", "27/math", "27/longbench"))
    args = cli.parse_args()
    settings, protocol, config = prepare_main(args.seed)
    if args.phase == "generate":
        runner.run(settings, protocol, config, args.group)
    else:
        runner.grade(settings, protocol, config)


if __name__ == "__main__":
    main()
