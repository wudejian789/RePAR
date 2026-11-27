"""Submit each 27B LongBench baseline prompt consecutively across repeat seeds."""
from repar.io import load_public_samples, read_json

import repeat_benchmark_recommended as repeat
import run_benchmark_recommended as runner


def main():
    first_seed, *repeat_seeds = repeat.SEEDS
    active_seed = first_seed
    first = read_json(runner.CONFIG_PATH)
    protocols = {model: read_json(path / "protocol.json") for model, path in runner.OLD.items()}
    first_config = protocols["08"]["identity"]["config"]
    runner.preflight(first, protocols, first_config)
    arguments = {first_seed: (first, protocols, first_config)}
    arguments.update((seed, repeat.prepare_main(seed)) for seed in repeat_seeds)
    questions = {q["_id"]: q for q in load_public_samples(first_config["longbench_data"])}
    assert len(questions) == 125
    data = {"longbench": questions, "sources": {}}
    archived_request = repeat._archived_request

    def seeded_request(source, context_sha):
        body, path = archived_request(source, context_sha)
        return (dict(body, seed=active_seed) if body is not None else None), path

    runner.archived_request = seeded_request
    for method in ("direct", "bm25"):
        data["sources"][("27", "longbench", method)] = runner.historical_rows("27", "longbench", method)
        assert set(data["sources"][("27", "longbench", method)]) == set(questions)
        for qid in questions:
            for seed, (settings, protocol, config) in arguments.items():
                active_seed = seed
                endpoint = protocol["27"]["identity"]["solver"]["base_url"]
                job = ("27", "longbench", method, qid, endpoint)
                _, status = runner.case(job, data, settings, config, protocol, {})
                print("DONE", seed, *job[:4], status, flush=True)


if __name__ == "__main__":
    main()
