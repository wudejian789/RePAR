"""Freeze, run, and audit a versioned 0.8B v4 candidate on both benchmarks."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from shutil import copy2
from statistics import mean

import requests

from repar.benchmarks.math_suite import usage
from repar.io import load_public_samples, read_json, write_json
from repar.v4 import workflow
from repar.v4.knowledge import request_identity

import run_benchmark_r31 as suite
from report_benchmark_recommended import audit_condition
from report_math_r28 import audit_grade, saved_request


BASELINE = Path("artifacts/benchmark_r31_08_20261124")


class Candidate:
    def __init__(self, path):
        self.path = path
        self.experiment = read_json(path)
        self.root = Path(self.experiment["output"])
        self.agent = read_json(self.experiment["agent_config"])

    def configuration(self, seed):
        suite.CONFIG = self.path
        return suite.configuration(seed)

    def implementation(self):
        suite.CONFIG = self.path
        return {**suite.implementation_identity(), Path(__file__).resolve().as_posix(): suite.runner.sha(Path(__file__))}

    def prepare(self, seed):
        settings = self.configuration(seed)
        protocol = read_json(suite.runner.OLD[suite.MODEL] / "protocol.json")
        protocol["identity"]["solver"].update(self.agent, seed=seed)
        config = dict(protocol["identity"]["config"], longbench_final_max_tokens=settings["longbench"]["max_tokens"],
                      longbench_answer_grammar=settings["longbench"]["grammar"])
        protocols = {suite.MODEL: protocol}
        suite.runner.preflight(settings, protocols, config)
        folder = Path(settings["output"])
        route = self.agent.get("reflection_service")
        if route:
            response = requests.get(route["base_url"] + "/props", timeout=15)
            response.raise_for_status()
            props = response.json()
            assert props["model_alias"] == self.agent["model"]
            base_server = protocol["deployments"]["solver"]["props"]
            assert all(props[k] == base_server[k] for k in ("build_info", "model_path", "chat_template"))
            frozen = {k: props[k] for k in ("model_alias", "model_path", "build_info", "total_slots",
                                            "default_generation_settings", "chat_template")}
            path = folder / "reflection_service.json"
            identity = {"route": route, "props": frozen}
            if path.exists():
                assert read_json(path) == identity
            else:
                write_json(path, identity)
        manifest = folder / "implementation.json"
        expected = self.implementation()
        if manifest.exists():
            assert read_json(manifest) == expected
        else:
            write_json(manifest, expected)
        identity = read_json(folder / "protocol.json")["identity"]
        workspace = Path.cwd().resolve()
        for name, checksum in {**expected, **identity["files"], **identity["core"]}.items():
            source = Path(name).resolve()
            relative = source.relative_to(workspace)
            if source.suffix != ".py" and not relative.is_relative_to("configs"):
                continue
            destination = self.root / "source" / relative
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                copy2(source, destination)
            assert suite.runner.sha(destination) == checksum, destination
        return settings, protocols, config

    def audit(self, seeds, datasets=None):
        datasets = datasets or suite.DATASETS
        math, judge, deployment = suite.judging.questions_and_judge()
        frozen = read_json(suite.runner.OLD[suite.MODEL] / "protocol.json")["identity"]
        longbench = {row["_id"]: row for row in load_public_samples(frozen["config"]["longbench_data"])}
        assert len(math) == 19 and len(longbench) == 125
        results = {}
        for seed in seeds:
            settings = self.configuration(seed)
            root = Path(settings["output"])
            manifest = read_json(root / "protocol.json")["identity"]
            assert manifest["configuration"] == settings
            implementation = read_json(root / "implementation.json")
            workspace = Path.cwd().resolve()
            for name, checksum in {**implementation, **manifest["files"], **manifest["core"]}.items():
                current = Path(name).resolve()
                relative = current.relative_to(workspace)
                snapshot = self.root / "source" / relative
                document = snapshot if current.suffix == ".py" or relative.is_relative_to("configs") else current
                assert suite.runner.sha(document) == checksum, document
            judge_record = read_json(root / "08_judge_protocol.json")
            assert judge_record["deployment"] == deployment
            assert all(judge_record["config"][k] == v for k, v in judge.items() if k != "deployment_id")
            for name, checksum in judge_record["files"].items():
                assert suite.runner.sha(Path(name)) == checksum, name
            if route := self.agent.get("reflection_service"):
                assert read_json(root / "reflection_service.json")["route"] == route
            results[seed] = {}
            for dataset, questions in (("math", math), ("longbench", longbench)):
                if dataset not in datasets:
                    continue
                results[seed][dataset] = self.audit_run(settings, dataset, questions, judge)
                score = results[seed][dataset]["score"]
                print(f"{seed} {dataset} {score}/{150 if dataset == 'math' else 125}", flush=True)
        if len(results) == 3:
            destination = self.root / "summary.json"
            results["mean"] = {
                dataset: mean(run[dataset]["score"] for run in results.values())
                for dataset in datasets
            }
        else:
            suffix = "_".join(datasets)
            destination = self.root / f"summary_seed_{seeds[0]}_{suffix}.json"
        write_json(destination, results)

    def audit_run(self, settings, dataset, questions, judge):
        model = suite.MODEL
        seed = settings["answer_seed"]
        root = Path(settings["output"])
        relative = Path(f"{model}/{dataset}/repar")
        folder = root / relative
        reference = BASELINE / f"seed_{seed}" / relative
        paths = {p.stem: p for p in (folder / "cases").glob("*.json")}
        assert paths.keys() == questions.keys()
        stats = audit_condition(settings, root, model, dataset, "repar", questions, judge, dataset == "longbench")
        math = dataset == "math"
        if math:
            assert {p.stem for p in (folder / "grades").glob("*.json")} == questions.keys()
        metrics, per_case, grades = Counter(), {}, []
        for qid, path in paths.items():
            row = read_json(path)
            prior = read_json(reference / "cases" / path.name)
            assert row["id"] == qid and row["agent_version"] == settings["agent_version"]
            assert row["status"] == "ok"
            assert all(row[k] == prior[k] for k in ("task", "context_sha256", "query_tokenizer", "base_url"))
            assert row["index_reuse"]["path"] == prior["index_reuse"]["path"]
            if math:
                assert row["index_reuse"]["sha256"] == prior["index_reuse"]["sha256"]
            body = saved_request(folder, row)
            assert body == {**saved_request(reference, prior), "messages": body["messages"]}
            if route := self.agent.get("reflection_service"):
                for call in row["calls"][:-1]:
                    body = read_json(folder / "cache/solver" / f"{call['key']}.json")["request"]
                    is_reflection = call["tag"].startswith("v4_reflection_")
                    assert body["chat_template_kwargs"]["enable_thinking"] == (route["enable_thinking"] if is_reflection else False)
                for step in row["trace"]:
                    if "reflection" in step:
                        assert step["reflection_service"] == route["base_url"]
            executions = [e for step in row["trace"] for e in step["executed"]]
            ids = {request_identity(e["name"], e["parameters"]) for e in executions}
            assert len(ids) == len(executions)
            metrics.update(e["name"] for e in executions)
            metrics[f"termination_{row['termination']}"] += 1
            metrics["single_tool_cases"] += len(executions) == 1
            metrics["final_sources"] += len(row["evidence_ids"])
            rating = row.get("correct")
            if math:
                grade = read_json(folder / "grades" / path.name)
                assert grade["id"] == qid
                audit_grade(folder / "cache/judge", grade, questions[qid], row["response"], judge)
                grades.append(grade)
                rating = grade["score"]
            per_case[qid] = {"score": rating, "sources": len(row["evidence_ids"]),
                             "finish": row["calls"][-1]["response"]["choices"][0]["finish_reason"]}
        if math:
            stats.update(complete=True, grade_count=len(grades), full=sum(g["fully_correct"] for g in grades),
                         judge_usage=usage([call for grade in grades for call in grade["judge_calls"]]),
                         judge_disagreements=sum(len(g["judge_calls"]) == 3 for g in grades))
        return {**stats, "score": sum(row["score"] for row in per_case.values()),
                "controls": dict(metrics), "per_case": per_case}


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("config", type=Path)
    cli.add_argument("phase", choices=("freeze", "generate", "grade", "report"))
    cli.add_argument("--seed", type=int, choices=suite.SEEDS, action="append")
    cli.add_argument("--dataset", choices=suite.DATASETS, action="append")
    cli.add_argument("--watch", action="store_true")
    args = cli.parse_args()
    candidate = Candidate(args.config)
    seeds = args.seed or (42,)
    if args.phase == "grade":
        suite.judging.CONFIG = candidate.path
        suite.judging.SEEDS = seeds
        suite.judging.configuration = candidate.configuration
        suite.judging.grade(suite.MODEL, args.watch)
    elif args.phase == "report":
        candidate.audit(seeds, args.dataset)
    else:
        suite.runner.prepare = suite.resume.prepare
        suite.runner.case = suite.resume.one_case
        workflow.structured = suite.resume.structured_unicode
        for seed in seeds:
            settings, protocol, config = candidate.prepare(seed)
            if args.phase == "generate":
                suite.runner.run(settings, protocol, config, select=[f"{suite.MODEL}/{dataset}"
                                 for dataset in (args.dataset or suite.DATASETS)])


if __name__ == "__main__":
    main()
