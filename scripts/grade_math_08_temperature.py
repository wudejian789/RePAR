"""Grade the 0.8B temperature comparison with the frozen 27B MATH rubric."""
from argparse import ArgumentParser
from pathlib import Path
from time import sleep

import requests

from repar.benchmarks.math_exam import aggregate_grades
from repar.benchmarks.math_judge import grade_once
from repar.io import digest, read_json, read_jsonl, write_json

import regrade_math_2025 as grading
from recover_math_2025_judge_quotes import QuoteRepairClient
from report_math_2025_regrade import score_judged_choice
from resume_math_2025_regrade import choice_observation
from run_benchmark_recommended import OLD, sha
from run_math_08_temperature import CONFIG_PATH, SEEDS, configuration


def questions_and_judge():
    source = read_json(OLD["08"] / "protocol.json")["identity"]["config"]
    manifest = read_json(grading.ROOT / "protocol.json")["identity"]
    question_path = Path(source["math_questions"])
    assert sha(question_path) == manifest["sources"][question_path.as_posix()]
    questions = {item["id"]: item for item in read_jsonl(question_path)}
    assert len(questions) == 19
    return questions, manifest["judge"], manifest["deployment"]


def mark(question, answer, clients, judge):
    if question["question_type"] in grading.CHOICE_TYPES:
        result = grading.judge_choice(question, answer, clients)
        return {**result, **score_judged_choice(question, result["prediction"])}
    grades, calls = [], []
    for i, client in enumerate(clients, start=1):
        start = len(client.calls)
        grades.append(grade_once(question, answer, client, judge["max_tokens"], f"math_open_judge_{i}"))
        calls.extend(client.calls[start:])
        if i == 2 and all(left["credit"] == right["credit"] for left, right in zip(
                sorted(grades[0]["units"], key=lambda unit: unit["id"]),
                sorted(grades[1]["units"], key=lambda unit: unit["id"]))):
            break
    return {"grades": grades, "judge_calls": calls, **aggregate_grades(question, grades)}


def run(watch):
    questions, judge, expected = questions_and_judge()
    response = requests.get(judge["base_url"] + "/props", timeout=15)
    response.raise_for_status()
    actual = response.json()
    for key in ("model_alias", "model_path", "build_info", "total_slots", "default_generation_settings", "chat_template"):
        assert actual[key] == expected[key], key
    experiment = read_json(CONFIG_PATH)
    root = Path(experiment["output"])
    files = (Path(__file__), Path("scripts/regrade_math_2025.py"), Path("scripts/resume_math_2025_regrade.py"),
             Path("scripts/recover_math_2025_judge_quotes.py"), grading.ROOT / "protocol.json")
    manifest = {"judge": judge, "deployment": expected, "files": {path.as_posix(): sha(path) for path in files}}
    path = root / "judge_protocol.json"
    if path.exists():
        assert read_json(path) == manifest
    else:
        write_json(path, manifest)
    grading.choice_observation = choice_observation
    paths = []
    for seed in SEEDS:
        settings = configuration(seed)
        for method in settings["conditions"]["08/math"]:
            folder = Path(settings["output"]) / "08/math" / method
            paths.extend((folder, qid) for qid in questions)
    clients = {}
    while paths:
        waiting = []
        for folder, qid in paths:
            prediction = folder / "cases" / f"{qid}.json"
            grade = folder / "grades" / f"{qid}.json"
            if grade.exists():
                continue
            if not prediction.exists():
                waiting.append((folder, qid))
                continue
            if folder not in clients:
                deployment = f"{judge['deployment_id']}/{experiment['experiment_id']}"
                clients[folder] = [QuoteRepairClient(dict(judge, seed=seed, deployment_id=deployment),
                                                     folder / "cache/judge", call_seconds=180)
                                   for seed in judge["replicate_seeds"]]
            record = read_json(prediction)
            assert record["id"] == qid and record["status"] == "ok"
            answer = record["response"]
            result = mark(questions[qid], answer, clients[folder], judge)
            write_json(grade, {**result, "id": qid, "response_sha256": digest(answer)})
            print("GRADED", folder, qid, result["score"], flush=True)
        if not watch or not waiting:
            break
        paths = waiting
        sleep(5)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--watch", action="store_true")
    run(cli.parse_args().watch)
