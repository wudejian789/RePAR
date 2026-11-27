"""Audit and summarize the recommended-generation rerun."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from repar.benchmarks.math_exam import aggregate_grades, score_choice
from repar.benchmarks.math_judge import judge_messages, validate_grade
from repar.benchmarks.math_suite import usage
from repar.evaluation import extract_answer, official_prompt
from repar.io import digest, load_public_samples, public_item, read_json, read_jsonl, write_json
from repar.text import terms as standard_terms

from run_benchmark_recommended import CONFIG_PATH, OLD, check_final, sha
from repeat_benchmark_recommended import SEEDS, REPEAT_ROOT, configuration


METHODS = {"direct": "直接回答", "bm25": "BM_RAG", "raptor": "RAPTOR", "hipporag2": "HippoRAG2",
           "hypergraphrag": "HyperGraphRAG", "repar": "RePAR v4-r27"}
REPORT = Path("all_results_new.md")


def historical_cells():
    """Read archived build cost columns from the reference report."""
    groups = {}
    for line in Path("all_results.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## 0.8B"):
            model = "08"
        elif line.startswith("## 27B"):
            model = "27"
        elif line.startswith("### MATH"):
            dataset = "math"
        elif line.startswith("### LongBench"):
            dataset = "longbench"
        elif line.startswith("## 逐题"):
            break
        elif line.startswith("| "):
            columns = [cell.strip() for cell in line.split("|")[1:-1]]
            if columns[0] not in METHODS.values():
                continue
            entry = groups.setdefault((model, dataset, columns[0]), {})
            if len(columns) == 6:
                entry["build_wall"] = columns[2]
            elif len(columns) == 9:
                entry["build_tokens"] = [column.split("/")[0] for column in columns[1:]]
    return groups


def measures(values):
    return {"sum": sum(values), "max": max(values, default=None), "median": median(values) if values else None}


def audit_condition(settings, root, model, dataset, method, questions, judge, strict, *, first_settings=None):
    path = root / model / dataset / method
    rows = {p.stem: read_json(p) for p in sorted((path / "cases").glob("*.json"))}
    assert set(rows) <= set(questions) and all(row["id"] == qid for qid, row in rows.items())
    grade_path = path / "math/grades.jsonl"
    grades = read_jsonl(grade_path) if grade_path.exists() else []
    judge_calls = []
    if grades:
        assert len({g["id"] for g in grades}) == len(grades)
        if len(grades) == len(questions):
            lock = read_json(path / "math/generation_manifest.json")
            assert lock["predictions_sha256"] == sha(path / "math/predictions.jsonl")
        for grade in grades:
            qid = grade["id"]
            question = questions[qid]
            response = rows[qid]["response"]
            assert grade["response_sha256"] == digest(response)
            if question["question_type"] in {"single_choice", "multiple_choice"}:
                expected = score_choice(question, response)
                assert not grade["judge_calls"]
            else:
                expected = aggregate_grades(question, [validate_grade(question, g) for g in grade["grades"]])
                seeds = set()
                messages = judge_messages(question, response)
                for call in grade["judge_calls"]:
                    record = read_json(path / "cache/judge" / (call["key"] + ".json"))
                    body = record["request"]
                    assert record["response"] == call["response"]
                    assert body["model"] == judge["model"] and body["temperature"] == judge["temperature"] == 0
                    assert body["max_tokens"] == judge["max_tokens"]
                    assert body["messages"][:len(messages)] == messages
                    seeds.add(body["seed"])
                assert seeds == set(judge["replicate_seeds"][:len(grade["grades"])])
                judge_calls.extend(grade["judge_calls"])
            assert all(expected[k] == grade[k] for k in ("score", "fully_correct"))
    status = Counter()
    finish = Counter()
    costs = {"input": [], "output": [], "case": [], "http": [], "load": [],
             "prefill_seconds": [], "decode_seconds": [], "server_cached_input_tokens": []}
    cache_hits = 0
    per_question = {}
    solver = read_json(OLD[model] / "protocol.json")["identity"]["solver"]
    solver["seed"] = settings["answer_seed"]
    first_condition = Path((first_settings or configuration(SEEDS[0]))["output"]) / model / dataset / method
    for qid, row in rows.items():
        status[row["status"]] += 1
        calls = row["calls"]
        prompt = [c["response"]["usage"]["prompt_tokens"] for c in calls]
        completion = [c["response"]["usage"]["completion_tokens"] for c in calls]
        costs["input"].extend(prompt)
        costs["output"].extend(completion)
        for call in calls:
            if call["cached"]:
                cache_hits += 1
                continue
            costs["http"].append(call["seconds"])
            timings = call["response"].get("timings", {})
            for source, target in (("prompt_ms", "prefill_seconds"), ("predicted_ms", "decode_seconds")):
                if source in timings:
                    costs[target].append(timings[source] / 1000)
            details = call["response"]["usage"].get("prompt_tokens_details", {})
            if "cached_tokens" in details:
                costs["server_cached_input_tokens"].append(details["cached_tokens"])
        if row["seconds"] is not None:
            costs["case"].append(row["seconds"])
        if "index_load_wall_seconds" in row:
            costs["load"].append(row["index_load_wall_seconds"])
        if calls:
            record = read_json(path / "cache/solver" / (calls[-1]["key"] + ".json"))
            check_final(record["request"], solver, settings, model, dataset)
            assert record["response"] == calls[-1]["response"]
            if method != "repar" and solver["seed"] != SEEDS[0]:
                initial = first_condition / "cases" / f"{qid}.json"
                if initial.exists() or strict:
                    first_call = read_json(initial)["calls"][-1]
                    first_request = read_json(first_condition / "cache/solver" / (first_call["key"] + ".json"))["request"]
                    assert record["request"] == dict(first_request, seed=solver["seed"]), (model, dataset, method, qid)
            if row.get("historical_request"):
                assert record["request"]["messages"] == read_json(Path(row["historical_request"]))["request"]["messages"]
            if row.get("historical_evidence_file"):
                old = read_json(Path(row["historical_evidence_file"]))
                context = "\n\n".join(f"[{e['id']}]\n{e['text']}" for e in old["evidence"])
                restored_prompt = official_prompt(public_item(questions[qid]), context, rag=True)
                assert record["request"]["messages"] == [{"role": "user", "content": restored_prompt}]
            for c in calls[:-1]:
                control = read_json(path / "cache/solver" / (c["key"] + ".json"))["request"]
                expected_sampling = settings["sampling"][model] if dataset == "math" else {"temperature": settings["longbench"]["temperature"]}
                assert all(control[key] == value for key, value in expected_sampling.items())
                assert control["seed"] == solver["seed"] and control["response_format"]["type"] == "json_schema"
            if method == "repar" and dataset == "longbench" and row.get("query_tokenizer") != "historical":
                for step in row["trace"]:
                    for execution in step.get("executed", []):
                        if execution["name"] == "tree_search":
                            params = execution["parameters"]
                            for key in ("query", "anchor_query"):
                                terms = standard_terms(params.get(key, ""))
                                assert all(not any("\u3400" <= c <= "\u9fff" for c in term) for term in terms), (qid, key)
            finish[calls[-1]["response"]["choices"][0]["finish_reason"]] += 1
        else:
            assert row["status"] == "historical_index_failure" and row.get("historical_error")
            if solver["seed"] != SEEDS[0]:
                initial = first_condition / "cases" / f"{qid}.json"
                if initial.exists() or strict:
                    original = read_json(initial)
                    assert original["status"] == row["status"] and original["historical_error"] == row["historical_error"]
        if method == "repar" and dataset == "math":
            assert row["index_reuse"]["sha256"] == sha(Path(settings["chinese_index"]))
        if dataset == "longbench":
            if method == "repar":
                index = row["index_reuse"]
                assert index["reused"] and index["stats"]["extractor"]["model"] == "en_core_web_sm"
                assert Path(index["path"]).is_file() and index["context_sha256"] == digest(questions[qid]["context"])
            assert row["pred"] == extract_answer(row["response"])
            assert row["correct"] == (row["pred"] == questions[qid]["answer"])
        per_question[qid] = {"input": sum(prompt), "output": sum(completion), "seconds": row["seconds"],
                             "calls": len(calls), "status": row["status"]}
    complete = len(rows) == len(questions) and (dataset == "longbench" or len(grades) == len(questions))
    if strict and not complete:
        raise ValueError(f"{model}/{dataset}/{method} has {len(rows)}/{len(questions)} predictions and {len(grades)} grades")
    return {"count": len(rows), "expected": len(questions), "complete": complete,
            "score": (sum(g["score"] for g in grades) if dataset == "math" else sum(r["correct"] for r in rows.values())) if complete else None,
            "full": sum(g["fully_correct"] for g in grades) if complete and dataset == "math" else None,
            "calls": len(costs["input"]), "cache_hits": cache_hits, "grade_count": len(grades),
            "judge_usage": usage(judge_calls), "judge_disagreements": sum(len(g["grades"]) == 3 for g in grades),
            "case_mean": mean(costs["case"]) if costs["case"] else None, "statuses": dict(status),
            "answer_finish": dict(finish), "costs": {k: measures(v) for k, v in costs.items()},
            "per_question": per_question}


def cell(value):
    if value is None:
        return "—"
    return f"{value:,}" if isinstance(value, int) else f"{value:,.2f}"


def render(settings, summary, build, *, math_judge_note=None, math_generation_note=None):
    complete = sum(r["complete"] for methods in summary.values() for r in methods.values())
    lines = ["# 两种基模按推荐推理设置重测", "",
             f"更新时间为 {datetime.now().astimezone():%Y-%m-%d %H:%M}（Asia/Shanghai）。种子 {settings['answer_seed']}，{complete}/16 个条件完成核验。", "",
             "- MATH 全题型使用自由文本及 4,096-token 最终上限。"
             + (math_generation_note or "0.8B 和 27B 作答分别使用模型卡推荐采样。")
             + (math_judge_note or "非选择题沿用 27B 温度 0 的匿名阅卷。"),
             "- LongBench 的温度 0.1 与 128-token 最终上限来自官方脚本。已测方法统一使用四选一 GBNF 和非思考模式。其余采样沿用冻结服务的默认值。0.8B／27B 的 `top_k` 分别为 40／20，两者的 `top_p`、`min_p`、`presence_penalty`、`repeat_penalty` 分别为 0.95、0.05、0、1。",
             "- 两个底座的 r27 MATH 使用同一中文索引。其余条件复用历史建库产物。基线复用归档的最终证据或输入，r27 重新执行在线编排和检索。",
             "- 建库耗时及建库 token 为历史成本。作答耗时为本轮逐题墙钟，包含请求排队。基线从已保存输入开始计时，r27 包括在线控制和检索。",
             "- 建库或检索失败导致历史证据不可用时，按原协议计入 LongBench 分母，单独报告缺少新答题请求的数量。",
             "- 统计按逻辑调用的 usage 汇总，阅卷另计。表中缓存命中为客户端整次请求复用。服务返回的预填充与解码时间、前缀缓存 token 另见机器汇总。未测组合留空，未完成条件只列覆盖。",
             "", "[冻结配置](configs/benchmark_recommended_20261123.json) · "
             f"[部署与协议]({settings['output']}/protocol.json) · "
             f"[{'原作答用量' if math_judge_note else '逐题与调用统计'}]({settings['output']}/summary.json)", ""]
    for group, methods in summary.items():
        model, dataset = group.split("/")
        heading = ("0.8B" if model == "08" else "27B") + " · " + ("MATH 2025 NATIONAL I" if dataset == "math" else "LongBench v2 Multi-Document QA")
        lines.extend([f"## {heading}", "", "| 方法 | 成绩 | 进度 | 无可复用证据 | 建库历史秒数 | 作答秒均值 | 作答秒中位数 |",
                      "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
        for method, name in METHODS.items():
            if method not in methods:
                lines.append(f"| {name} | | | | | | |")
                continue
            row = methods[method]
            score = "待完成"
            if row["complete"]:
                score = f"{row['score']:.3f}/150（{row['full']}/19 全对）" if dataset == "math" else f"{row['score']}/125（{row['score']/125:.1%}）"
            progress = f"{row['count']}/{row['expected']}"
            if dataset == "math":
                progress += f"，已评 {row['grade_count']}"
            build_wall = "158.36（中文索引，共享）" if dataset == "math" and method == "repar" else build[(model, dataset, name)]["build_wall"]
            lines.append(f"| {name} | {score} | {progress} | {row['statuses'].get('historical_index_failure', 0)} | "
                         f"{build_wall} | {cell(row['case_mean'])} | {cell(row['costs']['case']['median'])} |")
        lines.extend(["", "| 方法 | 建库／作答调用 | 输入 token 建库／作答 | 输出 token 建库／作答 | 单次输入最大 建库／作答 | 单次输出最大 建库／作答 | 单次输入中位 建库／作答 | 单次输出中位 建库／作答 | 建库／作答缓存命中 | 最终触及上限 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for method, name in METHODS.items():
            if method not in methods:
                lines.append(f"| {name} | | | | | | | | | |")
                continue
            row = methods[method]
            b = build[(model, dataset, name)]["build_tokens"]
            inputs, outputs = (row["costs"][key] for key in ("input", "output"))
            lines.append(f"| {name} | {b[0]}/{row['calls']:,} | {b[1]}/{inputs['sum']:,} | {b[2]}/{outputs['sum']:,} | "
                         f"{b[3]}/{cell(inputs['max'])} | {b[4]}/{cell(outputs['max'])} | "
                         f"{b[5]}/{cell(inputs['median'])} | {b[6]}/{cell(outputs['median'])} | "
                         f"{b[-1]}/{row['cache_hits']} | {row['answer_finish'].get('length', 0)} |")
        if dataset == "math":
            lines.extend(["", "| 方法 | 阅卷调用 | 阅卷输入 token | 阅卷输出 token | 阅卷调用累计秒数 | 阅卷缓存命中 | 触发第三次阅卷的题数 |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
            for method, row in methods.items():
                judge_use = row["judge_usage"]
                lines.append(f"| {METHODS[method]} | {judge_use['calls']} | {judge_use['prompt_tokens']:,} | "
                             f"{judge_use['completion_tokens']:,} | {cell(judge_use['recorded_seconds'])} | "
                             f"{judge_use['cache_hits']} | {row['judge_disagreements']} |")
        lines.extend(["", "<details><summary>逐题输入／输出 tokens</summary>", "",
                      "| 题目 ID | " + " | ".join(METHODS.values()) + " |",
                      "| --- |" + " --- |" * len(METHODS)])
        qids = dict.fromkeys(qid for row in methods.values() for qid in row["per_question"])
        if dataset == "math":
            qids = sorted(qids, key=lambda qid: int(qid.rsplit("_Q", 1)[1]))
        for qid in qids:
            cells = []
            for method in METHODS:
                item = methods.get(method, {}).get("per_question", {}).get(qid)
                cells.append(f"{item['input']:,} / {item['output']:,}" if item else "")
            lines.append(f"| {qid} | " + " | ".join(cells) + " |")
        lines.extend(["", "</details>", ""])
    return lines


def render_repeats(summaries, seed_details, *, math_judge_note=None, summary_path=None, temperature_1_means=None):
    completed = sum(row["complete"] for summary in summaries.values() for methods in summary.values() for row in methods.values())
    records = sum(row["count"] for summary in summaries.values() for methods in summary.values() for row in methods.values())
    lines = ["# 两种基模按推荐推理设置重测 · 三次均值", "",
             f"更新时间为 {datetime.now().astimezone():%Y-%m-%d %H:%M}（Asia/Shanghai）。{completed}/48 次条件运行完成核验，已记录 {records:,}/3,774 条逐题结果。", "",
             "每个条件分别使用种子 42、43、44。三次分别完成核验后公布算术均值。MATH 满分 150 分，LongBench 各轮按 125 题计算准确率。"
             + (math_judge_note or "建库产物及评审协议沿用本轮冻结设置。"), "",
             "作答累计秒数是逐题墙钟之和，包含排队。27B LongBench 的直接回答和 BM_RAG 按题连续提交三个种子，后续请求可复用服务端提示前缀。各轮实际前缀缓存 token、预填充和解码时间见机器汇总。", "",
             f"[三轮机器汇总]({summary_path or (REPEAT_ROOT / 'summary.json').as_posix()}) · "
             "[作答运行维护日志](docs/2026-11-23_2109_基模推荐参数评测重跑.md)", ""]
    means = {}
    for group, first_methods in summaries[SEEDS[0]].items():
        model, dataset = group.split("/")
        heading = ("0.8B" if model == "08" else "27B") + " · " + ("MATH 2025 NATIONAL I" if dataset == "math" else "LongBench v2 Multi-Document QA")
        show_temperature_reference = temperature_1_means is not None and group == "08/math"
        labels = ["方法", *(["温度 1.0 均分"] if show_temperature_reference else []),
                  "seed 42", "seed 43", "seed 44", "三次均值", "平均作答累计秒数", "平均作答输入／输出 token"]
        lines.extend([f"## {heading}", "", "| " + " | ".join(labels) + " |",
                      "| " + " | ".join(["---", *["---:" for _ in labels[1:]]]) + " |"])
        means[group] = {}
        def score(value):
            if dataset == "math":
                return f"{value:.3f}/150"
            return f"{value:.2f}/125（{value/125:.2%}）"

        for method in first_methods:
            rows = [summaries[seed].get(group, {}).get(method, {}) for seed in SEEDS]
            values = [score(row["score"]) if row.get("complete") else "待完成" for row in rows]
            if show_temperature_reference:
                values.insert(0, score(temperature_1_means[method]))
            average = {}
            if all(row.get("complete") for row in rows):
                average = {"score": mean(row["score"] for row in rows),
                           **{key: mean(row["costs"][key]["sum"] for row in rows) for key in ("case", "input", "output")}}
                if dataset == "math":
                    average["fully_correct"] = mean(row["full"] for row in rows)
            means[group][method] = average or None
            mean_score = score(average["score"]) if average else "—"
            if average and dataset == "math":
                mean_score += f"（平均 {average['fully_correct']:.2f}/19 全对）"
            avg_seconds = cell(average["case"]) if average else "—"
            avg_tokens = f"{cell(average['input'])} / {cell(average['output'])}" if average else "—"
            lines.append(f"| {METHODS[method]} | {' | '.join(values)} | {mean_score} | {avg_seconds} | {avg_tokens} |")
        lines.append("")
    for seed, detail in seed_details.items():
        lines.extend([f"## seed {seed} 逐轮统计", "", *["#" + line if line.startswith("## ") else line for line in detail[2:]], ""])
    return lines, means


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--allow-partial", action="store_true")
    cli.add_argument("--single-seed", action="store_true")
    args = cli.parse_args()
    settings = read_json(CONFIG_PATH)
    config = read_json(OLD["08"] / "protocol.json")["identity"]["config"]
    datasets = {"math": {r["id"]: r for r in read_jsonl(config["math_questions"])},
                "longbench": {r["_id"]: r for r in load_public_samples(config["longbench_data"])}}
    seeds = SEEDS[:1] if args.single_seed else SEEDS
    summaries = {}
    details = {}
    build = historical_cells()
    for seed in seeds:
        settings = configuration(seed)
        root = Path(settings["output"])
        protocol = root / "protocol.json"
        if not protocol.exists() and args.allow_partial:
            summaries[seed] = {}
            continue
        frozen = read_json(protocol)
        expected_config = dict(settings)
        if seed == SEEDS[0]:
            del expected_config["answer_seed"]
        assert frozen["identity"]["configuration"] == expected_config
        for path, expected in frozen["identity"]["files"].items():
            assert sha(Path(path)) == expected, path
        summary = {}
        for group, methods in settings["conditions"].items():
            model, dataset = group.split("/")
            summary[group] = {method: audit_condition(settings, root, model, dataset, method, datasets[dataset],
                                                      frozen["judge"], not args.allow_partial) for method in methods}
        write_json(root / "summary.json", summary)
        summaries[seed] = summary
        details[seed] = render(settings, summary, build)
    if args.single_seed:
        report_path = Path(settings["output"]) / "2026-11-23_2109_seed_42.md"
        lines = [line.replace("](artifacts/", "](../../artifacts/").replace("](configs/", "](../../configs/")
                 for line in details[SEEDS[0]]]
    else:
        report_path = REPORT
        if not args.allow_partial:
            assert sum(row["count"] for summary in summaries.values() for methods in summary.values()
                       for row in methods.values()) == 3774
        lines, means = render_repeats(summaries, details)
        write_json(REPEAT_ROOT / "summary.json", {"seeds": SEEDS, "means": means, "runs": summaries})
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)
    completed = sum(r["complete"] for summary in summaries.values() for methods in summary.values() for r in methods.values())
    print(f"{completed}/{16 * len(seeds)} condition runs audited")


if __name__ == "__main__":
    main()
