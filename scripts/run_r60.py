"""Build dense indexes and evaluate r60 or r61 chunk-budget conditions."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from hashlib import file_digest
from pathlib import Path

from repar.benchmarks.math_exam import public_question, stitch_context, task_text
from repar.benchmarks.v3_suite import freeze, sha, unique
from repar.benchmarks.v4_suite import V4Suite
from repar.dense_index import DenseKnowledgeIndex, INDEX_VERSION, load_encoder
from repar.evaluation import extract_answer, qa_text
from repar.io import append_jsonl, load_public_samples, public_item, read_json, read_jsonl, write_json


R27 = Path("artifacts/repar_v4_r27_20261120/protocol.json")
ROOT = Path("artifacts/repar_v4_r60_20261126")
MODEL = Path("models/EmbeddingGemma-300M")
SEEDS = (42, 43, 44)


def settings(size, seed, version="r60"):
    old = read_json(R27)["identity"]
    root = ROOT if version == "r60" else Path("artifacts/repar_v4_r61_20261126")
    solver = {**old["solver"], "agent_version": f"react-v4-{version}", "seed": seed, "index_version": INDEX_VERSION,
              "deployment_id": f"{old['solver']['deployment_id']}/{version}-{size}",
              "graph_model": "auto", "chunk_max_tokens": size,
              "embedding": {"path": MODEL.as_posix(), "manifest_sha256": sha(MODEL/"source_manifest.json"),
                            "build_device": "cuda", "query_device": "cuda", "batch_size": 16},
              "source_policy": "Original text partitioned by EmbeddingGemma, spaCy entity graph",
              "notes": "Original r27 control with embedding-based ordered tree retrieval"}
    solver.pop("chunk_words")
    if version == "r61":
        solver["notes"] = "r60 indexes with a gap-focused query prompt and exact scoped cosine in hybrid mode"
    directory = ROOT/f"tokens_{size}"/"indexes"
    source = old["config"]
    config = {k: source[k] for k in ("math_questions", "math_textbook", "math_parent", "longbench_data",
              "longbench_parent", "longbench_final_max_tokens", "longbench_answer_grammar", "judge")}
    config.update({
        "experiment_id": f"repar-v4-{version}-{size}-seed-{seed}",
        "output": (root/f"tokens_{size}"/f"seed_{seed}").as_posix(),
        "dense_index_directory": directory.as_posix(),
        "variant": f"bound_views_v4_{version}", "method_display": f"v4-{version} ({size} tokens, seed {seed})",
        "source_policy": solver["source_policy"],
        "exposure": "Development comparison on previously evaluated benchmarks.",
        "report_filename": (f"2026-11-26_0327_r60_{size}_seed_{seed}.md" if version == "r60" else
                            f"2026-11-26_1038_r61_{size}_seed_{seed}.md"),
        "additional_snapshot_files": ["scripts/run_r60.py", "scripts/run_repar_v4.py", "scripts/research_kb_v4.py"],
        "immutable_references": [R27.as_posix(), (MODEL/"source_manifest.json").as_posix(),
                                 (directory/"*.pkl").as_posix()],
    })
    if version == "r61":
        config["immutable_references"] += [(directory.parent/"indexes.json").as_posix()]
    configs = Path("configs")/version
    config["solver_config"] = (configs/f"solver_{size}_{seed}.json").as_posix()
    return configs/f"experiment_{size}_{seed}.json", config, solver


def sources(config):
    math, _ = stitch_context(config["math_textbook"])
    lb = load_public_samples(config["longbench_data"])
    if len(lb) != 125 or len({row["_id"] for row in lb}) != 125:
        raise ValueError("Expected complete and unique LongBench Multi-Document inputs")
    return math, lb


def build_graph_and_save(index, path):
    index._build_graph()
    index.save(path)
    return index.stats


def build_indexes(config, solver, graph_workers):
    math, lb = sources(config)
    contexts = list(dict.fromkeys([math, *(r["context"] for r in lb)]))
    directory = config["dense_index_directory"]
    summary = []
    pending = deque()

    def record():
        path, reused, result = pending.popleft()
        stats = result if reused else result.result()
        summary.append({"path": path.as_posix(), "reused": reused, **stats})
        print("INDEX", solver["chunk_max_tokens"], len(summary), "/", len(contexts), stats["leaves"], flush=True)

    with ProcessPoolExecutor(graph_workers) as pool:
        for context in contexts:
            path = DenseKnowledgeIndex.cache_path(context, solver, directory)
            reused = path.is_file()
            if reused:
                result = DenseKnowledgeIndex.cached(context, solver, directory, build=False).stats
            else:
                index = DenseKnowledgeIndex(context, solver, build_graph=False)
                result = pool.submit(build_graph_and_save, index, path)
            pending.append((path, reused, result))
            if len(pending) >= 2 * graph_workers:
                record()
        while pending:
            record()
    write_json(Path(directory).parent/"indexes.json", summary)


def lock_condition(path, config, solver):
    for target, value in ((Path(config["solver_config"]), solver), (path, config)):
        if target.is_file():
            if read_json(target) != value:
                raise ValueError(f"Existing experiment settings changed: {target}")
        else:
            write_json(target, value)
    freeze(path)
    return V4Suite(path)


def check_run_index(suite, index, metadata):
    path = Path(metadata["path"])
    with path.open("rb") as stream:
        checksum = file_digest(stream, "sha256").hexdigest()
    if suite.protocol["identity"]["references"][path.as_posix()] != checksum:
        raise ValueError(f"Frozen index changed: {path}")
    if index.context_hash != metadata["context_sha256"]:
        raise ValueError("Source hash mismatch")


def evaluate(suite, workers, grading_pool):
    config = suite.config
    root = suite.output
    math, lb = sources(config)
    questions = {q["id"]: public_question(q) for q in read_jsonl(config["math_questions"])}
    if len(questions) != 19:
        raise ValueError("Expected 19 distinct MATH questions")
    index, metadata = suite.load_index(math, "math")
    check_run_index(suite, index, metadata)
    # Transformers initializes weights in a temporary device context. Finish
    # loading once before inference threads can enter that context together.
    load_encoder(*index.encoder_settings())
    write_json(root/"math/index_reuse.json", metadata)
    grade = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        math_path = root/"math/predictions.jsonl"
        math_done = unique(math_path)
        if not set(math_done) <= questions.keys() or any(r["status"] != "ok" for r in math_done.values()):
            raise ValueError("Invalid stored math predictions")

        def predict_math(item):
            qid, question = item
            row = suite.generate(qid, task_text(question), index, "math")
            return {**row, "question": question}

        for count, row in enumerate(pool.map(predict_math, (pair for pair in questions.items() if pair[0] not in math_done)),
                                    len(math_done)+1):
            if row["status"] != "ok":
                raise RuntimeError(row["error"])
            append_jsonl(math_path, row)
            print("MATH", root, count, "/ 19", flush=True)
        unique(math_path, expected=questions)
        suite.lock_predictions("math")
        grade = grading_pool.submit(suite.grade_math)

        lb_path = root/"longbench/predictions.jsonl"
        lb_done = unique(lb_path)
        labels = {item["_id"]: item["answer"] for item in lb}
        if not set(lb_done) <= labels.keys() or any(r["status"] != "ok" for r in lb_done.values()):
            raise ValueError("Invalid stored LongBench predictions")

        def predict_longbench(item):
            qid = item["_id"]
            index, metadata = suite.load_index(item["context"], "longbench")
            check_run_index(suite, index, metadata)
            row = suite.generate(qid, qa_text(item), index, "longbench", longbench_item=item)
            pred = extract_answer(row["response"])
            return {**row, "index_reuse": metadata, "pred": pred, "gold": labels[qid], "correct": pred == labels[qid]}

        public = (public_item(item) for item in lb if item["_id"] not in lb_done)
        for count, row in enumerate(pool.map(predict_longbench, public), len(lb_done)+1):
            if row["status"] != "ok" or row["pred"] is None:
                raise RuntimeError(f"Unsuccessful LongBench task {row['id']}")
            append_jsonl(lb_path, row)
            print("LONGBENCH", root, count, "/ 125", flush=True)
        unique(lb_path, expected=labels)
        suite.lock_predictions("longbench")
    return suite, grade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("r60", "r61"), default="r60")
    parser.add_argument("--sizes", type=int, nargs="+", default=[192, 384, 768, 1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, choices=SEEDS)
    parser.add_argument("--phase", choices=("index", "evaluate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--graph-workers", type=int, default=6)
    args = parser.parse_args()
    if min(args.workers, args.graph_workers) < 1 or any(size < 1 for size in args.sizes):
        parser.error("Worker and token budgets must be positive")
    if args.version == "r61" and args.phase != "evaluate":
        parser.error("r61 evaluates the frozen r60 indexes with --phase evaluate")
    with ThreadPoolExecutor(max_workers=1) as grading_pool:
        reports = []
        for size in args.sizes:
            _, config, solver = settings(size, args.seeds[0], args.version)
            if args.phase in {"index", "all"}:
                build_indexes(config, solver, args.graph_workers)
            if args.phase in {"evaluate", "all"}:
                for seed in args.seeds:
                    suite = lock_condition(*settings(size, seed, args.version))
                    reports.append(evaluate(suite, args.workers, grading_pool))
        for suite, grade in reports:
            grade.result()
            suite.report()


if __name__ == "__main__":
    main()
