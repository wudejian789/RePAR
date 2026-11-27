"""Expose local frozen research corpora as optional Studio knowledge bases."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
import warnings

from repar.io import read_json

from .frozen_data import file_digest


def example_libraries(repo):
    """Discover optional saved examples; incomplete archives never block startup."""
    try:
        return _example_libraries(Path(repo))
    except (OSError, ValueError, KeyError, TypeError) as error:
        warnings.warn(f"跳过不可用的本地示例知识库：{error}", RuntimeWarning, stacklevel=2)
        return {}


def _example_libraries(repo):
    protocol = repo / "artifacts/repar_v5_20261126/seed_42/protocol.json"
    overview = repo / "webserver/data/overview.json"
    if not protocol.is_file() or not overview.is_file():
        return {}
    identity = read_json(protocol)["identity"]
    refs = identity["references"]
    manifest_name = (Path(identity["config"]["dense_index_directory"]).parent / "indexes.json").as_posix()
    manifest = repo / manifest_name
    if not manifest.is_file():
        return {}
    if file_digest(manifest) != refs[manifest_name]:
        raise ValueError("工作台示例的冻结索引清单校验失败。")
    indexes = {item["context_sha256"]: item for item in read_json(manifest)}
    definitions = []
    math_path = protocol.parent / "math/index_reuse.json"
    if math_path.is_file():
        math = read_json(math_path)["context_sha256"]
        definitions.append(("math", "MATH 2025 · 数学教材", "数学教材语料", math))
    for case in read_json(overview)["cases"]:
        definitions.append(("longbench", f"#{case['number']} · {case['sub_domain']}",
                            f"{case['id']}.txt", case["context_sha256"]))

    examples = {}
    for group, title, source, context_sha in definitions:
        stats = indexes.get(context_sha)
        if not stats:
            continue
        path = repo / stats["path"]
        checksum = refs.get(stats["path"])
        if not checksum or not path.is_file():
            continue
        library_id = uuid5(NAMESPACE_URL, f"repar:studio:{group}:{source}:{checksum}").hex
        library = {
            "id": library_id, "name": title, "collection": group, "status": "ready",
            "sources": [{"name": source, "start": 0, "end": stats["characters"]}],
            "stats": {key: stats[key] for key in ("leaves", "entities", "relations", "characters", "version")},
        }
        examples[library_id] = (library, path, checksum, context_sha)
    return examples
