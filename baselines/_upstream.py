"""Load immutable, hash-verified upstream modules without eager demo imports."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types


def index_identity(method, context, client, encoder, parameters, working_dir=None):
    """Refuse incompatible or unlabelled persisted indexes, including partial runs."""
    spec = json.loads((Path(__file__).parent / method / 'sources.json').read_text(encoding='utf-8'))
    identity = {
        'adapter_version': 1, 'method': method,
        'context_sha256': hashlib.sha256(context.encode()).hexdigest(),
        'upstream_commit': spec['upstream']['commit'],
        'adapter_sha256': hashlib.sha256((Path(__file__).parent / method / 'pipeline.py').read_bytes()).hexdigest(),
        'model_support_sha256': hashlib.sha256(Path(__import__('repar.model_support', fromlist=['x']).__file__).read_bytes()).hexdigest(),
        'client_implementation_sha256': hashlib.sha256(Path(__import__(type(client).__module__, fromlist=['x']).__file__).read_bytes()).hexdigest(),
        'model_config': client.config, 'encoder': encoder.identity, 'parameters': parameters,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if working_dir is not None:
        target = Path(working_dir)
        target.mkdir(parents=True, exist_ok=True)
        manifest = target / 'repar_index_manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8'))['identity'] != key:
                raise ValueError('Incompatible index identity; use a new working directory')
        elif any(target.iterdir()):
            raise ValueError('Nonempty index directory has no identity; use a new directory')
        else:
            manifest.write_text(json.dumps({'identity': key, 'spec': identity}, ensure_ascii=False, indent=2), encoding='utf-8')
    return key


def validate_inputs(context, chunk_words):
    if not isinstance(context, str) or not context.strip():
        raise ValueError('context must be nonempty text')
    if not isinstance(chunk_words, int) or chunk_words < 1:
        raise ValueError('chunk_words must be a positive integer')


def validate_top_k(top_k):
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer')


def upstream(method, package_path):
    registry = Path(os.environ.get("REPAR_BASELINE_REGISTRY", Path(__file__).parent))
    manifest = json.loads((registry / method / "sources.json").read_text(encoding="utf-8"))
    spec = manifest["upstream"]
    root = Path(os.environ.get("REPAR_REPOSITORY", registry.parent))
    source = root / spec["source_dir"]
    for item in spec["files"]:
        path = source / item["path"]
        path.resolve().relative_to(source.resolve())
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Missing or modified upstream snapshot: {path}")
    # These upstreams use absolute imports/dynamic prompt imports internally.
    namespace = {'HippoRAG2': 'hipporag', 'HyperGraphRAG': 'hypergraphrag'}.get(method, '_repar_upstream_' + method.lower())
    if namespace in sys.modules and list(getattr(sys.modules[namespace], '__path__', [])) != [str(source / package_path)]:
        raise RuntimeError(f'Conflicting upstream module already imported: {namespace}')
    if namespace not in sys.modules:
        module = types.ModuleType(namespace)
        module.__path__ = [str(source / package_path)]
        module.__package__ = namespace
        sys.modules[namespace] = module
    return namespace, spec


def module(namespace, relative):
    return importlib.import_module(namespace + "." + relative)
