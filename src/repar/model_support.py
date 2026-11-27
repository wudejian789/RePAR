"""Optional local model adapters and bounded, auditable inference.

No dataset formats or research strategies belong in this module.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import threading
import time

import numpy as np
import requests

from .client import LLMClient


class ExecutionBudgetExceeded(RuntimeError):
    pass


class BoundedLLMClient(LLMClient):
    """One HTTP attempt per call; keep the existing deployment-aware cache."""
    def __init__(self, config, cache_dir, *, deadline=None, call_seconds=60, journal=None, answer_grammar=False):
        config = dict(config)
        if not answer_grammar:
            config.pop("answer_grammar", None)
        config.pop("controller_schema", None)
        super().__init__(config, cache_dir)
        self.deadline = deadline
        self.call_seconds = call_seconds
        self.journal = Path(journal) if journal else None
        self._lock = threading.RLock()

    def _event(self, event):
        if self.journal:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def post(self, route, body, timeout=None):
        remaining = self.deadline - time.monotonic() if self.deadline else self.call_seconds
        if remaining <= 0:
            raise ExecutionBudgetExceeded("Execution deadline exhausted before HTTP call")
        started = time.perf_counter()
        try:
            response = self.session.post(self.base + route, json=body,
                                         timeout=(min(10, remaining), min(self.call_seconds, remaining)))
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            self._event({"event": "http_error", "route": route, "request": body,
                         "seconds": time.perf_counter() - started, "error": str(exc)})
            raise

    def complete(self, *args, **kwargs):
        # Local llama-server has one slot. Serialize upstream worker callbacks.
        with self._lock:
            if self.deadline and time.monotonic() >= self.deadline:
                raise ExecutionBudgetExceeded("Execution deadline exhausted before generation")
            result = super().complete(*args, **kwargs)
            self._event({"event": "completion", **self.calls[-1]})
            return result

    def complete_tool(self, *args, **kwargs):
        with self._lock:
            if self.deadline and time.monotonic() >= self.deadline:
                raise ExecutionBudgetExceeded('Execution deadline exhausted before native tool generation')
            result = super().complete_tool(*args, **kwargs)
            self._event({'event': 'completion', **self.calls[-1]})
            return result


def require_unconstrained(client):
    if getattr(client, "config", {}).get("answer_grammar"):
        raise ValueError("Indexing/reasoning requires a client without the answer-only grammar")


class SentenceEncoder:
    """Frozen sentence-transformer on CPU; window and mean-pool long inputs.

    Model files must already be local. No implicit download or GPU allocation.
    Each window is embedded with its model's own tokenization and mean pooling.
    The normalized mean of window vectors represents a long string.
    """
    def __init__(self, model_path, *, batch_size=32, threads=4):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(threads)
        self.model_path = str(Path(model_path).resolve())
        manifest = Path(model_path) / "download_manifest.json"
        if not manifest.is_file():
            raise ValueError("Encoder requires download_manifest.json with pinned file hashes")
        spec = json.loads(manifest.read_text(encoding="utf-8"))
        for item in spec["files"]:
            path = Path(model_path) / item["path"]
            path.resolve().relative_to(Path(model_path).resolve())
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Encoder checksum mismatch: {path}")
        self.identity = spec["model"] + "@" + spec["revision"]
        self.model = SentenceTransformer(self.model_path, device="cpu", local_files_only=True)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self._lock = threading.RLock()
        self.calls = []

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        started = time.perf_counter()
        windows, counts = [], []
        tokenizer = self.model.tokenizer
        width = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(False)
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            parts = [tokenizer.decode(ids[i:i + width], skip_special_tokens=True)
                     for i in range(0, len(ids), width)] or [""]
            counts.append(len(parts))
            windows.extend(parts)
        with self._lock:
            vectors = self.model.encode(windows, batch_size=self.batch_size, normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=False)
        output, cursor = [], 0
        for count in counts:
            vector = vectors[cursor:cursor + count].mean(axis=0)
            output.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
            cursor += count
        self.calls.append({"texts": len(texts), "windows": len(windows),
                           "seconds": time.perf_counter() - started})
        return np.asarray(output, dtype=np.float32)
