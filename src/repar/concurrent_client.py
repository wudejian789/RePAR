"""Thread-local generation clients for audited execution against one deployment pool."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import threading
import time

import requests

from .client import LLMClient


class ConcurrentLLMClient:
    """Preserve request construction/cache identity; allow independent calls in parallel.

    A loopback gateway owns the global endpoint-slot limits across worker processes.
    Complete request/response bodies remain in the normal cache. Journals retain
    compact completion metadata, including actual backend and queue/service time.
    """
    def __init__(self, config, cache_dir, *, journal=None, call_seconds=600,
                 answer_grammar=False, case_id=''):
        self.config = dict(config)
        if not answer_grammar:
            self.config.pop('answer_grammar', None)
        self.config.pop('controller_schema', None)
        self.base = self.config['base_url'].rstrip('/')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.journal = Path(journal) if journal else None
        self.call_seconds = call_seconds
        self.case_id = case_id
        self.calls = []
        self.errors = 0
        self._local = threading.local()
        self._lock = threading.RLock()
        self._key_locks = {}

    def _event(self, event):
        if self.journal:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')

    def _delegate(self):
        if not hasattr(self._local, 'client'):
            owner = self

            class Delegate(LLMClient):
                def post(self, route, body, timeout=None):
                    response = self.session.post(self.base + route, json=body,
                        timeout=(10, owner.call_seconds*4 + 60),
                        headers={'X-Repar-Case': owner.case_id})
                    response.raise_for_status()
                    result = response.json()
                    if route == '/v1/chat/completions':
                        result['_repar_transport'] = {
                            'endpoint': response.headers.get('X-Repar-Endpoint'),
                            'request_id': response.headers.get('X-Repar-Request-ID'),
                            'queue_seconds': float(response.headers.get('X-Repar-Queue-Seconds', '0')),
                            'service_seconds': float(response.headers.get('X-Repar-Service-Seconds', '0')),
                            'attempts': int(response.headers.get('X-Repar-Attempts', '1'))}
                    return result

            self._local.client = Delegate(self.config, self.cache_dir)
        return self._local.client

    @property
    def last_call(self):
        return self._local.last_call

    def complete(self, messages, max_tokens=128, json_mode=False, tag='', schema=None):
        # Identical concurrent requests share one completed cache record, without
        # serializing unrelated requests or leaking another thread's last_call.
        identity = json.dumps([messages, max_tokens, json_mode, schema, self.config],
                              sort_keys=True, ensure_ascii=False)
        identity = hashlib.sha256(identity.encode()).hexdigest()
        with self._lock:
            lock = self._key_locks.setdefault(identity, threading.Lock())
        with lock:
            client = self._delegate()
            started = time.perf_counter()
            try:
                result = client.complete(messages, max_tokens, json_mode, tag, schema)
                record = client.calls[-1]
                self._local.last_call = record
                compact = {**record, 'response': {
                    'usage': record['response'].get('usage', {}),
                    'choices': [{'finish_reason': x.get('finish_reason')}
                                for x in record['response']['choices']],
                    '_repar_transport': record['response'].get('_repar_transport', {})}}
                if record['cached']:
                    compact['original_seconds'] = record['seconds']
                    compact['seconds'] = time.perf_counter()-started
                with self._lock:
                    self.calls.append(compact)
                    self._event({'event': 'completion', **compact})
                return result
            except Exception as exc:
                with self._lock:
                    self.errors += 1
                    self._event({'event': 'completion_error', 'tag': tag,
                        'messages': messages, 'max_tokens': max_tokens,
                        'seconds': time.perf_counter()-started,
                        'error_type': type(exc).__name__, 'error': str(exc)})
                raise
            finally:
                client.calls.clear()

    def tokenize(self, text):
        return self._delegate().tokenize(text)

    def detokenize(self, tokens):
        return self._delegate().detokenize(tokens)
