from __future__ import annotations

import json
import math
import time
from pathlib import Path

import requests

from .io import digest, read_json, write_json


def sampling_parameters(config, structured=False):
    """Explicit sampler settings; protocol roles are independent of task type."""
    fields = {'temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'frequency_penalty', 'repeat_penalty'}
    result = {}
    for setting in ('sampling', 'control_sampling' if structured else 'answer_sampling'):
        supplied = config.get(setting, {})
        if not isinstance(supplied, dict) or set(supplied)-fields:
            raise ValueError(f'Unsupported explicit sampler setting: {setting}')
        for key, value in supplied.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('Sampler values must be numeric: '+key)
            if not math.isfinite(value):
                raise ValueError('Sampler values must be finite: '+key)
            if key in {'temperature', 'top_k'} and value < 0:
                raise ValueError('Sampler value must be nonnegative: '+key)
            if key == 'top_k' and not isinstance(value, int):
                raise ValueError('top_k must be an integer')
            if key in {'top_p', 'min_p'} and not 0 <= value <= 1:
                raise ValueError('Probability sampler outside[0,1]: '+key)
            if key == 'repeat_penalty' and value <= 0:
                raise ValueError('repeat_penalty must be positive')
            if key in {'presence_penalty', 'frequency_penalty'} and not -2 <= value <= 2:
                raise ValueError('Penalty sampler outside[-2,2]: '+key)
            result[key] = value
    return result


class LLMClient:
    def __init__(self, config, cache_dir="artifacts/cache/llm"):
        self.config = config
        self.base = config["base_url"].rstrip("/")
        self.session = requests.Session()
        self.cache_dir = Path(cache_dir)
        self.calls = []

    def post(self, route, body, timeout=3600):
        last = None
        for attempt in range(3):
            try:
                r = self.session.post(self.base + route, json=body, timeout=(20, timeout))
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as exc:
                # Invalid prompts must not silently become missing benchmark rows.
                if exc.response.status_code < 500:
                    raise RuntimeError(exc.response.text[:2000]) from exc
                last = exc
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = exc
            time.sleep(2 ** attempt)
        raise RuntimeError(f"API failed after three attempts: {last}")

    def tokenize(self, text):
        return self.post("/tokenize", {"content": text, "add_special": False})["tokens"]

    def detokenize(self, tokens):
        return self.post("/detokenize", {"tokens": tokens})["content"]

    def complete(self, messages, max_tokens=128, json_mode=False, tag="", schema=None):
        body = {
            "model": self.config["model"], "messages": messages,
            "temperature": self.config["temperature"], "seed": self.config["seed"],
            "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.config["enable_thinking"]},
        }
        body.update(sampling_parameters(self.config, structured=json_mode))
        if json_mode:
            if schema is not None or self.config.get("controller_schema"):
                body["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "repar_control", "strict": True,
                    "schema": schema if schema is not None else read_json(self.config["controller_schema"])}}
            else:
                body["response_format"] = {"type": "json_object"}
        elif self.config.get("answer_grammar"):
            body["grammar"] = self.config["answer_grammar"]
        record = self._complete_body(body, tag)
        return record["response"]["choices"][0]["message"].get("content", "") or ""

    def complete_tool(self, messages, tools, max_tokens=128, tag=''):
        """Native model tool protocol with the same deployment-aware cache."""
        body = {"model": self.config['model'], 'messages': messages,
            'temperature': self.config['temperature'], 'seed': self.config['seed'],
            'max_tokens': max_tokens, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': self.config['enable_thinking']},
            'tools': tools, 'tool_choice': 'required', 'parallel_tool_calls': False}
        body.update(sampling_parameters(self.config, structured=True))
        return self._complete_body(body, tag)['response']['choices'][0]['message']

    def _complete_body(self, body, tag):
        identity = {"request": body, "deployment_id": self.config.get("deployment_id", self.base)}
        key = digest(json.dumps(identity, sort_keys=True, ensure_ascii=False))
        path = self.cache_dir / (key + ".json")
        if path.exists():
            record = read_json(path)
            record = {**record, "cached": True, "tag": tag}
        else:
            started = time.perf_counter()
            response = self.post("/v1/chat/completions", body)
            record = {"key": key, "request": body, "response": response,
                      "seconds": time.perf_counter() - started, "cached": False, "tag": tag}
            write_json(path, record)
        self.calls.append({k: v for k, v in record.items() if k != "request"})
        return record
