"""Optional frozen pairwise relevance scoring of literal source candidates.

Scores only rank source text. They are neither facts nor entailment judgments.
All input tokens participate through overlapping bounded windows; the selected
evidence remains exact offsets in the original, unchanged knowledge source.
"""
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time


def token_windows(ids, capacity):
    if capacity < 1:
        raise ValueError('Pair window capacity must be positive')
    if len(ids) <= capacity:
        return [ids]
    stride = max(1, capacity*3//4)
    starts = list(range(0, len(ids)-capacity+1, stride))
    if starts[-1] != len(ids)-capacity:
        starts.append(len(ids)-capacity)
    return [ids[start:start+capacity] for start in starts]


@lru_cache(maxsize=2)
def load_reranker(directory, expected_manifest, threads):
    path = Path(directory)
    raw = (path/'download_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest:
        raise ValueError('Source reranker manifest differs from configured identity')
    manifest = json.loads(raw)
    for record in manifest['files']:
        if hashlib.sha256((path/record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Frozen source reranker file changed: '+record['path'])
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True,
        trust_remote_code=False, use_safetensors=True).to('cpu').eval()
    if model.config.num_labels != 1:
        raise ValueError('A single relevance logit is required')
    return tokenizer, model, manifest


class SourceReranker:
    def __init__(self, config, *, event_path=None):
        started = time.perf_counter()
        self.tokenizer, self.model, self.manifest = load_reranker(config['path'], config['manifest_sha256'], config.get('threads', 2))
        self.load_seconds = time.perf_counter()-started
        self.max_length = min(512, self.tokenizer.model_max_length)
        self.batch_size = config.get('batch_size', 8)
        self.cache_limit = config.get('cache_items', 4096)
        self.batch_by_length = config.get('batch_by_length', False)
        if not isinstance(self.batch_by_length, bool):
            raise ValueError('Source reranker batch_by_length must be boolean')
        if type(self.batch_size) is not int or self.batch_size < 1 or type(self.cache_limit) is not int or self.cache_limit < 1:
            raise ValueError('Source reranker requires positive integer batch and cache sizes')
        self.cache = OrderedDict()
        self.journal = Path(event_path) if event_path else None
        self.calls = self.pairs = self.windows = self.long_queries = self.long_sources = self.hits = 0
        self.seconds = 0.0

    def score_pairs(self, pairs):
        import torch
        capacity = self.max_length-self.tokenizer.num_special_tokens_to_add(pair=True)
        features, groups, feature_ids = [], [], {}
        for query, text in pairs:
            q = self.tokenizer.encode(query, add_special_tokens=False, verbose=False)
            query_parts = token_windows(q, max(1, capacity//2))
            self.long_queries += len(query_parts) > 1
            query_groups = []
            for part in query_parts:
                source_parts = self.source_parts(text, capacity-len(part))
                self.long_sources += len(source_parts) > 1
                indices = []
                for source in source_parts:
                    key = (tuple(part), tuple(source))
                    if key not in feature_ids:
                        item = self.tokenizer.prepare_for_model(part, pair_ids=source, truncation=False,
                            return_attention_mask=True, return_token_type_ids='token_type_ids' in self.tokenizer.model_input_names)
                        if len(item['input_ids']) > self.max_length:
                            raise ValueError('Source reranker pair exceeds model capacity')
                        feature_ids[key] = len(features)
                        features.append(item)
                    indices.append(feature_ids[key])
                query_groups.append(indices)
            groups.append(query_groups)
        order = list(range(len(features)))
        if getattr(self, 'batch_by_length', False):
            order.sort(key=lambda i: len(features[i]['input_ids']))
        batches = [order[start:start+self.batch_size] for start in range(0, len(order), self.batch_size)]
        scores = [0.0]*len(features)
        with torch.inference_mode():
            for indices in batches:
                batch = self.tokenizer.pad([features[i] for i in indices], return_tensors='pt')
                logits = self.model(**batch).logits.flatten().tolist()
                for i, value in zip(indices, logits, strict=True):
                    scores[i] = float(value)
        self.windows += len(features)
        self.pairs += len(pairs)
        # Every query window contributes; each may match anywhere in the source.
        result = [sum(max(scores[i] for i in indices) for indices in group)/len(group) for group in groups]
        if getattr(self, 'journal', None):
            from repar.io import append_jsonl
            event = {'query_source_pairs': pairs, 'window_inputs': [dict(f) for f in features],
                     'window_logits': scores, 'query_window_feature_indices': groups, 'relevance_scores': result}
            if self.batch_by_length:
                event['batch_feature_indices'] = batches
            append_jsonl(self.journal, event)
        return result

    def source_parts(self, text, capacity):
        """Score literal sentence neighborhoods, without changing source leaves.

        Every sentence is an anchor with its immediately adjacent sentences.
        Overlong neighborhoods use all overlapping token windows. Identical
        token neighborhoods are scored once, including repeated noisy text.
        """
        from .memory import source_boundaries
        bounds = source_boundaries(text)
        pieces, seen = [], set()
        for i in range(max(1, len(bounds)-1)):
            start = bounds[max(0, i-1)]
            end = bounds[min(len(bounds)-1, i+2)]
            ids = self.tokenizer.encode(text[start:end], add_special_tokens=False, verbose=False)
            for part in token_windows(ids, capacity):
                key = tuple(part)
                if key not in seen:
                    pieces.append(part)
                    seen.add(key)
        return pieces or [[]]

    def scores(self, query, texts):
        if not texts:
            return []
        import math
        started = time.perf_counter()
        self.calls += 1
        keys = [(query, text) for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        self.hits += len(keys)-len(missing)
        if missing:
            values = self.score_pairs(missing)
            if not all(math.isfinite(value) for value in values):
                raise ValueError('Nonfinite source relevance score')
            self.cache.update(zip(missing, values))
        result = [self.cache[key] for key in keys]
        for key in keys:
            self.cache.move_to_end(key)
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        self.seconds += time.perf_counter()-started
        return result

    def stats(self):
        value = {'model': self.manifest['model'], 'revision': self.manifest['revision'], 'device': 'cpu',
            'load_seconds': self.load_seconds, 'scoring_seconds': self.seconds, 'scoring_calls': self.calls,
            'query_source_pairs': self.pairs, 'pair_windows': self.windows, 'long_query_inputs': self.long_queries,
            'long_source_inputs_per_query_window': self.long_sources, 'cache_hits': self.hits,
            'generative_model_calls': 0, 'scores_are_entailment_probabilities': False,
            'aggregation': 'mean_over_query_windows_of_max_over_literal_sentence_neighborhood_windows'}
        if self.batch_by_length:
            value['batch_order'] = 'ascending_input_length'
        if self.journal and self.journal.exists():
            value['raw_pair_model_journal'] = {'path': self.journal.as_posix(),
                'sha256': hashlib.sha256(self.journal.read_bytes()).hexdigest(),
                'includes': 'exact text pairs, unpadded window token inputs and raw model logits'}
        return value
