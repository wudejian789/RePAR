"""Query channels and paged reads on existing immutable knowledge indexes."""
from __future__ import annotations

import json
import math
from copy import deepcopy

from repar.text import terms


def request_identity(name, parameters):
    """Compare complete serialized requests, without inferring textual equivalence."""
    return json.dumps([name, parameters], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class KnowledgeAPI:
    def __init__(self, index, semantic_sources=None, reranker=None, candidate_top_k=8, *,
                 fuse_before_limit=False, use_channel_view_queries=False, limit_each_search_channel=False,
                 r3_retrieval=False, eligible_graph_edges_first=False, unavailable_sources=(),
                 bind_single_search_views=False):
        self.index = index
        self.semantic_sources = semantic_sources
        self.reranker = reranker
        if type(candidate_top_k) is not int or not 1 <= candidate_top_k <= 24:
            raise ValueError('Retrieval candidate budget must be an integer within1..24')
        self.candidate_top_k = candidate_top_k
        self.fuse_before_limit = fuse_before_limit
        self.use_channel_view_queries = use_channel_view_queries
        self.bind_single_search_views = bind_single_search_views
        self.limit_each_search_channel = limit_each_search_channel
        self.r3_retrieval = r3_retrieval
        self.eligible_graph_edges_first = eligible_graph_edges_first
        self.unavailable_sources = frozenset(unavailable_sources)
        self.search_cache, self.search_events = {}, []
        self.entity_scores = {}

    def _source_entities(self, observation, query):
        """Clip descriptions after ranking all already returned source IDs."""
        graph = getattr(self.index, 'entity_graph', None)
        if not graph or not query:
            return
        q = set(terms(query))
        for record in observation.get('results', []):
            if 'entity_ids' not in record:
                continue
            entities = [graph.describe(eid) for eid in record.get('entity_ids', [])]
            def score(entity):
                key = (query, entity['id'])
                if key not in self.entity_scores:
                    names = set(terms(entity['canonical']+' '+' '.join(entity.get('aliases', []))))
                    self.entity_scores[key] = sum(math.log(1+len(graph.entities)/(1+len(graph.postings.get(t, []))))
                        for t in names & q)/math.sqrt(max(1, len(names)))
                return self.entity_scores[key]
            ranked = sorted(entities, key=lambda e: (-score(e), int(e['id'][1:])))[:24]
            record['entities'] = ranked
            record['entity_descriptor_selection'] = {'query': query, 'candidate_ids': list(record.get('entity_ids', [])),
                'selected_ids': [e['id'] for e in ranked], 'limit': 24}

    def _hybrid_candidates(self, parameters, routed, excluded):
        """Fuse routed and complete scoped lexical lists at the candidate layer.

        The historical router protects only two lexical results. Keeping both
        full lists here lets high-ranked lexical matches compete even when a
        centroid excluded their branch. No index or source text is rebuilt.
        """
        if parameters.get('mode', 'hybrid') != 'hybrid' or not callable(getattr(self.index, 'bm25', None)):
            return routed
        limit = max(1, min(24, int(parameters.get('top_k', 8))))
        pool_limit = max(limit, self.candidate_top_k) if self.reranker else limit
        lexical_hits = self.index.bm25(parameters['query'], pool_limit+len(excluded), parameters.get('scope'))
        lexical = [self.index.record(i, score) for i, score in lexical_hits
                   if score > 0 and f'C{i}' not in excluded][:pool_limit]
        first = [r for r in routed['results'] if r.get('score', 0) > 0]
        ranks, records = {}, {}
        for channel in (first, lexical):
            for rank, r in enumerate(channel):
                records[r['id']] = r
                ranks[r['id']] = ranks.get(r['id'], 0)+1/(21+rank)
        selected = sorted(records, key=lambda cid: (-ranks[cid], records[cid]['start']))[:pool_limit]
        result = {**routed, 'results': [{**records[cid], 'score': ranks[cid]} for cid in selected],
            'candidate_generation': {'method': 'routed_and_lexical_reciprocal_rank', 'rank_constant': 20,
                'routed_ids': [r['id'] for r in first], 'lexical_ids': [r['id'] for r in lexical],
                'candidate_count': len(records), 'selected_ids': selected}}
        if self.semantic_sources is not None:
            semantic_options = {'exclude_passages': excluded} if excluded else {}
            semantic = self.semantic_sources.search(
                parameters['query'], pool_limit, parameters.get('scope'), **semantic_options)
            scores, candidates = {}, {}
            for channel in (result['results'], semantic):
                for rank, r in enumerate(channel):
                    candidates[r['id']] = r
                    scores[r['id']] = scores.get(r['id'], 0)+1/(21+rank)
            final = sorted(candidates, key=lambda cid: (-scores[cid], candidates[cid]['start']))[:pool_limit]
            result['candidate_generation']['semantic_channel'] = {
                'pre_semantic_ids': selected, 'semantic_ids': [r['id'] for r in semantic],
                'cosine_scores': [r['score'] for r in semantic], 'candidate_count': len(candidates),
                'selected_ids': final, 'fusion': 'equal_reciprocal_rank', 'rank_constant': 20}
            result['results'] = [{**candidates[cid], 'score': scores[cid]} for cid in final]
            if self.reranker:
                for rank, r in enumerate(semantic):
                    records[r['id']] = r
                    ranks[r['id']] = ranks.get(r['id'], 0)+1/(21+rank)
        if self.reranker:
            # Rank the union before a result cap can discard a unique source.
            # Candidate recall and evidence volume are separate resource limits.
            pool = sorted(records.values(), key=lambda r: r['start'])
            values = self.reranker.scores(parameters['query'], [r['text'] for r in pool])
            order = sorted(range(len(pool)), key=lambda i: (-values[i], -ranks[pool[i]['id']], pool[i]['start']))
            result['results'] = [{**pool[i], 'score': ranks[pool[i]['id']], 'relevance_logit': values[i]} for i in order[:limit]]
            result['candidate_generation']['pair_reranking'] = {'candidate_ids': [r['id'] for r in pool],
                'relevance_logits': values, 'reciprocal_rank_scores': [ranks[r['id']] for r in pool],
                'selected_ids': [r['id'] for r in result['results']], 'candidate_channel_limit': pool_limit,
                'requested_result_limit': limit, 'scores_are_entailment_probabilities': False}
        return result

    def _search(self, parameters):
        """Reuse an identical primitive query within one immutable index session."""
        key = request_identity("tree_search", parameters)
        cached = key in self.search_cache
        if not cached:
            primitive = dict(parameters)
            exclusions = primitive.pop('exclude_passages', [])
            if not isinstance(exclusions, list) or any(not isinstance(cid, str) for cid in exclusions):
                raise ValueError('Search exclusions must be a list of source passage IDs')
            excluded = {f'C{self.index._leaf_id(cid)}' for cid in exclusions}
            if excluded and primitive.get('mode', 'hybrid') != 'hybrid':
                raise ValueError('Source exclusion is available in hybrid search')
            excluded.update(self.unavailable_sources)
            if self.reranker and parameters.get('mode', 'hybrid') == 'hybrid':
                primitive['top_k'] = max(parameters.get('top_k', 8), self.candidate_top_k)
            routed = self.index.execute("tree_search", primitive)
            if excluded:
                routed['results'] = [record for record in routed['results'] if record['id'] not in excluded]
            has_optional_channel = self.semantic_sources is not None or bool(excluded)
            result = routed if self.r3_retrieval and not has_optional_channel else self._hybrid_candidates(parameters, routed, excluded)
            self.search_cache[key] = deepcopy(result)
        result = deepcopy(self.search_cache[key])
        self.search_events.append({"identity": key, "parameters": dict(parameters), "cached": cached,
                                   "source_ids": [r["id"] for r in result.get("results", [])],
                                   "candidate_generation": deepcopy(result.get('candidate_generation'))})
        return result

    def execute(self, name, parameters):
        p = dict(parameters)
        if name == "read" and "offset" in p:
            cid = p.pop("id")
            offset, length = p.pop("offset"), p.pop("length")
            raw = self.index.execute("read", {"ids": [cid], "radius": 0})
            record = next(r for r in raw["results"] if r["id"] == cid)
            if not record["start"] <= offset < record["end"] or not 1 <= length <= 16000:
                raise ValueError("Read cursor outside observed source bounds")
            raw["requested_view"] = {"id": cid, "start": offset, "end": min(offset+length, record["end"])}
            observation = raw
        elif name == "tree_search":
            first_event = len(self.search_events)
            anchor = p.pop("anchor_query", None)
            use_anchor = bool(anchor and anchor != p["query"])
            channel_p = dict(p)
            if self.fuse_before_limit and use_anchor and not self.limit_each_search_channel:
                channel_p['top_k'] = max(p['top_k'], self.candidate_top_k)
            focused = self._search(channel_p)
            channels = [{"channel": "focus", "parameters": channel_p,
                         "ids": [r["id"] for r in focused["results"] if r.get("score", 0) > 0]}]
            observation = focused
            if use_anchor:
                anchor_p = {**channel_p, "query": anchor}
                anchored = self._search(anchor_p)
                channels.append({"channel": "task", "parameters": anchor_p,
                                 "ids": [r["id"] for r in anchored["results"] if r.get("score", 0) > 0]})
                scores, records = {}, {}
                # Equal rank scales let either query contribute a unique source,
                # even when both channels return their full result limit.
                for result in (focused, anchored):
                    for rank, r in enumerate(result["results"]):
                        if r.get("score", 0) <= 0:
                            continue
                        records[r["id"]] = r
                        scores[r["id"]] = scores.get(r["id"], 0)+1/(20+rank+1)
                ids = sorted(records, key=lambda cid: (-scores[cid], records[cid]["start"]))
                if not self.limit_each_search_channel:
                    ids = ids[:p['top_k']]
                observation = {**focused, "results": [{**records[cid], "score": scores[cid]} for cid in ids]}
                observation["fusion"] = {"method": "equal_reciprocal_rank", "rank_constant": 20,
                    "channel_weights": {"focus": 1, "task": 1}, "candidate_count": len(records)}
                if self.fuse_before_limit:
                    observation['fusion']['requested_result_limit'] = p['top_k']
                if self.limit_each_search_channel:
                    observation['fusion']['result_limit_scope'] = 'each_query'
                if self.use_channel_view_queries:
                    matches = {}
                    for channel in channels:
                        for rank, cid in enumerate(channel['ids']):
                            if cid not in matches or rank < matches[cid]['rank']:
                                matches[cid] = {'rank': rank, 'channel': channel['channel'],
                                                'query': channel['parameters']['query']}
                    observation['source_view_query_bindings'] = {cid: matches[cid] for cid in ids}
            observation["query_channels"] = channels
            observation["results"] = [r for r in observation["results"] if r.get("score", 0) > 0]
            if self.use_channel_view_queries and not use_anchor and self.bind_single_search_views:
                observation['source_view_query_bindings'] = {
                    cid: {'rank': rank, 'channel': 'focus', 'query': channel_p['query']}
                    for rank, cid in enumerate(channels[0]['ids'])
                }
            observation["query_execution"] = deepcopy(self.search_events[first_event:])
        elif name == "graph_neighbors" and self.eligible_graph_edges_first:
            observation = self.index.execute(name, {**p, 'eligible_edges_first': True})
        else:
            observation = self.index.execute(name, p)
        if name == "graph_neighbors":
            excluded = set(p.get("exclude_passages", []))
            observation["results"] = [r for r in observation.get("results", []) if r["id"] not in excluded]
            observation["propositions"] = [r for r in observation.get("propositions", []) if r["passage_id"] not in excluded]
        if self.unavailable_sources:
            observation["results"] = [r for r in observation.get("results", []) if r["id"] not in self.unavailable_sources]
            if "propositions" in observation:
                observation["propositions"] = [
                    r for r in observation["propositions"] if r["passage_id"] not in self.unavailable_sources]
        if not self.r3_retrieval:
            self._source_entities(observation, parameters.get('query', ''))
        observation["status"] = "ok" if any(observation.get(k) for k in ("results", "entities", "children")) else "no_match"
        observation["executed_parameters"] = dict(parameters)
        observation["relation_status"] = "Automatically extracted relations locate sources; not verified entailment."
        return observation
