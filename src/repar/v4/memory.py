"""Token-bounded exact source views, diversity and a visible-span ledger."""
from __future__ import annotations

from collections import Counter
from bisect import bisect_left
from functools import lru_cache
import math
import re

from repar.text import terms


def merge_ranges(ranges):
    result = []
    for lo, hi in sorted(ranges):
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


@lru_cache(maxsize=4096)
def source_boundaries(text):
    """Conservative source offsets, without treating decimals or line wraps as sentences."""
    breaks = re.finditer(r'[。！？][”’"\')\]]*\s*|[.!?][”’"\')\]]*\s+(?=\S)|\n[ \t]*\n\s*', text)
    return tuple(sorted({0, len(text), *(m.end() for m in breaks)}))


@lru_cache(maxsize=4096)
def paragraph_bounds(text):
    return tuple(sorted({0, len(text), *(m.end() for m in re.finditer(r'\n[ \t]*\n\s*', text))}))


def complete_word_end(text, end):
    """Back off a hard budget cut inside a Latin word or numeric token."""
    if 0 < end < len(text):
        is_word = lambda c: c.isascii() and (c.isalnum() or c in "_'’-" )
        if is_word(text[end-1]) and is_word(text[end]):
            while end > 0 and is_word(text[end-1]):
                end -= 1
    return end


class EvidenceMemory:
    def __init__(self, context, client, budget=7000, similarity=None, *, source_view_policy='paragraph_restore',
                 entity_source_references=False, tokenize_terms=terms, retain_search_focus=False,
                 term_weight=None):
        self.context, self.budget = context, budget
        self.similarity = similarity
        if source_view_policy not in {'query_windows', 'paragraph_restore', 'scored_neighborhood'}:
            raise ValueError('Unknown original-source view policy')
        self.source_view_policy = source_view_policy
        self.entity_source_references = entity_source_references
        self.terms = tokenize_terms
        self.retain_search_focus = retain_search_focus
        self.term_weight = term_weight
        self.records, self.entities, self.visible, self.exposures = {}, {}, {}, []
        self.source_views, self.selections = {}, []
        self.source_view_queries = {}
        self.tree_nodes = {"T0"}
        self.browsed_nodes = {}
        self.count = lru_cache(maxsize=8192)(lambda text: len(client.tokenize(text)))

    def observe(self, observation):
        new = []
        bindings = observation.get('source_view_query_bindings', {})
        for r in observation.get("results", []):
            if self.context[r["start"]:r["end"]] != r["text"]:
                raise ValueError("Source does not match original offsets")
            if r["id"] not in self.records:
                new.append(r["id"])
            self.records[r["id"]] = r
            if r['id'] in bindings:
                self.source_view_queries[r['id']] = bindings[r['id']]['query']
            elif not self.retain_search_focus:
                self.source_view_queries.pop(r['id'], None)
            self.entities.update({e["id"]: e for e in r.get("entities", [])})
            self.tree_nodes.update(r.get("tree_path", []))
        self.entities.update({e["id"]: e for e in observation.get("entities", [])})
        for n in [observation.get("node"), *observation.get("children", [])]:
            if n and "id" in n:
                node_id = n["id"] if isinstance(n["id"], str) else f"T{n['id']}"
                self.tree_nodes.add(node_id)
                if observation.get('operator') == 'tree_browse':
                    self.browsed_nodes.setdefault(node_id, {}).update({
                        key: n[key] for key in ('lo', 'hi', 'preview') if key in n})
        return new

    @property
    def visible_characters(self):
        return sum(hi-lo for ranges in self.visible.values() for lo, hi in ranges)

    def mark_visible(self, views, stage):
        before = self.visible_characters
        for r in views:
            self.visible[r["id"]] = merge_ranges([*self.visible.get(r["id"], []), (r["start"], r["end"])])
            parent = self.records[r['id']]
            if self.context[r['start']:r['end']] != r['text']:
                raise ValueError('Visible source view does not match original text')
            key = (r['id'], r['start'], r['end'])
            if key not in self.source_views:
                boundaries = {parent['start']+b for b in source_boundaries(parent['text'])}
                self.source_views[key] = {**parent, **r,
                    'source_start': parent['start'], 'source_end': parent['end'],
                    'truncated_before': r['start'] > parent['start'],
                    'truncated_after': r['end'] < parent['end'],
                    'partial_sentence': r['start'] not in boundaries or r['end'] not in boundaries}
        self.exposures.append({"stage": stage, "spans": [{k: r[k] for k in ("id", "start", "end")} for r in views]})
        return self.visible_characters-before

    def unread(self, cid):
        r, cursor = self.records[cid], self.records[cid]["start"]
        for lo, hi in self.visible.get(cid, []):
            if lo > cursor:
                return cursor
            cursor = max(cursor, hi)
        return cursor if cursor < r["end"] else None

    def focused_unread(self, cid, query):
        """Locate the best matching unread sentence in an observed source."""
        record = self.records[cid]
        words = set(self.terms(query))
        base = record['start']
        text = record['text']
        visible = self.visible.get(cid, [])
        bounds = source_boundaries(text)
        best = None
        for start, end in zip(bounds, bounds[1:]):
            if any(base + start < hi and lo < base + end for lo, hi in visible):
                continue
            sentence = set(self.terms(text[start:end]))
            overlap = self.match_weight(words & sentence)
            if overlap:
                value = (overlap / math.sqrt(len(sentence)), -start)
                if best is None or value > best:
                    best = value
        if best is None:
            return None
        start = base - best[1]
        end = min((lo for lo, _ in visible if lo > start), default=record['end'])
        return start, end

    def match_weight(self, matching):
        if self.term_weight is None:
            return len(matching)
        return math.fsum(self.term_weight(term) for term in matching)

    def page(self, record, start, end, cap, *, align=False):
        start, end = max(record["start"], start), min(record["end"], end)
        if self.count(self.context[start:end]) > cap:
            low, high = start, end
            while low < high:
                mid = (low+high+1)//2
                if self.count(self.context[start:mid]) <= cap:
                    low = mid
                else:
                    high = mid-1
            end = low
            if align:
                relative = source_boundaries(record["text"])
                choices = [record["start"]+b for b in relative if start < record["start"]+b <= end]
                if choices:
                    end = choices[-1]
                else:
                    end = start+complete_word_end(self.context[start:record["end"]], end-start)
        boundaries = {record["start"]+b for b in source_boundaries(record["text"])}
        return {**record, "start": start, "end": end, "text": self.context[start:end],
                "source_start": record["start"], "source_end": record["end"],
                "truncated_before": start > record["start"], "truncated_after": end < record["end"],
                "partial_sentence": start not in boundaries or end not in boundaries}

    def _best(self, record, query, cap, excluded=()):
        if self.source_view_policy == 'scored_neighborhood':
            return self._neighborhood_best(record, query, cap, excluded)
        if not excluded and self.count(record["text"]) <= cap:
            return self.page(record, record["start"], record["end"], cap)
        q = set(self.terms(query))
        boundaries = source_boundaries(record["text"])
        candidates = []
        for a, b in zip(boundaries, boundaries[1:]):
            absolute, sentence_end = record["start"]+a, record["start"]+b
            if b <= a or any(absolute < hi and lo < sentence_end for lo, hi in excluded):
                continue
            text = record["text"][a:b]
            words = set(self.terms(text))
            candidates.append((self.match_weight(q & words)/math.sqrt(max(1, len(words))), -a, absolute, sentence_end))
        if not candidates:
            return None
        if self.similarity:
            scores = self.similarity.scores(query, [self.context[r[2]:r[3]] for r in candidates])
            _, _, start, anchor_end = candidates[max(range(len(candidates)), key=lambda i: (scores[i], candidates[i][0], candidates[i][1]))]
        else:
            _, _, start, anchor_end = max(candidates)
        end = min([record["end"], *[lo for lo, _ in excluded if lo >= anchor_end]])
        if self.count(self.context[start:anchor_end]) > cap:
            return self.page(record, start, end, cap, align=True)
        # Include the preceding complete sentence only when the selected sentence
        # still fits. No fixed-character backoff can cut a word or drop the anchor.
        absolute_boundaries = [record["start"]+b for b in boundaries]
        position = bisect_left(absolute_boundaries, start)
        if position > 0:
            previous = absolute_boundaries[position-1]
            overlaps = any(previous < hi and lo < anchor_end for lo, hi in excluded)
            if not overlaps and self.count(self.context[previous:anchor_end]) <= cap:
                start = previous
        view = self.page(record, start, end, cap, align=True)
        return view

    def _neighborhood_best(self, record, query, cap, excluded):
        """Display the literal neighborhood that actually competes on relevance.

        A generous token ceiling does not establish that the entire source leaf
        is useful. Keep adjacent complete sentences with each anchor, and expose
        the selected original range. More source context remains available via
        the existing cursor and adjacent-source read operators.
        """
        bounds = [record['start']+offset for offset in source_boundaries(record['text'])]
        candidates = {}
        for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
            if start == end or any(start < hi and lo < end for lo, hi in excluded):
                continue
            lo, hi = bounds[max(0, i-1)], bounds[min(len(bounds)-1, i+2)]
            lo = max([lo, *[b for a, b in excluded if b <= start]])
            hi = min([hi, *[a for a, b in excluded if a >= end]])
            if self.count(self.context[lo:hi]) > cap:
                lo, hi = start, end
            view = self.page(record, lo, hi, cap, align=True)
            if view['end'] > view['start']:
                candidates[(view['start'], view['end'])] = view
        if not candidates:
            return None
        views = list(candidates.values())
        q = set(self.terms(query))
        def lexical(view):
            words = set(self.terms(view['text']))
            return self.match_weight(q & words)/math.sqrt(max(1, len(words)))
        scores = self.similarity.scores(query, [view['text'] for view in views]) if self.similarity else [lexical(view) for view in views]
        return max(zip(views, scores), key=lambda pair: (pair[1], lexical(pair[0]), -pair[0]['start']))[0]

    @staticmethod
    def block(r):
        partial = " partial_sentence=true" if r.get("partial_sentence") else ""
        return f"[{r['id']} chars={r['start']}:{r['end']}{partial}]\n{r['text']}"

    def discover(self, query, budget=None, *, records=None, stage=None, relevance_order=False):
        budget = self.budget if budget is None else budget
        source = list(self.records.values()) if records is None else list(records)
        if not source:
            return "", []
        q, document_frequency = set(self.terms(query)), Counter()
        words = {r["id"]: set(self.terms(r["text"])) for r in source}
        for values in words.values():
            document_frequency.update(values)
        semantic = dict(zip((r['id'] for r in source), self.similarity.scores(query, [r['text'] for r in source]))) if self.similarity else None
        def relevance(r):
            matching = q & words[r["id"]]
            power = (self.match_weight(matching) if self.term_weight is not None else
                     sum(math.log(1+len(source)/(1+document_frequency[t])) for t in matching))
            return power/math.sqrt(max(1, len(words[r["id"]])))
        pending, chosen, groups, used_terms = source[:], [], Counter(), set()
        # A first pass spreads the finite budget over sources; a second pass
        # can use remaining capacity for additional exact ranges of a source.
        share = max(80, budget//max(1, len(source))-40)
        for extra in ((False,) if self.source_view_policy == 'scored_neighborhood' else (False, True)):
            if extra:
                pending = source[:]
            while pending:
                used = self.count("\n\n".join(self.block(r) for r in chosen)) if chosen else 0
                remaining = budget-used-40
                if remaining < 40:
                    break
                def rank(r):
                    path = r.get("tree_path", [])
                    group = path[1] if len(path) > 1 else r["id"]
                    novelty = len(words[r["id"]]-used_terms)/max(1, len(words[r["id"]]))
                    lexical = relevance(r)+0.08*novelty+0.06/(1+groups[group])
                    return (semantic[r['id']], lexical, -r['start']) if semantic is not None else (lexical, -r['start'])
                record = max(pending, key=rank)
                pending.remove(record)
                occupied = [(v["start"], v["end"]) for v in chosen if v["id"] == record["id"]]
                cap = min(remaining, max(share, remaining//max(1, len(pending)+1)))
                view = self._best(record, self.source_view_queries.get(record['id'], query), cap, occupied)
                if not view or view["end"] <= view["start"]:
                    continue
                trial = sorted([*chosen, view], key=lambda r: (r["start"], r["end"]))
                if self.count("\n\n".join(self.block(r) for r in trial)) > budget:
                    continue
                chosen.append(view)
                used_terms.update(words[record["id"]])
                path = record.get("tree_path", [])
                groups[path[1] if len(path) > 1 else record["id"]] += 1
        if relevance_order:
            # Preserve the packing rank across distinct passages and the
            # original order of ranges within a passage.
            passage_rank = {cid: rank for rank, cid in enumerate(dict.fromkeys(r["id"] for r in chosen))}
            chosen.sort(key=lambda r: (passage_rank[r["id"]], r["start"]))
            while chosen and self.count("\n\n".join(self.block(r) for r in chosen)) > budget:
                chosen.pop()
        else:
            chosen.sort(key=lambda r: (r["start"], r["end"]))
        if stage:
            self.mark_visible(chosen, stage)
        return "\n\n".join(self.block(r) for r in chosen), chosen

    def render(self, query, budget=None, *, records=None, stage=None, already_shown=()):
        """Select whole observed raw ranges, without re-extracting by a new query.

        Explicit records request fresh views for an actual observation. The
        controller, retained feedback, and final response share stable ranges.
        Supported-source claims do not grant a range extra weight.
        """
        budget = self.budget if budget is None else budget
        if self.source_view_policy == 'query_windows':
            return self.discover(query, budget, records=records, stage=stage)
        if records is not None:
            allowed = {r['id'] for r in records}
        else:
            allowed = set(self.records)
        candidates = [r for r in self.source_views.values() if r['id'] in allowed]
        known = {r['id'] for r in self.source_views.values()}
        unseen = sorted((self.records[cid] for cid in allowed-known), key=lambda r: r['start'])
        if unseen:
            _, fresh = self.discover(query, budget, records=unseen)
            candidates.extend(fresh)
        if already_shown:
            candidates = [r for r in candidates if not any(
                r['id'] == shown['id'] and shown['start'] <= r['start'] and r['end'] <= shown['end']
                for shown in already_shown)]
        # Previously unseen sources may enter a request for the first time;
        # an old ID never receives a different window just because query changed.
        if not candidates:
            return '', []
        q, frequency = set(self.terms(query)), Counter()
        words = {(r['id'], r['start'], r['end']): set(self.terms(r['text'])) for r in candidates}
        for w in words.values():
            frequency.update(w)
        semantic = dict(zip(((r['id'], r['start'], r['end']) for r in candidates),
            self.similarity.scores(query, [r['text'] for r in candidates]))) if self.similarity else None
        def relevance(r):
            w = words[(r['id'], r['start'], r['end'])]
            return sum(math.log(1+len(candidates)/(1+frequency[t])) for t in q & w)/math.sqrt(max(1, len(w)))
        chosen, groups, used, selected_keys = [], Counter(), set(), set()
        # First spread capacity across observed source IDs, then add further
        # ranges. Union overlapping selected ranges only within the same source.
        for extra in (False, True):
            pending = [r for r in candidates if (r['id'], r['start'], r['end']) not in selected_keys]
            while pending:
                def rank(r):
                    w = words[(r['id'], r['start'], r['end'])]
                    path = r.get('tree_path', [])
                    group = path[1] if len(path) > 1 else r['id']
                    lexical = relevance(r)+0.08*len(w-used)/max(1, len(w))+0.06/(1+groups[group])
                    return (semantic[(r['id'], r['start'], r['end'])], lexical, -r['start']) if semantic is not None else (lexical, -r['start'])
                r = max(pending, key=rank)
                pending.remove(r)
                if not extra and any(v['id'] == r['id'] for v in chosen):
                    continue
                trial = self.union_views([*chosen, r])
                if self.count('\n\n'.join(self.block(v) for v in trial)) > budget:
                    continue
                chosen = trial
                selected_keys.add((r['id'], r['start'], r['end']))
                used.update(words[(r['id'], r['start'], r['end'])])
                path = r.get('tree_path', [])
                groups[path[1] if len(path) > 1 else r['id']] += 1
        chosen.sort(key=lambda r: (r['start'], r['end']))
        anchors = [{k: r[k] for k in ('id', 'start', 'end')} for r in chosen]
        chosen = self.restore_context(chosen, budget)
        retained = lambda r: any(r['id'] == v['id'] and v['start'] <= r['start'] and r['end'] <= v['end'] for v in chosen)
        self.selections.append({'stage': stage or 'pack', 'budget': budget,
            'tokens': self.count('\n\n'.join(self.block(r) for r in chosen)),
            'anchor_ranges': anchors,
            'selected_ranges': [{k: r[k] for k in ('id', 'start', 'end')} for r in chosen],
            'retained': [{k: r[k] for k in ('id', 'start', 'end')} for r in candidates if retained(r)],
            'dropped_for_budget': [{k: r[k] for k in ('id', 'start', 'end')} for r in candidates if not retained(r)]})
        if stage:
            self.mark_visible(chosen, stage)
        return '\n\n'.join(self.block(r) for r in chosen), chosen

    def restore_context(self, views, budget):
        """Grow around anchors within their original paragraph envelopes.

        Original anchors are never shortened or moved. Only immediately
        adjacent sentence ranges compete for remaining capacity; unrelated
        paragraphs are not added just to consume the budget.
        """
        chosen, depths = self.union_views(views), Counter()
        if self.source_view_policy == 'scored_neighborhood':
            return chosen
        while True:
            proposals = []
            for r in chosen:
                parent = self.records[r['id']]
                paragraphs = [parent['start']+b for b in paragraph_bounds(parent['text'])]
                lower = max(b for b in paragraphs if b <= r['start'])
                upper = min(b for b in paragraphs if b >= r['end'])
                bounds = [parent['start']+b for b in source_boundaries(parent['text'])]
                before = [b for b in bounds if lower <= b < r['start']]
                after = [b for b in bounds if r['end'] < b <= upper]
                options = [(before[-1], r['end'], 'before')] if before else []
                if after:
                    options.append((r['start'], after[0], 'after'))
                for start, end, direction in options:
                    candidate = self.page(parent, start, end, self.count(self.context[start:end]))
                    trial = self.union_views([v for v in chosen if v is not r]+[candidate])
                    cost = self.count('\n\n'.join(self.block(v) for v in trial))
                    if cost <= budget:
                        proposals.append(((depths[r['id']], end-start-(r['end']-r['start']), r['start'], direction), trial, r['id']))
            if not proposals:
                break
            _, chosen, cid = min(proposals, key=lambda p: p[0])
            depths[cid] += 1
        return chosen

    def union_views(self, views):
        result = []
        for r in sorted(views, key=lambda r: (r['id'], r['start'], r['end'])):
            if result and result[-1]['id'] == r['id'] and r['start'] <= result[-1]['end']:
                prior = result.pop()
                parent = self.records[r['id']]
                start, end = prior['start'], max(prior['end'], r['end'])
                # The interval has no unseen gap, so the union is exact text
                # previously supplied by its constituent source views.
                result.append(self.page(parent, start, end, self.count(self.context[start:end])))
            else:
                result.append(dict(r))
        return sorted(result, key=lambda r: (r['start'], r['end']))

    def observation_view(self, observation, query, budget, stage):
        records = observation.get("results", [])
        requested = observation.get("requested_view")
        if requested:
            record = self.records[requested["id"]]
            views = [self.page(record, requested["start"], requested["end"], max(1, budget-100), align=True)]
            self.mark_visible(views, stage)
        else:
            _, views = self.discover(query, budget, records=records, stage=stage)
        value = {k: observation[k] for k in ("operator", "status", "error", "query_channels", "fusion", "relation_status") if k in observation}
        if 'requested_result_limit' in observation.get('fusion', {}):
            returned = {r['id'] for r in records}
            value['query_channels'] = [{**channel, 'ids': [cid for cid in channel['ids'] if cid in returned]}
                                       for channel in observation['query_channels']]
        value["source_views"] = [{k: r[k] for k in ("id", "start", "end", "text", "source_start", "source_end", "truncated_before", "truncated_after", "partial_sentence")} for r in views]
        entity_fields = ('id', 'name', 'source_count', 'passage_ids') if self.entity_source_references else ('id', 'name', 'source_count')
        value["entities"] = [{k: e.get(k) for k in entity_fields} for e in observation.get("entities", [])[:12]]
        value["relations"] = observation.get("relations", [])[:12]
        value["propositions"] = observation.get("propositions", [])[:12]
        value["children"] = observation.get("children", [])[:12]
        value["read_cursors"] = [{"id": r["id"], "offset": self.unread(r["id"]), "source_end": r["end"]} for r in records if self.unread(r["id"]) is not None]
        return value
