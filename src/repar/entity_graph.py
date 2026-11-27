"""Source-grounded entity/relation graph, independent of questions and task prompts.

Entities are many-to-many with passages. Predicates are extracted from dependency
arcs; co-mention links are explicitly weaker and never relabelled as facts.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import groupby

from .index import terms


@lru_cache(maxsize=2)
def load_parser(model="en_core_web_sm"):
    import spacy
    nlp = spacy.load(model)
    nlp.max_length = 2_000_000
    return nlp


HAN = re.compile(r"[\u3400-\u9fff\U00020000-\U000323af]")


def graph_language(text):
    han = len(HAN.findall(text))
    latin = len(re.findall(r"[A-Za-z]+", text))
    return "zh_core_web_sm" if han and han >= latin else "en_core_web_sm"


def chinese_noun_phrases(doc):
    from spacy.tokens import Span

    phrases = {span.root.i: span for span in doc.ents if span.root.pos_ in {"NOUN", "PROPN"}}
    for noun in doc:
        if noun.pos_ not in {"NOUN", "PROPN"} or noun.i in phrases:
            continue
        if noun.dep_ in {"compound:nn", "name"} and noun.head.pos_ in {"NOUN", "PROPN"}:
            continue
        members = {noun.i}
        for child in noun.children:
            if child.dep_ in {"compound:nn", "name", "amod"}:
                members.update(token.i for token in child.subtree)
        start, end = noun.i, noun.i + 1
        while start - 1 in members:
            start -= 1
        while end in members:
            end += 1
        phrases[noun.i] = Span(doc, start, end)
    return sorted(phrases.values(), key=lambda span: (span.start, span.end))


class EntityGraph:
    def __init__(self, texts, spans, model="en_core_web_sm", progress=None):
        self.entities, self.propositions, self.relations = [], [], []
        self.passage_entities = [[] for _ in texts]
        self.passage_propositions = [[] for _ in texts]
        self.entity_lookup = {}
        self.adjacency = defaultdict(list)
        relation_lookup = {}
        if model == "auto":
            models = {}
            self.extractor = {"model": model, "models": models,
                              "kind": "named-entities+noun-phrases+dependency-predicates"}

            def documents():
                for name, group in groupby(texts, key=graph_language):
                    parser = load_parser(name)
                    models[name] = parser.meta["version"]
                    yield from parser.pipe(group, batch_size=48)
        else:
            parser = load_parser(model)
            self.extractor = {"model": model, "version": parser.meta.get("version"),
                              "kind": "named-entities+noun-phrases+dependency-predicates"}

            def documents():
                yield from parser.pipe(texts, batch_size=48)
        # Process every leaf, not only query-selected or salient sentences.
        for leaf, doc in enumerate(documents()):
            offset = spans[leaf][0]
            chinese = doc.lang_ == "zh"
            chunks = chinese_noun_phrases(doc) if chinese else list(doc.noun_chunks)
            by_sentence = defaultdict(list)
            for chunk in chunks:
                tokens = [t for t in chunk if not t.is_space and not t.is_punct]
                while tokens and tokens[0].pos_ in {"DET", "PRON"}:
                    tokens.pop(0)
                if not tokens or len(tokens) > 8 or chunk.root.pos_ not in {"NOUN", "PROPN"}:
                    continue
                if len(tokens) == 1 and (tokens[0].is_stop or (not chinese and len(tokens[0].text) < 3)):
                    continue
                start, end = tokens[0].idx, tokens[-1].idx+len(tokens[-1])
                label = doc.text[start:end]
                if len(label) > 100 or not re.search(r"[^\W\d_]" if chinese else r"[A-Za-z]", label):
                    continue
                key = "".join(label.casefold().split()) if chinese else " ".join(t.lemma_.lower() for t in tokens)
                typ = next((e.label_ for e in doc.ents if e.start <= chunk.root.i < e.end and e.label_ not in {"DATE","TIME","MONEY","PERCENT","QUANTITY","CARDINAL","ORDINAL"}), "CONCEPT")
                by_sentence[chunk.sent.start].append((chunk, key, label, typ, start, end))
            for sentence in doc.sents:
                mentions = by_sentence.get(sentence.start, [])
                if not mentions:
                    continue
                pid = f"P{len(self.propositions)}"
                proposition = {"id": pid, "text": sentence.text, "start": offset+sentence.start_char,
                               "end": offset+sentence.end_char, "passage_id": f"C{leaf}",
                               "entity_ids": [], "relation_ids": []}
                roots = {}
                for chunk, key, label, typ, start, end in mentions:
                    if key not in self.entity_lookup:
                        eid = f"E{len(self.entities)}"
                        self.entity_lookup[key] = eid
                        self.entities.append({"id": eid, "name": label, "canonical": key, "type": typ,
                                              "aliases": [], "mentions": [], "proposition_ids": [], "passage_ids": []})
                    eid = self.entity_lookup[key]
                    entity = self.entities[int(eid[1:])]
                    if label not in entity["aliases"]:
                        entity["aliases"].append(label)
                    entity["mentions"].append({"passage_id": f"C{leaf}", "proposition_id": pid,
                                               "start": offset+start, "end": offset+end, "text": label})
                    if pid not in entity["proposition_ids"]:
                        entity["proposition_ids"].append(pid)
                    if not entity["passage_ids"] or entity["passage_ids"][-1] != f"C{leaf}":
                        entity["passage_ids"].append(f"C{leaf}")
                    if eid not in proposition["entity_ids"]:
                        proposition["entity_ids"].append(eid)
                    if eid not in self.passage_entities[leaf]:
                        self.passage_entities[leaf].append(eid)
                    roots[chunk.root.i] = eid

                def resolve(token):
                    if token.i in roots:
                        return roots[token.i]
                    return None

                def add_relation(subject, predicate, obj, kind="predicate", negated=False, modality=None):
                    if not subject or not obj or subject == obj:
                        return
                    times = [e.text for e in sentence.ents if e.label_ in {"DATE", "TIME"}]
                    key = (subject, predicate, obj, kind, negated, tuple(modality or []), tuple(times))
                    if key not in relation_lookup:
                        rid = f"R{len(self.relations)}"
                        relation_lookup[key] = rid
                        relation = {"id": rid, "subject": subject, "predicate": predicate, "object": obj,
                                    "kind": kind, "negated": negated, "modality": modality or [],
                                    "time": times, "proposition_ids": [], "passage_ids": [],
                                    "status": "automatically_extracted_not_verified"}
                        self.relations.append(relation)
                        self.adjacency[subject].append((rid, obj, "outgoing"))
                        self.adjacency[obj].append((rid, subject, "incoming"))
                    rid = relation_lookup[key]
                    relation = self.relations[int(rid[1:])]
                    if pid not in relation["proposition_ids"]:
                        relation["proposition_ids"].append(pid)
                    if f"C{leaf}" not in relation["passage_ids"]:
                        relation["passage_ids"].append(f"C{leaf}")
                    if rid not in proposition["relation_ids"]:
                        proposition["relation_ids"].append(rid)

                for verb in sentence:
                    if verb.pos_ not in {"VERB", "AUX"}:
                        continue
                    children = list(verb.children)
                    subjects = [c for c in children if c.dep_ in {"nsubj", "csubj"}]
                    passive = [c for c in children if c.dep_ in {"nsubjpass", "csubjpass"}]
                    objects = [c for c in children if c.dep_ in {"dobj", "obj", "attr", "oprd"}]
                    agents = [o for c in children if c.dep_ == "agent" for o in c.children if o.dep_ == "pobj"]
                    if passive and agents:
                        subjects, objects = agents, passive
                    if not subjects and verb.dep_ == "conj":
                        subjects = [c for c in verb.head.children if c.dep_ == "nsubj"]
                    negated = any(c.dep_ == "neg" or (chinese and c.dep_ == "advmod" and c.text in {"不", "没", "没有", "未", "并未", "从未"}) for c in children)
                    modality = [c.text for c in children if c.tag_ == "MD" or (chinese and c.dep_ == "aux:modal")]
                    predicate = verb.text.casefold() if chinese else verb.lemma_.lower()+"".join("_"+c.lemma_ for c in children if c.dep_ == "prt")
                    for subject in subjects:
                        for obj in objects:
                            add_relation(resolve(subject), predicate, resolve(obj), negated=negated, modality=modality)
                        for prep in children:
                            if prep.dep_ == "prep":
                                for obj in prep.children:
                                    if obj.dep_ == "pobj":
                                        add_relation(resolve(subject), predicate+"_"+prep.lemma_.lower(), resolve(obj), negated=negated, modality=modality)
                # A separate, visibly weaker relation permits navigating mentions
                # when no predicate could be extracted. Never call it entailment.
                ids = proposition["entity_ids"][:12]
                if not proposition["relation_ids"]:
                    for i, eid in enumerate(ids):
                        for other in ids[i+1:i+4]:
                            add_relation(eid, "co_mentioned", other, kind="co_mention")
                self.propositions.append(proposition)
                self.passage_propositions[leaf].append(pid)
            if progress and (leaf+1) % 100 == 0:
                progress(leaf+1, len(texts))
        self.postings = defaultdict(list)
        for entity in self.entities:
            text = entity["canonical"]+" "+" ".join(self.propositions[int(p[1:])]["text"][:300]
                                                               for p in entity["proposition_ids"][:3])
            for term, count in Counter(terms(text)).items():
                self.postings[term].append((entity["id"], count))

    def entity(self, eid):
        if not re.fullmatch(r"E\d+", str(eid)) or int(eid[1:]) >= len(self.entities):
            raise ValueError(f"Unknown entity ID {eid!r}; select one returned by graph_entities or passage entity_ids")
        return self.entities[int(eid[1:])]

    def describe(self, eid):
        e = self.entity(eid)
        return {k: e[k] for k in ("id", "name", "canonical", "type")} | {
            "aliases":e["aliases"][:12],"passage_ids":e["passage_ids"][:40],
            "source_count":len(e["passage_ids"]),
            "mention_count": len(e["mentions"]), "relation_count": len(self.adjacency[eid])}

    def search(self, query, top_k=8):
        scores = defaultdict(float)
        total = len(self.entities)
        for term in sorted(set(terms(query))):
            posting = self.postings.get(term, [])
            weight = math.log(1+total/(1+len(posting)))
            for eid, tf in posting:
                scores[eid] += weight*(1+math.log(tf))
        query_terms = set(terms(query))
        for eid in scores:
            e = self.entity(eid)
            own = set(terms(e["canonical"]))
            scores[eid] += 4*len(own & query_terms)/max(1,len(own))
        return [{**self.describe(eid), "score": scores[eid]} for eid in
                sorted(scores, key=lambda x: (-scores[x], int(x[1:])))[:top_k]]

    def neighbors(self, seeds, query="", hops=1, top_k=12, kinds=None, direction="both", exclude_passages=None,
                  *, eligible_edges_first=False):
        for eid in seeds:
            self.entity(eid)
        allowed = set(kinds or ["predicate", "co_mention"])
        frontier = [(e, []) for e in seeds]
        visited = set(seeds)
        paths, found_relations, proposition_scores = {}, {}, defaultdict(float)
        query_terms = set(terms(query))
        excluded = set(exclude_passages or [])
        # Shared entity identity itself is a valid bridge across passages.
        for eid in seeds:
            for pid in self.entity(eid)["proposition_ids"]:
                proposition_scores[pid] = 1.0
        for depth in range(max(1,min(3,int(hops)))):
            next_frontier = []
            for eid, path in frontier:
                edges = self.adjacency.get(eid, [])
                if eligible_edges_first:
                    edges = [edge for edge in edges
                             if self.relations[int(edge[0][1:])]["kind"] in allowed
                             and direction in ("both", edge[2])]
                def rank(edge):
                    r = self.relations[int(edge[0][1:])]
                    text = r["predicate"]+" "+self.entity(edge[1])["canonical"]
                    return (len(query_terms & set(terms(text)))+(2 if r["kind"] == "predicate" else 0), -int(edge[0][1:]))
                for rid, target, traversal in sorted(edges,key=rank,reverse=True)[:32]:
                    relation = self.relations[int(rid[1:])]
                    if relation["kind"] not in allowed or (direction != "both" and direction != traversal):
                        continue
                    hop = {"from": eid, "to": target, "relation_id": rid, "direction": traversal}
                    current_path = path+[hop]
                    found_relations[rid] = relation
                    if target not in paths:
                        paths[target] = current_path
                    for pid in relation["proposition_ids"]:
                        proposition_scores[pid] = max(proposition_scores[pid], 2/(depth+1))
                    for pid in self.entity(target)["proposition_ids"]:
                        proposition_scores[pid] = max(proposition_scores[pid], 1/(depth+1))
                    if target not in visited:
                        visited.add(target)
                        next_frontier.append((target,current_path))
            frontier = next_frontier[:48]
        for pid in proposition_scores:
            p = self.propositions[int(pid[1:])]
            proposition_scores[pid] += len(query_terms & set(terms(p["text"]))) / max(1, len(query_terms))
        ranked = sorted(proposition_scores, key=lambda p: (self.propositions[int(p[1:])]["passage_id"] in excluded,
                                                          -proposition_scores[p], int(p[1:])))
        selected, leaf_ids = [], []
        for pid in ranked:
            p = self.propositions[int(pid[1:])]
            selected.append(p)
            if p["passage_id"] not in leaf_ids:
                leaf_ids.append(p["passage_id"])
            if len(leaf_ids) >= top_k:
                break
        entity_ids = list(dict.fromkeys([*seeds,*paths]))[:64]
        related_ids = {r for p in selected for r in p["relation_ids"]}
        for rid in sorted(related_ids,key=lambda value:int(value[1:])):
            found_relations[rid] = self.relations[int(rid[1:])]
        # Keep graph endpoints present even when the output is bounded.
        for r in found_relations.values():
            for eid in (r["subject"],r["object"]):
                if eid not in entity_ids:
                    entity_ids.append(eid)
        return {"entities": [self.describe(e) for e in entity_ids],
                "relations": list(found_relations.values()), "propositions": selected,
                "paths": [{"entity_id": eid, "steps": path} for eid,path in paths.items()],
                "passage_ids": leaf_ids, "scores": {p["passage_id"]: proposition_scores[p["id"]] for p in selected}}
