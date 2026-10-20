# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "gliner2==2.0.0",
#     "torch==2.13.0",
#     "transformers>=5.9,<5.17",
#     "huggingface-hub>=1.5,<2",
#     "safetensors>=0.4,<1",
#     "peft>=0.17,<1",
#     "numpy>=1.24,<3",
#     "sentencepiece>=0.2",
#     "protobuf>=4",
# ]
# ///
"""Run GLiNER2.5 on one sentence and return what every stage of the model produced.

    uv run server.py [--model fastino/gliner2.5-multi-v1] [--port 8765]

Then open http://localhost:8765. The page (index.html) draws the result in 3D.
"""

import argparse
import json
import math
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import torch
from gliner2 import AttributeGroup, AutoExtractor

HERE = Path(__file__).parent
MAX_WORDS, MAX_QUERIES, MAX_CLS, MAX_CHOICES, STRIP = 60, 12, 3, 8, 32
TOP = 8  # ponytail: record instances and field candidates sent per field; raise it if a schema needs more
CAP: dict = {}  # ponytail: global capture dict, safe because HTTPServer serves one request at a time


def r(x, nd=3):
    """Tensor -> nested lists of rounded floats (smaller JSON)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().tolist()
    if isinstance(x, list):
        return [r(v, nd) for v in x]
    return round(x, nd) if math.isfinite(x) else 0.0


def tap(obj, attr, store):
    """Wrap obj.attr so that store(output, *args) sees every call."""
    fn = getattr(obj, attr)

    def call(*a, **k):
        out = fn(*a, **k)
        store(out, *a)
        return out

    setattr(obj, attr, call)


def load(name):
    model = AutoExtractor.from_pretrained(name)
    model.eval()
    enc_forward = model.encoder.forward

    def encoder_forward(*a, **k):
        k.update(output_hidden_states=True, output_attentions=True)
        CAP["enc"] = out = enc_forward(*a, **k)
        return out

    model.encoder.forward = encoder_forward
    tap(model, "_encode_core", lambda out, batch: CAP.update(batch=batch, core=out))
    head = model.boundary_head

    def keep_first(n):
        def hook(m, i, o):
            CAP.setdefault(n, o)  # no return: a hook that returns a value replaces the module output
        return hook

    # Span attributes re-run some of these modules on the decided entities, so keep the first (main) pass.
    for n in ("boundary_encoder", "boundary_query_head", "shared_pool_builder",
              "shared_pool_scorer", "null_projection", "count_head"):
        if getattr(head, n, None) is not None:
            getattr(head, n).register_forward_hook(keep_first(n))
    model.classifier.register_forward_hook(lambda m, i, o: CAP.setdefault("cls", []).append(o))

    attach = model._attach_entity_attributes

    def attach_attributes(*a, **k):
        CAP["in_attr"] = True
        try:
            return attach(*a, **k)
        finally:
            CAP["in_attr"] = False

    model._attach_entity_attributes = attach_attributes
    tap(head, "score_explicit_spans",
        lambda out, *a: CAP.get("in_attr") and CAP.__setitem__("attr", (a[4][0, 0], out[0])))
    if model.enable_relations:
        tap(model.relation_pair_generator, "generate", lambda out, *a: CAP.__setitem__("rel_pairs", out))
        model.relation_scorer.register_forward_hook(lambda m, i, o: CAP.__setitem__("rel_logits", o))
    if model.enable_records:
        tap(model.record_decoder, "forward_group", lambda out, *a: CAP.setdefault("records", []).append(out))
    return model


def params(module):
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def model_info(model, name):
    enc, s = model.encoder, model.boundary_settings
    head = model.boundary_head
    return {
        "name": name,
        "encoder": enc.config.model_type,
        "layers": enc.config.num_hidden_layers,
        "hidden": enc.config.hidden_size,
        "heads": enc.config.num_attention_heads,
        "vocab": enc.config.vocab_size,
        "boundary_dim": s.boundary_dim,
        "pool_top_k": s.pool_boundary_top_k,
        "pool_size": s.pool_size,
        "boundary_attention_layers": s.boundary_attention_layers,
        "boundary_attention_window": s.boundary_attention_window,
        "relations": {"enabled": model.enable_relations, "heads": s.relation_heads_per_type,
                      "tails": s.relation_tails_per_type, "pair_cap": s.relation_pair_cap,
                      "argument_threshold": s.relation_argument_proposal_threshold},
        "records": {"enabled": model.enable_records, "dim": s.record_dim,
                    "instance_queries": s.record_instance_queries},
        "params": {
            "total": params(model),
            "embeddings": params(enc.embeddings),
            "layer": params(enc.encoder.layer[0]),
            "encoder": params(enc),
            "boundary_encoder": params(head.boundary_encoder),
            "query_head": params(head.boundary_query_head),
            "pool": params(head.shared_pool_builder),
            "scorer": params(head.shared_pool_scorer),
            "classifier": params(model.classifier),
            "relation_scorer": params(getattr(model, "relation_scorer", None)),
            "record_decoder": params(getattr(model, "record_decoder", None)),
        },
    }


def pca3(x, prev=None):
    """3D PCA of [N, H], rotated to match the previous layer (Procrustes) so points move smoothly."""
    x = torch.nn.functional.normalize(x.float(), dim=-1)
    x = x - x.mean(0)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    y = x @ v[:3].T
    y = y / y.pow(2).sum(-1).mean().sqrt().clamp_min(1e-6)
    if prev is not None:
        u, _, vt = torch.linalg.svd(y.T @ prev)
        y = y @ (u @ vt)
    return y


def build_schema(model, spec):
    s = model.create_schema()
    if spec["entities"]:
        s.entities({n: d or {} for n, d in spec["entities"].items()})
    if spec["attributes"]:
        s.entity_attributes({g: AttributeGroup(v["labels"], multi_label=v["multi_label"],
                                               applies_to=v["applies_to"], qualify_labels=True)
                             for g, v in spec["attributes"].items()})
    for task, v in spec["classifications"].items():
        s.classification(task, v["labels"], multi_label=v["multi_label"])
    if spec["relations"]:
        s.relations(spec["relations"])
    for name, v in spec["structures"].items():
        b = s.structure(name, mode="natural", anchor=v["anchor"])
        for f in v["fields"]:
            b.field(f, dtype="str", cardinality="required_one" if f == v["anchor"] else None)
    s.build()
    return s


def analyze(model, text, spec, threshold):
    CAP.clear()
    schema = build_schema(model, spec)
    t0 = time.perf_counter()
    with torch.inference_mode():
        result = model.extract(text, schema, threshold=threshold, include_confidence=True, include_spans=True)
    ms = (time.perf_counter() - t0) * 1000
    b, enc, core = CAP["batch"], CAP["enc"], CAP["core"]
    # ponytail: word gaps index the text directly; a non-zero offset would shift every span drawn
    assert core["word_offsets"][0] == 0, core["word_offsets"]
    tok = model.processor.tokenizer
    ids = b.input_ids[0].tolist()
    pieces = tok.convert_ids_to_tokens(ids)
    word_tok = b.text_word_indices[0][b.text_word_mask[0]].tolist()
    marker_tok = b.query_marker_indices[0][b.query_marker_mask[0]].tolist()
    cls_tok = b.cls_marker_indices[0][b.cls_marker_mask[0]].tolist()
    starts, ends = b.start_mappings[0], b.end_mappings[0]
    src = b.original_texts[0]
    n_words = len(word_tok)
    sep = pieces.index("[SEP_TEXT]")
    special = set(model.processor.SPECIAL_TOKENS)
    chars = lambda s, e: (starts[s], ends[e - 1])  # half-open word span -> character span
    cut = lambda s, e: src[slice(*chars(s, e))]

    queries = [{"type": q.task_type, "task": q.task_name, "role": q.role_name, "seg": q.task_index}
               for q in b.query_layouts[0].queries]
    names = [q["role"] if q["type"] == "entities" else f"{q['task']} · {q['role']}" for q in queries]
    tasks = [{"type": t, "name": b.schema_tokens_list[0][g][2].split(" [DESCRIPTION]")[0]}  # drop label descriptions
             for g, t in enumerate(b.task_types[0])]
    cls_names = list(spec["classifications"])
    cls_of, k = {}, 0  # [L] token -> [classification task, choice]
    for ti, task in enumerate(cls_names):
        for j in range(len(spec["classifications"][task]["labels"])):
            cls_of[cls_tok[k]] = [ti, j]
            k += 1

    tokens, seg = [], 0
    for i, (p, tid) in enumerate(zip(pieces, ids)):
        kind = "text" if i > sep else ("marker" if p in special else "schema")
        word = None
        if i > sep:
            word = max(j for j, t in enumerate(word_tok) if t <= i) if i >= word_tok[0] else None
        label = marker_tok.index(i) if i in marker_tok else None
        tokens.append({"piece": p, "id": tid, "kind": kind, "word": word, "label": label,
                       "first": i in word_tok, "seg": seg if i < sep else None, "cls": cls_of.get(i)})
        seg += p == "[SEP_STRUCT]"
    words = [{"text": src[starts[j]:ends[j]], "start": starts[j], "end": ends[j], "token": word_tok[j]}
             for j in range(n_words)]

    layers, prev = [], None
    for li, h in enumerate(enc.hidden_states):
        h = h[0]
        pos = pca3(h, prev)
        prev = pos
        hw, hq = h[word_tok], h[marker_tok]
        sim = torch.nn.functional.cosine_similarity(hq[:, None], hw[None], dim=-1)  # [Q, W]
        layer = {"pos": r(pos), "strip": r(h[:, :STRIP], 2), "label_sim": r(sim)}
        if li > 0:
            att = enc.attentions[li - 1][0].mean(0)  # [T, T], averaged over heads
            # how much each label marker looks at each word (summed over the word's pieces)
            per_word = torch.zeros(len(marker_tok), n_words)
            for i, t in enumerate(tokens):
                if t["word"] is not None:
                    per_word[:, t["word"]] += att[marker_tok, i]
            layer.update(attn=r(att), label_attn=r(per_word))
        layers.append(layer)

    out = {
        "ms": round(ms),
        "input": {"text": text, "labels": names, "threshold": threshold, "schema": spec},
        "tasks": tasks,
        "queries": queries,
        "tokens": tokens,
        "words": words,
        "markers": marker_tok,
        "layers": layers,
        "abstention_threshold": model.boundary_settings.abstention_threshold,
        "result": result,
    }
    out.update(span_stage(model, n_words, chars, cut) if marker_tok else
               {"boundary": None, "pool": None, "candidates": [], "null": None, "count": []})
    ent_q = {q["role"]: i for i, q in enumerate(queries) if q["type"] == "entities"}
    entities = [{"label": ent_q[label], "text": it["text"], "start": it["start"], "end": it["end"],
                 "score": round(it["confidence"], 4)}
                for label, items in result.get("entities", {}).items() for it in items]
    out["entities"] = sorted(entities, key=lambda e: e["start"])
    out["classifications"] = classification_stage(model, spec, cls_tok, result)
    out["attributes"] = attribute_stage(model, spec, out["entities"], names, chars)
    out["relations"] = relation_stage(model, result, chars, cut)
    out["records"] = record_stage(model, result, threshold, cut)
    return out


def span_stage(model, n_words, chars, cut):
    """Gaps, start/end/inside marks, shared candidate pool and its scores (all extractive queries)."""
    benc = CAP["boundary_encoder"].states[0]  # [W+1, d]
    marg = CAP["boundary_query_head"]
    pool = CAP["shared_pool_builder"]
    score = CAP["shared_pool_scorer"][0][0]  # [C, Q]
    temp = model.boundary_settings.pair_temperature
    null = torch.sigmoid(CAP["null_projection"][0, :, 0]) if "null_projection" in CAP else None
    valid = pool.mask[0]
    idx = pool.indices[0][valid].tolist()
    prior = (pool.compat_logits if pool.compat_logits is not None else pool.proposal_logits)
    prior = prior[0][valid] if prior is not None else torch.zeros(len(idx))
    probs = torch.sigmoid(score[valid] / temp)
    candidates = [{"s": s, "e": e, "text": cut(s, e), "cs": chars(s, e)[0], "ce": chars(s, e)[1],
                   "prior": round(float(pv), 3), "probs": r(pr)}
                  for (s, e), pv, pr in zip(idx, prior, probs)]
    candidates.sort(key=lambda c: -max(c["probs"]))
    return {
        "boundary": {
            "strip": r(benc[:, :STRIP], 2),
            "start": r(torch.sigmoid(marg.start_logits[0, :, :n_words + 1])),
            "end": r(torch.sigmoid(marg.end_logits[0, :, :n_words + 1])),
            "inside": r(torch.sigmoid(marg.inside_logits[0, :, :n_words])),
        },
        "pool": {"possible": n_words * (n_words + 1) // 2, "kept": len(idx),
                 "starts": sorted({s for s, _ in idx}), "ends": sorted({e for _, e in idx})},
        "candidates": candidates,
        "null": r(null) if null is not None else None,
        # the count head predicts a log-rate per query; exp() is the expected number of mentions
        "count": r(CAP["count_head"][0, :, 0].exp(), 2) if "count_head" in CAP else [],
    }


def classification_stage(model, spec, cls_tok, result):
    """[L] marker vectors -> shared MLP -> one logit per label -> softmax (one label) or sigmoid (multi-label)."""
    out, k = [], 0
    temp = model.boundary_settings.classification_temperature
    for (task, cfg), logits in zip(spec["classifications"].items(), CAP.get("cls", [])):
        lg = logits.squeeze(-1) / temp
        probs = torch.sigmoid(lg) if cfg["multi_label"] else torch.softmax(lg, -1)
        got = result.get(task) or []
        got = got if isinstance(got, list) else [got]
        n = len(cfg["labels"])
        out.append({"task": task, "labels": cfg["labels"], "multi_label": cfg["multi_label"], "threshold": 0.5,
                    "tokens": cls_tok[k:k + n], "logits": r(lg), "probs": r(probs),
                    "chosen": [cfg["labels"].index(g["label"]) for g in got]})
        k += n
    return out


def attribute_stage(model, spec, entities, names, chars):
    """Attribute labels re-scored at the exact spans of the decided entities, then softmax within each group."""
    if "attr" not in CAP:
        return []
    spans, logits = CAP["attr"]  # [P, 2] word spans, [R, P] one row per attribute label
    logits = logits / model.boundary_settings.pair_temperature
    at = {}
    for e in entities:
        at.setdefault((e["start"], e["end"]), []).append(e)
    out, row = [], 0
    for group, cfg in spec["attributes"].items():
        n = len(cfg["labels"])
        block, row = logits[row:row + n], row + n
        items = []
        for p, (s, e) in enumerate(spans.tolist()):
            for ent in at.get(chars(s, e), []):
                if cfg["applies_to"] and names[ent["label"]] not in cfg["applies_to"]:
                    continue
                probs = torch.sigmoid(block[:, p]) if cfg["multi_label"] else torch.softmax(block[:, p], -1)
                items.append({"start": ent["start"], "end": ent["end"], "label": ent["label"], "probs": r(probs)})
        out.append({"group": group, "labels": cfg["labels"], "applies_to": cfg["applies_to"],
                    "multi_label": cfg["multi_label"], "threshold": 0.5, "items": items})
    return out


def relation_stage(model, result, chars, cut):
    """Head-role candidates x tail-role candidates -> relation scorer -> one probability per pair."""
    pairs = CAP.get("rel_pairs")
    if pairs is None or not len(pairs) or "rel_logits" not in CAP:
        return []
    probs = torch.sigmoid(CAP["rel_logits"].float() / model.boundary_settings.relation_temperature)
    kept = {(t, it["head"]["start"], it["head"]["end"], it["tail"]["start"], it["tail"]["end"])
            for t, items in result.get("relation_extraction", {}).items() for it in items}
    out = []
    for i in range(len(pairs)):
        h = (int(pairs.head_start[i]), int(pairs.head_end[i]))
        t = (int(pairs.tail_start[i]), int(pairs.tail_end[i]))
        typ = pairs.relation_types[i]
        out.append({"type": typ, "head": h, "tail": t, "head_text": cut(*h), "tail_text": cut(*t),
                    "head_prob": round(float(pairs.head_prob[i]), 3), "tail_prob": round(float(pairs.tail_prob[i]), 3),
                    "prob": round(float(probs[i]), 3), "kept": (typ, *chars(*h), *chars(*t)) in kept})
    return sorted(out, key=lambda p: -p["prob"])


def record_stage(model, result, threshold, cut):
    """One instance per anchor mention; each field picks a candidate span (or ABSENT) for that instance."""
    temp = model.boundary_settings.record_temperature
    out = []
    for g in CAP.get("records", []):
        obj = torch.sigmoid(g.object_logits.float() / temp)
        instances = []
        for i in sorted(range(g.num_instances), key=lambda i: -float(obj[i]))[:TOP]:
            fields = []
            for f, fs in enumerate(g.field_specs):
                lg = g.assign_logits[f][i].float() / temp
                scalar = fs.cardinality.is_scalar
                p = torch.softmax(lg, -1) if scalar else torch.sigmoid(lg)  # column 0 is ABSENT
                spans = g.field_spans[f].tolist()
                cands = [c for c in range(len(spans)) if g.field_cand_mask[f][c]]
                cands = sorted(cands, key=lambda c: -float(p[c + 1]))[:TOP]
                fields.append({"absent": round(float(p[0]), 3) if scalar else None,
                               "cands": [{"span": spans[c], "text": cut(*spans[c]), "prob": round(float(p[c + 1]), 3)}
                                         for c in cands]})
            span = g.instance_spans[i]
            instances.append({"span": list(span) if span else None, "text": cut(*span) if span else None,
                              "prob": round(float(obj[i]), 3), "fields": fields})
        out.append({"name": g.spec.task_name, "mode": g.spec.mode,
                    "fields": [{"name": fs.name, "anchor": fs.is_anchor, "scalar": fs.cardinality.is_scalar}
                               for fs in g.field_specs],
                    "threshold": threshold, "instances": instances, "result": result.get(g.spec.task_name, [])})
    return out


def clean(x, n=40):
    return str(x).strip()[:n] if x is not None else ""


def uniq(xs):
    if isinstance(xs, (str, dict)):
        raise TypeError("Expected a list of names.")
    return list(dict.fromkeys(c for c in map(clean, xs) if c))


def validate(body):
    text = clean(body.get("text"), 600)
    if not text or len(text.split()) > MAX_WORDS:
        raise ValueError(f"Write a sentence of 1 to {MAX_WORDS} words.")
    threshold = float(body.get("threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        raise ValueError("The threshold must be between 0 and 1.")
    ents = body.get("entities", body.get("labels", []))  # "labels": entity-only shorthand
    ents = ents if isinstance(ents, dict) else dict.fromkeys(uniq(ents))
    spec = {"entities": {clean(k): clean(v, 120) or None for k, v in ents.items() if clean(k)},
            "attributes": {}, "classifications": {}, "relations": uniq(body.get("relations", [])), "structures": {}}
    for g, v in body.get("attributes", {}).items():
        v = v if isinstance(v, dict) else {"labels": v}
        spec["attributes"][clean(g)] = {"labels": uniq(v["labels"]), "multi_label": bool(v.get("multi_label")),
                                        "applies_to": uniq(v.get("applies_to") or []) or None}
    for t, v in body.get("classifications", {}).items():
        v = v if isinstance(v, dict) else {"labels": v}
        spec["classifications"][clean(t)] = {"labels": uniq(v["labels"]), "multi_label": bool(v.get("multi_label"))}
    for n, v in body.get("structures", {}).items():
        v = v if isinstance(v, dict) else {"fields": v}
        fields = uniq(v["fields"])
        spec["structures"][clean(n)] = {"fields": fields, "anchor": clean(v.get("anchor")) or (fields or [""])[0]}

    n_queries = (len(spec["entities"]) + sum(len(a["labels"]) for a in spec["attributes"].values())
                 + 2 * len(spec["relations"]) + sum(len(s["fields"]) for s in spec["structures"].values()))
    if not n_queries and not spec["classifications"]:
        raise ValueError("Ask for at least one thing: entities, a classification, relations or a structure.")
    if n_queries > MAX_QUERIES:
        raise ValueError(f"Use at most {MAX_QUERIES} entity types, attribute labels, relation roles and structure "
                         "fields in total (each relation counts twice: head and tail).")
    if len(spec["classifications"]) > MAX_CLS:
        raise ValueError(f"Use at most {MAX_CLS} classification tasks.")
    if any(not 2 <= len(c["labels"]) <= MAX_CHOICES for c in spec["classifications"].values()):
        raise ValueError(f"Each classification needs 2 to {MAX_CHOICES} labels.")
    if any(not g or not a["labels"] for g, a in spec["attributes"].items()):
        raise ValueError("Each attribute group needs a name and at least one label.")
    if any(not n or s["anchor"] not in s["fields"] for n, s in spec["structures"].items()):
        raise ValueError("Each structure needs a name and its anchor must be one of its fields.")
    return text, spec, threshold


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def send_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/info":
            return self.send_json(200, INFO)
        if self.path not in ("/", "/index.html"):
            return self.send_error(404)
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/analyze":
            return self.send_error(404)
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 20000)))
            self.send_json(200, analyze(MODEL, *validate(body)))
        except (ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError) as e:
            self.send_json(400, {"error": str(e)})
        except Exception as e:  # keep the page alive and show the error
            self.send_json(500, {"error": f"{type(e).__name__}: {e}"})


def check():
    """One entity-only sentence (what the page sends today) and one sentence with every task."""
    out = analyze(MODEL, *validate({"text": "Maria Garcia lives in Madrid.", "labels": ["person", "city"]}))
    n_words, n_tok = len(out["words"]), len(out["tokens"])
    assert len(out["layers"]) == INFO["layers"] + 1
    assert all(len(l["attn"]) == n_tok for l in out["layers"][1:])
    assert len(out["boundary"]["start"][0]) == n_words + 1 and len(out["boundary"]["inside"][0]) == n_words
    assert {(e["label"], e["text"]) for e in out["entities"]} >= {(0, "Maria Garcia"), (1, "Madrid")}, out["entities"]

    m = analyze(MODEL, *validate({
        "text": "Alice was delighted to join Acme in Paris, but Bob was angry.",
        "entities": {"person": "Named people", "organization": None},
        "attributes": {"sentiment": {"labels": ["positive", "negative"], "applies_to": ["person"]}},
        "classifications": {"topic": ["technology", "business", "sports"]},
        "relations": ["works_for"],
        "structures": {"job": ["employee", "company"]},
    }))
    assert [t["type"] for t in m["tasks"]] == ["json_structures", "entities", "relations", "classifications"], m["tasks"]
    assert len(m["queries"]) == len(m["markers"]) == len(m["count"]) == 8, m["input"]["labels"]
    c = m["classifications"][0]
    assert c["labels"][c["chosen"][0]] == "business" and len(c["tokens"]) == 3, c
    assert all(m["tokens"][k]["cls"] == [0, j] for j, k in enumerate(c["tokens"]))
    mood = {i["start"]: a["labels"][i["probs"].index(max(i["probs"]))] for a in m["attributes"] for i in a["items"]}
    assert mood == {0: "positive", 47: "negative"}, m["attributes"]
    assert any(p["kept"] and (p["head_text"], p["tail_text"]) == ("Alice", "Acme") for p in m["relations"]), m["relations"][:3]
    job = m["records"][0]
    assert job["result"][0]["employee"]["text"] == "Alice" and job["instances"][0]["text"] == "Alice", job
    return (f"OK: {n_tok} tokens, {n_words} words, {len(out['candidates'])} candidates, {out['ms']} ms; "
            f"multi-task: {len(m['queries'])} queries, {len(m['relations'])} relation pairs, {m['ms']} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fastino/gliner2.5-multi-v1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--check", action="store_true", help="run two sentences, verify the output and exit")
    args = ap.parse_args()
    print(f"Loading {args.model}…")
    MODEL = load(args.model)
    INFO = model_info(MODEL, args.model)
    if args.check:
        raise SystemExit(check())
    print(f"Ready: http://localhost:{args.port}")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
