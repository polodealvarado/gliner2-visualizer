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
from gliner2 import AutoExtractor

HERE = Path(__file__).parent
MAX_WORDS, MAX_LABELS, STRIP = 60, 8, 32
CAP: dict = {}  # ponytail: global capture dict, safe because HTTPServer serves one request at a time


def r(x, nd=3):
    """Tensor -> nested lists of rounded floats (smaller JSON)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().tolist()
    if isinstance(x, list):
        return [r(v, nd) for v in x]
    return round(x, nd) if math.isfinite(x) else 0.0


def load(name):
    model = AutoExtractor.from_pretrained(name)
    model.eval()
    enc_forward = model.encoder.forward

    def encoder_forward(*a, **k):
        k.update(output_hidden_states=True, output_attentions=True)
        CAP["enc"] = out = enc_forward(*a, **k)
        return out

    model.encoder.forward = encoder_forward
    core = model._encode_core

    def encode_core(batch):
        CAP["batch"] = batch
        return core(batch)

    model._encode_core = encode_core
    head = model.boundary_head
    for n in ("boundary_encoder", "boundary_query_head", "shared_pool_builder",
              "shared_pool_scorer", "null_projection"):
        getattr(head, n).register_forward_hook(lambda m, i, o, n=n: CAP.__setitem__(n, o))
    return model


def params(module):
    return sum(p.numel() for p in module.parameters())


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
        "params": {
            "total": params(model),
            "embeddings": params(enc.embeddings),
            "layer": params(enc.encoder.layer[0]),
            "encoder": params(enc),
            "boundary_encoder": params(head.boundary_encoder),
            "query_head": params(head.boundary_query_head),
            "pool": params(head.shared_pool_builder),
            "scorer": params(head.shared_pool_scorer),
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


def analyze(model, text, labels, threshold):
    CAP.clear()
    t0 = time.perf_counter()
    with torch.inference_mode():
        result = model.extract_entities(
            text, labels, threshold=threshold, include_confidence=True, include_spans=True
        )
    ms = (time.perf_counter() - t0) * 1000
    b, enc = CAP["batch"], CAP["enc"]
    tok = model.processor.tokenizer
    ids = b.input_ids[0].tolist()
    pieces = tok.convert_ids_to_tokens(ids)
    word_tok = b.text_word_indices[0][b.text_word_mask[0]].tolist()
    marker_tok = b.query_marker_indices[0][b.query_marker_mask[0]].tolist()
    starts, ends = b.start_mappings[0], b.end_mappings[0]
    src = b.original_texts[0]
    n_words = len(word_tok)
    sep = pieces.index("[SEP_TEXT]")
    special = set(model.processor.SPECIAL_TOKENS)

    tokens = []
    for i, (p, tid) in enumerate(zip(pieces, ids)):
        kind = "text" if i > sep else ("marker" if p in special else "schema")
        word = None
        if i > sep:
            word = max(j for j, t in enumerate(word_tok) if t <= i) if i >= word_tok[0] else None
        label = marker_tok.index(i) if i in marker_tok else None
        tokens.append({"piece": p, "id": tid, "kind": kind, "word": word, "label": label,
                       "first": i in word_tok})
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
    candidates = [{"s": s, "e": e, "text": src[starts[s]:ends[e - 1]], "cs": starts[s], "ce": ends[e - 1],
                   "prior": round(float(pv), 3), "probs": r(pr)}
                  for (s, e), pv, pr in zip(idx, prior, probs)]
    candidates.sort(key=lambda c: -max(c["probs"]))

    entities = []
    for label, items in result.get("entities", {}).items():
        for it in items:
            entities.append({"label": labels.index(label), "text": it["text"], "start": it["start"],
                             "end": it["end"], "score": round(it["confidence"], 4)})
    entities.sort(key=lambda e: e["start"])

    return {
        "ms": round(ms),
        "input": {"text": text, "labels": labels, "threshold": threshold},
        "tokens": tokens,
        "words": words,
        "markers": marker_tok,
        "layers": layers,
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
        "abstention_threshold": model.boundary_settings.abstention_threshold,
        "entities": entities,
    }


def validate(body):
    text = str(body.get("text", "")).strip()
    labels = [str(x).strip()[:40] for x in body.get("labels", []) if str(x).strip()]
    labels = list(dict.fromkeys(labels))
    threshold = float(body.get("threshold", 0.5))
    if not text or len(text.split()) > MAX_WORDS:
        raise ValueError(f"Write a sentence of 1 to {MAX_WORDS} words.")
    if not 1 <= len(labels) <= MAX_LABELS:
        raise ValueError(f"Use 1 to {MAX_LABELS} labels.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("The threshold must be between 0 and 1.")
    return text[:600], labels, threshold


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
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as e:
            self.send_json(400, {"error": str(e)})
        except Exception as e:  # keep the page alive and show the error
            self.send_json(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fastino/gliner2.5-multi-v1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--check", action="store_true", help="run one sentence, verify the output and exit")
    args = ap.parse_args()
    print(f"Loading {args.model}…")
    MODEL = load(args.model)
    INFO = model_info(MODEL, args.model)
    if args.check:
        out = analyze(MODEL, *validate({"text": "Maria Garcia lives in Madrid.", "labels": ["person", "city"]}))
        n_words, n_tok = len(out["words"]), len(out["tokens"])
        assert len(out["layers"]) == INFO["layers"] + 1
        assert all(len(l["attn"]) == n_tok for l in out["layers"][1:])
        assert len(out["boundary"]["start"][0]) == n_words + 1 and len(out["boundary"]["inside"][0]) == n_words
        assert {(e["label"], e["text"]) for e in out["entities"]} >= {(0, "Maria Garcia"), (1, "Madrid")}, out["entities"]
        raise SystemExit(f"OK: {n_tok} tokens, {n_words} words, {len(out['candidates'])} candidates, {out['ms']} ms")
    print(f"Ready: http://localhost:{args.port}")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
