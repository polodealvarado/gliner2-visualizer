# Inside GLiNER2.5

An interactive, didactic 3D view of GLiNER2.5 for non-technical readers.
You type a sentence and the labels you are looking for. The real model runs, and the page draws what each stage produces, from the pieces of the text to the final entities.

## Start

```bash
uv run server.py            # base model fastino/gliner2.5-multi-v1
open http://localhost:8765
```

`uv` installs the dependencies declared in the header of `server.py`.
The first run downloads the model from Hugging Face (~1.1 GB).

| Option | Use |
|---|---|
| `--model <repo>` | Another GLiNER2.5 *boundary* checkpoint, e.g. `fastino/gliner2.5-base-v1`. |
| `--port <n>` | Port (default 8765). |
| `--check` | Runs one sentence, verifies the output and exits. |

## What you see

Ten stages stacked in a tower, from bottom to top: text and labels, tokenizer, embeddings, 12 Transformer layers, word and label vectors, gaps between words, "starts? ends?", candidates, scoring and decision.
Each stage has three tabs: "What it does" (plain language), "With your sentence" (the real values) and "Under the hood" (shapes, parameters and the source file).
The "Meaning map" view shows how each piece moves through the model's space, layer by layer.

## Files

- `server.py`: loads the model, captures the intermediate outputs with hooks and serves them at `/api/analyze`. It captures the encoder hidden states and attention, gaps, start/end/inside logits, the candidate pool and its scores, abstention, expected counts, classification logits, relation pairs and their scores, attribute scores and record instances.
- `index.html`: the page (Three.js from a CDN, no build step).

## API

`POST /api/analyze` runs one `model.extract()` call with any mix of tasks:

```json
{
  "text": "Tim Cook, the CEO of Apple, happily presented the iPhone 17 in Cupertino.",
  "threshold": 0.5,
  "entities": {"person": "Named people", "company": null, "product": null},
  "attributes": {"mood": {"labels": ["happy", "annoyed"], "applies_to": ["person"]}},
  "classifications": {"topic": ["technology", "sports", "politics"]},
  "relations": ["works_for", "located_in"],
  "structures": {"launch": {"fields": ["product", "company", "place"], "anchor": "product"}}
}
```

`{"text": "...", "labels": ["person", "city"]}` is the entity-only shorthand.
Limits: 60 words; 12 span queries in total (entity types + attribute labels + 2 per relation + structure fields); 3 classification tasks with 2 to 8 labels each.

## Simplifications

- Attention is shown as the average of the 12 heads.
- The map is a 3D PCA projection, so distances are approximate. The similarity figures use the real 768 numbers.
- Only entity extraction is walked through. The model also classifies text and extracts relations, records and span attributes.
- Limits: 60 words and 8 labels per sentence, so the drawing stays readable.

## Credits

GLiNER2 and the GLiNER2.5 checkpoints are by [Fastino AI](https://github.com/fastino-ai/GLiNER2) (Apache-2.0; paper: [arXiv:2507.18546](https://arxiv.org/abs/2507.18546)).
This visualization is an independent project and is not affiliated with Fastino.
