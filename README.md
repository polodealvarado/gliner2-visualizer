# Inside GLiNER2.5

An interactive, didactic 3D view of GLiNER2.5 for non-technical readers.
You type a sentence and a schema: entity types, attributes, relations, a structure and classification labels. The real model answers all of it in one pass, and the page draws what each stage produces, from the pieces of the text to the final answers.

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

One tab per task: Entities, Classification, Relations, Structure, Attributes and All tasks. Each tab shows only the fields and examples of its task.

The tower is the same at the bottom for every task: text and schema, tokenizer, embeddings, 12 Transformer layers, and word and query vectors. The span stages follow (gaps between words, "starts? ends?", candidates, scoring), shared by entity types, attribute labels, relation roles and structure fields. On top sits the head of the task: entity decision, attributes, head × tail pairs and relation scores, record instances and field assignment, or label scores and label decision. "All tasks" stacks every head and ends with all the answers.
Each stage has three tabs: "What it does" (plain language), "With your sentence" (the real values) and "Under the hood" (shapes, parameters and the source file).
The "Meaning map" view shows how each piece moves through the model's space, layer by layer.

## Schema syntax

| Field | Example | Meaning |
|---|---|---|
| Entities | `person = named people, company` | Entity types; an optional description follows `=`. |
| Attributes | `mood: happy, annoyed → person` | A label group scored at each entity it applies to. Separate groups with `;`. |
| Classification | `topic: tech, sports; tone: good, bad` | Labels for the whole text. Separate tasks with `;`. |
| Relations | `works_for, located_in` | Relation types; each one has a head and a tail role. |
| Structure | `launch: product*, company, place` | Record fields; `*` marks the anchor (default: the first field). |

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
- Each task is decoded on its own. GLiNER2's constrained decoders (`Classifier` rules between tasks, `JointIE` typed relation graphs) are not shown.
- Record plates list at most 8 instances per structure, and 8 candidates per field.
- Limits: 60 words, 12 span queries and 3 classification tasks per sentence, so the drawing stays readable.

## Credits

GLiNER2 and the GLiNER2.5 checkpoints are by [Fastino AI](https://github.com/fastino-ai/GLiNER2) (Apache-2.0; paper: [arXiv:2507.18546](https://arxiv.org/abs/2507.18546)).
This visualization is an independent project and is not affiliated with Fastino.
