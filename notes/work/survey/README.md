# Survey

Literature survey for long-context speculative decoding. Read 1 then 2. The rest is lookup.

| Order | File | Answers |
|---|---|---|
| 1 | [landscape.md](landscape.md) | What is known: settled facts, contested questions, the arithmetic |
| 2 | [ideas-kept.md](ideas-kept.md) | What we bet on: ranked ideas with kill tests and risks |
| — | [ideas-rejected.md](ideas-rejected.md) | Why dead directions are dead (append-only) |
| — | [bibliography.md](bibliography.md) | Which paper said what (generated reference) |

Notation: α = acceptance rate, τ = accepted tokens/round. Numbers as published; a speedup means nothing without its baseline, context length, and batch size.

The bibliography is generated. Edit [data/papers.yaml](data/papers.yaml), then run `.venv/bin/python notes/work/survey/data/build.py`. Never edit bibliography.md by hand. [data/raw-data.json](data/raw-data.json) is the raw survey output, write-once.
