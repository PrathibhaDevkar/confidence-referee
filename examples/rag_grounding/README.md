# RAG-grounding proof-of-concept

Validates confidence_referee's `GroundingSignal` against a second, genuinely
different domain from meetings - see the main README's "Validated against
a second domain" section for the result and what it proves.

`sample_data.jsonl` is a 300-example random sample (seed=42) of
[HaluEval](https://github.com/RUCAIBox/HaluEval)'s `qa_data.json`
(RUCAIBox, MIT license) - included here only as a small, fixed slice for
a reproducible proof-of-concept, not the full dataset.

Run it: `python3 run_poc.py` (from this directory) or
`python3 examples/rag_grounding/run_poc.py` (from the repo root).
