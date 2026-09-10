# confidence-referee

A reusable confidence-calibration layer for LLM extraction pipelines: given
a claim, its source, and (optionally) a second independent extraction to
compare it against, score how much the claim should actually be trusted -
instead of presenting every extracted fact with the same, false certainty.

Generalized out of [confidence-aware-meeting-intelligence](https://github.com/PrathibhaDevkar/confidence-aware-meeting-intelligence),
where this pattern was first built and validated: filtering to the
top-confidence tier nearly doubled action-item extraction precision (58%
vs. a 35% no-filtering baseline) on genuinely held-out data. That project
is now a real integration (see `confidence/referee_adapter.py` there), not
just the source this package was extracted from. A second, independent
domain - RAG-answer hallucination detection - validates that the
abstraction actually generalizes rather than just being designed to look
like it does (see "Validated against a second domain" below).

## The idea

Four independent signals, combined and calibrated against a hand-labeled
gold set:

- **Grounding** - is the claim's text actually present in its source?
- **Cross-model agreement** - does a second, independently-run extraction
  method back up this claim?
- **Self-reported confidence** - what did the originating model itself say?
- **A domain-specific validator** (optional) - anything a particular domain
  can independently re-derive about a claim that the extracting model
  might have gotten wrong (meetings: who's actually responsible for an
  action item, re-derived from the text rather than trusted from the
  model's own claim).

These get combined into a named feature dict and scored by a calibrated
logistic-regression model - trained per-domain, tagged with which domain
it belongs to so it can never be silently applied to the wrong one.

## Quick example

```python
from confidence_referee import Claim, ConfidenceReferee, AgreementSignal, GroundingSignal, SelfReportSignal
from confidence_referee.calibration import Calibrator

class MySource:
    def full_text(self) -> str:
        return "the full text a claim should be checked against"

# Train once, against your own labeled data:
calibrator, report = Calibrator.train(
    feature_dicts=[...],  # list of {"grounding_score": 0.9, "agreement": 0.8, ...}
    labels=[...],          # 1 if the claim was actually correct, 0 otherwise
    domain="my-domain",
)
calibrator.save("calibrator.joblib")

# Score at runtime:
calibrator = Calibrator.load("calibrator.joblib", expected_domain="my-domain")
referee = ConfidenceReferee(
    signals=[GroundingSignal(), AgreementSignal(), SelfReportSignal()],
    calibrator=calibrator,
)
result = referee.score(Claim(text="a claim", source=MySource(), model_confidence=0.7))
print(result.score, result.tier)
```

## What's domain-agnostic vs. what you supply

`GroundingSignal`, `AgreementSignal`, `SelfReportSignal`, `Calibrator`, and
the tiering strategies (`fixed_threshold_tier`, `quantile_tiers`) are used
as-is by any domain. What you supply per domain:

- A `SourceDocument` adapter - anything with a `full_text()` method, so
  `GroundingSignal` can check claims against your kind of source.
- (Optional) A `DomainValidator` - only if your domain has something like
  meetings' owner-attribution: a categorical property of a claim that can
  be independently re-derived, not just trusted from the model.
- Your own labeled dataset, to train a `Calibrator` for your domain.

`examples/meetings/adapter.py` is the original sketch of how this would
work, written before the real integration existed - kept for reference,
but superseded by the actual `confidence/referee_adapter.py` in the
meeting-intelligence repo, which is what that project really runs on now.

## Validated against a second domain

`examples/rag_grounding/` is a deliberately lightweight proof-of-concept
against RAG-answer hallucination detection - a domain with no owner-like
category (no `DomainValidator`), no second extraction arm to compare
against (no `AgreementSignal`), and no self-reported model confidence (no
`SelfReportSignal`). It uses `GroundingSignal` alone, completely unmodified,
against a 300-example sample of [HaluEval](https://github.com/RUCAIBox/HaluEval)'s
QA data: for each question, scoring the real answer (should be grounded in
the source knowledge snippet) against the hallucinated answer (should not
be).

Result: `GroundingSignal`'s raw score alone separates the two with 0.97
AUC (mean grounding_score 0.978 for real answers vs. 0.600 for
hallucinated ones), and a `Calibrator` trained on that single feature
reaches 94% held-out accuracy. No changes were needed to `GroundingSignal`,
`Claim`, `ConfidenceReferee`, or `Calibrator` to get this - only a new
`SourceDocument` adapter (`KnowledgeSource`, ~10 lines) and a data-loading
script. This is the actual evidence that the interfaces generalize, not
just a design intention: `python3 examples/rag_grounding/run_poc.py`.

## Status

Validated against two domains: meeting action-item extraction (the
original, full production integration) and RAG-answer hallucination
detection (a lightweight proof-of-concept, not a full second project). The
API is still kept intentionally small - two domains is enough to trust the
seams (SourceDocument, optional DomainValidator, named feature dicts), not
a reason to start speculatively building for a third.

## Install (editable, for development)

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
