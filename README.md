# confidence-referee

A reusable confidence-calibration layer for LLM extraction pipelines: given
a claim, its source, and (optionally) a second independent extraction to
compare it against, score how much the claim should actually be trusted -
instead of presenting every extracted fact with the same, false certainty.

Generalized out of [confidence-aware-meeting-intelligence](https://github.com/PrathibhaDevkar/confidence-aware-meeting-intelligence),
where this pattern was first built and validated: filtering to the
top-confidence tier nearly doubled action-item extraction precision (58%
vs. a 35% no-filtering baseline) on genuinely held-out data. That project
remains the one validated integration; this package is the domain-agnostic
core extracted out of it.

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

See `examples/meetings/adapter.py` for the sketch of how
confidence-aware-meeting-intelligence's `Transcript` and owner-attribution
logic plug in - not yet wired into a live integration (that's the next
phase of generalizing this out), but the shape is deliberately visible now.

## Status

Extracted from one validated domain (meeting action-item extraction).
No second domain has been built against this package yet - the API here is
kept intentionally small rather than over-abstracted for domains that don't
exist yet.

## Install (editable, for development)

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
