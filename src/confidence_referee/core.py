"""The domain-agnostic core: Claim, Signal, DomainValidator, ConfidenceResult,
and the ConfidenceReferee orchestrator that ties them together.

Design note on feature representation: signals here are combined into a
NAMED dict (feature name -> float), not a fixed-position list. The
originating meeting-intelligence project's build_feature_vector() used a
positional list, which meant the owner-attribution one-hot columns were
baked into the function's argument order - any change to which categories
exist, or whether a domain has a DomainValidator at all, meant touching that
function's signature. Named features let confidence_referee.calibration
turn a dict into an ordered vector via name lookup at train/score time
instead, so signals and domain-validators can be added, removed, or
omitted per-domain without changing any shared function's shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class Claim:
    """One extracted claim/statement a domain wants a trust score for -
    e.g. a meeting action item, a RAG-generated answer sentence, a line in
    an AI-generated financial summary.

    text: the claim itself, or the quote/evidence span backing it - whatever
        the domain's GroundingSignal should check against its source.
    source: whatever the domain's GroundingSignal needs to verify grounding
        (see confidence_referee.signals.grounding.SourceDocument) - e.g. a
        meeting transcript, a set of retrieved RAG chunks, a filing.
    comparison_texts: outputs from another, independent extraction method
        for the same input, if one exists - used by AgreementSignal. Empty
        list if there's no second arm to compare against.
    model_confidence: the originating model's own self-reported confidence,
        already normalized to [0, 1] by the caller - used by SelfReportSignal.
    domain_context: anything extra a DomainValidator needs that doesn't fit
        the fields above (e.g. meetings' DomainValidator needs the full
        transcript to find who's speaking, not just this claim's text).
    """

    text: str
    source: Any = None
    comparison_texts: list[str] = field(default_factory=list)
    model_confidence: float = 0.0
    domain_context: Optional[Any] = None


@dataclass
class ConfidenceResult:
    score: float
    tier: str
    rationale: Optional[str] = None


class Signal(Protocol):
    """A domain-agnostic piece of evidence about whether a Claim should be
    trusted. `name` must be stable - it's the feature-dict key used
    everywhere a Signal's output is trained on or scored against."""

    name: str

    def compute(self, claim: Claim) -> float: ...


class DomainValidator(Protocol):
    """The one slot in this system that's genuinely domain-specific. A
    domain with no natural categorical validator (e.g. RAG grounding) can
    simply not supply one - ConfidenceReferee treats it as optional."""

    categories: list[str]

    def category(self, claim: Claim) -> str: ...


class ConfidenceReferee:
    """Combines registered Signals (and an optional DomainValidator) into a
    named feature dict for one Claim, then asks a Calibrator to turn that
    into a ConfidenceResult. Holds no domain knowledge itself - everything
    domain-specific comes in through what's registered."""

    def __init__(
        self,
        signals: list[Signal],
        calibrator: "confidence_referee.calibration.Calibrator",  # noqa: F821 - see calibration.py
        domain_validator: Optional[DomainValidator] = None,
    ) -> None:
        self.signals = signals
        self.domain_validator = domain_validator
        self.calibrator = calibrator

    def build_features(self, claim: Claim) -> dict[str, float]:
        features = {signal.name: signal.compute(claim) for signal in self.signals}
        if self.domain_validator is not None:
            category = self.domain_validator.category(claim)
            for c in self.domain_validator.categories:
                features[f"category:{c}"] = 1.0 if category == c else 0.0
        return features

    def score(self, claim: Claim) -> ConfidenceResult:
        return self.calibrator.score(self.build_features(claim))

    def score_batch(self, claims: list[Claim]) -> list[ConfidenceResult]:
        return [self.score(claim) for claim in claims]
