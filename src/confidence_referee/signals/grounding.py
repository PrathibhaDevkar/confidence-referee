"""Is the claim's text actually present in its source? Generic across any
domain where "source" means "something with searchable text" - ported from
confidence-aware-meeting-intelligence's confidence/grounding_check.py, but
generalized away from that project's Transcript type.

This is the one adapter seam flagged during that project's Phase 0: the
original is_grounded() did `" ".join(u.text for u in transcript.utterances)`
directly, so grounding logic and the meeting-specific Transcript type were
one file. Here, GroundingSignal only requires a `SourceDocument` - anything
with a `full_text()` method. A domain adapts its own source type by
implementing that one method (see examples/meetings/adapter.py for how
meeting-intelligence's Transcript would do it) rather than this file
needing to know what a Transcript, a RAG chunk set, or a filing is.
"""
from __future__ import annotations

from typing import Protocol

from rapidfuzz import fuzz

from confidence_referee.core import Claim

DEFAULT_FUZZY_THRESHOLD = 0.85


class SourceDocument(Protocol):
    """Minimal shape a domain's source must provide for grounding checks."""

    def full_text(self) -> str: ...


def is_grounded(
    claim_text: str, source: SourceDocument, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD
) -> tuple[bool, float]:
    """Returns (is_grounded, match_score in [0, 1])."""
    text = (claim_text or "").strip()
    if not text:
        return False, 0.0

    full_text = source.full_text()
    if text in full_text:
        return True, 1.0

    score = fuzz.partial_ratio(text, full_text) / 100.0
    return score >= fuzzy_threshold, score


class GroundingSignal:
    name = "grounding_score"

    def __init__(self, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD) -> None:
        self.fuzzy_threshold = fuzzy_threshold

    def compute(self, claim: Claim) -> float:
        _, score = is_grounded(claim.text, claim.source, self.fuzzy_threshold)
        return score

    def is_grounded(self, claim: Claim) -> bool:
        grounded, _ = is_grounded(claim.text, claim.source, self.fuzzy_threshold)
        return grounded
