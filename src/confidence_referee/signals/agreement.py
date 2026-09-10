"""Do two independently-produced extractions agree? Generic across any
domain that runs more than one extraction method against the same input -
ported from confidence-aware-meeting-intelligence's
confidence/cross_model_agreement.py, unchanged in logic.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from confidence_referee.core import Claim


def best_match_score(text: str, candidate_texts: list[str]) -> float:
    """Best fuzzy match ratio, in [0, 1], between `text` and any string in
    `candidate_texts`. Direction-agnostic - works for "does this second
    extraction support that first one" in either direction."""
    best = 0.0
    for other in candidate_texts:
        score = fuzz.partial_ratio(text, other) / 100.0
        best = max(best, score)
    return best


class AgreementSignal:
    name = "agreement"

    def compute(self, claim: Claim) -> float:
        return best_match_score(claim.text, claim.comparison_texts) if claim.comparison_texts else 0.0
