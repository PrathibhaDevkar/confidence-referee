"""The originating model's own reported confidence, already normalized to
[0, 1] by the caller before it reaches a Claim. Exists as a Signal class so
self-reported confidence is combined the same way as every other signal,
not threaded through as a bare float with its own special case.
"""
from __future__ import annotations

from confidence_referee.core import Claim


class SelfReportSignal:
    name = "model_confidence"

    def compute(self, claim: Claim) -> float:
        return claim.model_confidence
