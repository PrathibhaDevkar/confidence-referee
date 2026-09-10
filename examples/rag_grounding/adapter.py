"""Adapts HaluEval's QA-hallucination data to confidence_referee's
interfaces - the second domain proving the abstraction generalizes, not
just in theory. Deliberately the LIGHTEST possible integration: no
DomainValidator (this domain has no analog to meetings' owner-attribution),
no AgreementSignal or SelfReportSignal (there's no second extraction arm
and no self-reported model confidence in this dataset) - just
GroundingSignal, reused completely unmodified from confidence_referee.

Data: a 300-example random sample (seed=42) of HaluEval's qa_data.json
(https://github.com/RUCAIBox/HaluEval), each row a QA pair with a knowledge
snippet, a right_answer, and a hallucinated_answer for the same question.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from confidence_referee import Claim

SAMPLE_PATH = Path(__file__).resolve().parent / "sample_data.jsonl"


@dataclass
class KnowledgeSource:
    """Satisfies confidence_referee's SourceDocument protocol - the same
    protocol meeting-intelligence's TranscriptSource satisfies, for a
    completely different kind of "source" (a short knowledge snippet
    instead of a full meeting transcript)."""

    knowledge: str

    def full_text(self) -> str:
        return self.knowledge


def load_examples() -> list[dict]:
    return [json.loads(line) for line in SAMPLE_PATH.read_text().splitlines() if line.strip()]


def build_claims(examples: list[dict]) -> tuple[list[Claim], list[int]]:
    """Two claims per example: the right_answer (label 1 - should be
    grounded in the knowledge snippet) and the hallucinated_answer (label 0
    - should NOT be well-grounded). comparison_texts is left empty and
    model_confidence left at its default (0.0) - this domain has neither a
    second extraction arm nor a self-reported confidence to supply."""
    claims: list[Claim] = []
    labels: list[int] = []
    for row in examples:
        source = KnowledgeSource(row["knowledge"])
        claims.append(Claim(text=row["right_answer"], source=source))
        labels.append(1)
        claims.append(Claim(text=row["hallucinated_answer"], source=source))
        labels.append(0)
    return claims, labels
