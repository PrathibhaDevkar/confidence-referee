"""Sketch of how confidence-aware-meeting-intelligence would plug into
confidence_referee, once Phase 2 actually wires the two repos together.
Not imported by anything yet - this exists so the shape of the integration
is visible and reviewable now, during Phase 1, rather than discovered later.

Two adapters are needed on the meetings side:

1. A SourceDocument wrapper around Transcript, for GroundingSignal.
2. A DomainValidator wrapping the existing owner_attribution.py logic,
   unchanged - only the calling convention (Claim instead of separate
   evidence_span/transcript args) changes.
"""
from __future__ import annotations

from dataclasses import dataclass

# In the real integration these three imports come from the
# confidence-aware-meeting-intelligence repo (as a dependency, or via a thin
# vendored copy) - written here as the interface they'd need to satisfy.
#
# from pipeline.schema import Transcript
# from confidence.owner_attribution import classify_owner_mention
# from confidence.grounding_check import find_evidence_speaker

from confidence_referee import Claim, DomainValidator


@dataclass
class TranscriptSource:
    """Adapts meeting-intelligence's Transcript to confidence_referee's
    SourceDocument protocol (see signals/grounding.py) - the one seam
    flagged during that project's Phase 0 review."""

    transcript: "Transcript"  # noqa: F821 - see import note above

    def full_text(self) -> str:
        return " ".join(u.text for u in self.transcript.utterances)


class MeetingOwnerValidator:
    """Wraps meeting-intelligence's existing, unchanged owner-attribution
    logic behind the DomainValidator interface. This is the ONLY
    meeting-specific piece in the whole integration - everything else
    (GroundingSignal, AgreementSignal, SelfReportSignal, tiering,
    Calibrator) is reused exactly as confidence_referee ships it."""

    categories = [
        "explicit_self", "explicit_named", "explicit_third_person_named",
        "inferred_pronoun", "inferred_unassigned",
    ]

    def category(self, claim: Claim) -> str:
        transcript = claim.domain_context  # the Transcript, passed through here
        # speaker = find_evidence_speaker(claim.text, transcript)
        # return classify_owner_mention(claim.text, speaker, transcript.participants)
        raise NotImplementedError("Wired up in Phase 2, against the real imports above.")


def build_claim(evidence_span: str, transcript: "Transcript", comparison_texts: list[str], model_confidence: float) -> Claim:
    """What meeting-intelligence's composite_score.score_action_item would
    call instead of building its own feature vector by hand."""
    return Claim(
        text=evidence_span,
        source=TranscriptSource(transcript),
        comparison_texts=comparison_texts,
        model_confidence=model_confidence,
        domain_context=transcript,
    )
