from confidence_referee import AgreementSignal, Claim, GroundingSignal, SelfReportSignal
from confidence_referee.signals.grounding import SourceDocument


class FakeSource:
    def __init__(self, text: str) -> None:
        self._text = text

    def full_text(self) -> str:
        return self._text


def test_grounding_signal_exact_match():
    claim = Claim(text="I will send the report", source=FakeSource("okay I will send the report tomorrow"))
    signal = GroundingSignal()
    assert signal.is_grounded(claim) is True
    assert signal.compute(claim) == 1.0


def test_grounding_signal_ungrounded():
    claim = Claim(text="completely unrelated made-up sentence", source=FakeSource("we discussed the budget"))
    signal = GroundingSignal()
    assert signal.is_grounded(claim) is False


def test_grounding_signal_empty_claim():
    claim = Claim(text="", source=FakeSource("some transcript text"))
    signal = GroundingSignal()
    assert signal.is_grounded(claim) is False
    assert signal.compute(claim) == 0.0


def test_agreement_signal_matches_best_candidate():
    claim = Claim(text="send the quarterly report", comparison_texts=["unrelated", "I'll send the quarterly report"])
    signal = AgreementSignal()
    assert signal.compute(claim) > 0.7


def test_agreement_signal_no_candidates():
    claim = Claim(text="send the report", comparison_texts=[])
    signal = AgreementSignal()
    assert signal.compute(claim) == 0.0


def test_self_report_signal_passthrough():
    claim = Claim(text="anything", model_confidence=0.42)
    signal = SelfReportSignal()
    assert signal.compute(claim) == 0.42
