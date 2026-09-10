from confidence_referee import Claim, ConfidenceReferee
from confidence_referee.calibration import Calibrator


class ConstantSignal:
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self._value = value

    def compute(self, claim: Claim) -> float:
        return self._value


class FakeOwnerValidator:
    categories = ["self", "other"]

    def category(self, claim: Claim) -> str:
        return "self" if "I" in claim.text else "other"


def test_build_features_without_domain_validator():
    referee = ConfidenceReferee(signals=[ConstantSignal("a", 0.5), ConstantSignal("b", 0.9)], calibrator=None)
    features = referee.build_features(Claim(text="hello"))
    assert features == {"a": 0.5, "b": 0.9}


def test_build_features_with_domain_validator_adds_category_columns():
    referee = ConfidenceReferee(
        signals=[ConstantSignal("a", 0.5)], calibrator=None, domain_validator=FakeOwnerValidator()
    )
    features = referee.build_features(Claim(text="I will do it"))
    assert features == {"a": 0.5, "category:self": 1.0, "category:other": 0.0}


def test_referee_score_delegates_to_calibrator():
    rows = [
        {"a": 0.9, "b": 0.9}, {"a": 0.8, "b": 0.85}, {"a": 0.1, "b": 0.2}, {"a": 0.05, "b": 0.15},
        {"a": 0.85, "b": 0.9}, {"a": 0.15, "b": 0.1},
    ]
    labels = [1, 1, 0, 0, 1, 0]
    calibrator, _report = Calibrator.train(rows, labels, domain="test", test_size=0.34)

    referee = ConfidenceReferee(signals=[ConstantSignal("a", 0.9), ConstantSignal("b", 0.9)], calibrator=calibrator)
    result = referee.score(Claim(text="anything"))
    assert 0.0 <= result.score <= 1.0
    assert result.tier in {"High", "Medium", "Low"}
