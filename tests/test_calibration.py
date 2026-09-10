import tempfile
from pathlib import Path

import pytest

from confidence_referee.calibration import Calibrator


def _toy_dataset():
    rows = [
        {"grounded": 1.0, "agreement": 0.9}, {"grounded": 1.0, "agreement": 0.8},
        {"grounded": 0.0, "agreement": 0.1}, {"grounded": 0.0, "agreement": 0.2},
        {"grounded": 1.0, "agreement": 0.7}, {"grounded": 0.0, "agreement": 0.05},
    ]
    labels = [1, 1, 0, 0, 1, 0]
    return rows, labels


def test_train_produces_scoreable_calibrator():
    rows, labels = _toy_dataset()
    calibrator, report = Calibrator.train(rows, labels, domain="toy", test_size=0.34)
    assert calibrator.domain == "toy"
    assert set(calibrator.feature_names) == {"grounded", "agreement"}
    assert report["n_examples"] == 6

    result = calibrator.score({"grounded": 1.0, "agreement": 0.9})
    assert 0.0 <= result.score <= 1.0


def test_missing_feature_defaults_to_zero_not_a_crash():
    rows, labels = _toy_dataset()
    calibrator, _ = Calibrator.train(rows, labels, domain="toy", test_size=0.34)
    result = calibrator.score({"grounded": 1.0})  # "agreement" omitted
    assert 0.0 <= result.score <= 1.0


def test_save_and_load_round_trip():
    rows, labels = _toy_dataset()
    calibrator, _ = Calibrator.train(rows, labels, domain="toy", test_size=0.34)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        calibrator.save(path)
        reloaded = Calibrator.load(path, expected_domain="toy")
        assert reloaded.domain == "toy"
        assert reloaded.feature_names == calibrator.feature_names


def test_load_refuses_mismatched_domain():
    rows, labels = _toy_dataset()
    calibrator, _ = Calibrator.train(rows, labels, domain="toy", test_size=0.34)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        calibrator.save(path)
        with pytest.raises(ValueError, match="not 'other-domain'"):
            Calibrator.load(path, expected_domain="other-domain")
