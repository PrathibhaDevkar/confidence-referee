"""Calibrator: trains and applies the logistic-regression scoring model over
a NAMED feature dict (see core.py's module docstring for why named, not
positional). Also the fix for a real risk flagged while reviewing
confidence-aware-meeting-intelligence: that project's saved model file
carried no record of which domain or feature layout it was trained for, so
loading the wrong one against the wrong feature vector could fail silently
or silently misalign columns. Calibrator.load() refuses to load a mismatched
domain instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from confidence_referee.core import ConfidenceResult
from confidence_referee.tiering import fixed_threshold_tier


@dataclass
class Calibrator:
    model: Any
    feature_names: list[str]
    domain: str
    schema_version: int = 1

    @classmethod
    def train(
        cls,
        feature_dicts: list[dict[str, float]],
        labels: list[int],
        domain: str,
        schema_version: int = 1,
        test_size: float = 0.25,
        random_state: int = 42,
        **logistic_regression_kwargs: Any,
    ) -> tuple["Calibrator", dict]:
        """Fits a LogisticRegression over the union of feature names seen
        across `feature_dicts` (missing keys default to 0.0 per-row, so a
        domain without a DomainValidator - and thus no category: features -
        works the same as one with it). Returns (calibrator, report) where
        report holds the held-out classification metrics so callers can
        print/log them without this function dictating how.
        """
        feature_names = sorted({name for row in feature_dicts for name in row})
        X = np.array([[row.get(name, 0.0) for name in feature_names] for row in feature_dicts])
        y = np.array(labels)

        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        kwargs = {"max_iter": 1000, "class_weight": "balanced", **logistic_regression_kwargs}
        model = LogisticRegression(**kwargs)
        model.fit(X_train, y_train)

        calibrator = cls(model=model, feature_names=feature_names, domain=domain, schema_version=schema_version)
        report = {
            "n_examples": len(y),
            "n_positive": int(y.sum()),
            "train_idx": train_idx,
            "test_idx": test_idx,
            "X_test": X_test,
            "y_test": y_test,
        }
        return calibrator, report

    def score(self, features: dict[str, float]) -> ConfidenceResult:
        vector = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        prob = float(self.model.predict_proba(vector)[0, 1])
        return ConfidenceResult(score=prob, tier=fixed_threshold_tier(prob))

    def score_many(self, feature_dicts: list[dict[str, float]]) -> np.ndarray:
        """Batch scoring - returns raw probabilities (not ConfidenceResults)
        for evaluation code that needs the array directly, e.g. to compute
        quantile tiers over a whole held-out set at once."""
        X = np.array([[row.get(name, 0.0) for name in self.feature_names] for row in feature_dicts])
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "domain": self.domain,
                "schema_version": self.schema_version,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, expected_domain: Optional[str] = None) -> "Calibrator":
        bundle = joblib.load(path)
        calibrator = cls(
            model=bundle["model"],
            feature_names=bundle["feature_names"],
            domain=bundle.get("domain", "unknown"),
            schema_version=bundle.get("schema_version", 0),
        )
        if expected_domain is not None and calibrator.domain != expected_domain:
            raise ValueError(
                f"Calibrator at {path} was trained for domain {calibrator.domain!r}, "
                f"not {expected_domain!r} - refusing to score with a mismatched calibrator."
            )
        return calibrator
