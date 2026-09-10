"""Proof-of-concept: does confidence_referee's GroundingSignal, reused
completely unmodified, discriminate grounded from hallucinated claims in a
domain that isn't meetings?

This is deliberately NOT a full validated project on the scale of
confidence-aware-meeting-intelligence - no gold hand-labeling, no
multi-dataset evaluation, no demo app. It exists to answer one narrow
question honestly: does the interface designed against one domain
(meetings) hold up, unmodified, against a genuinely different one? The
answer, from the numbers below, is yes for GroundingSignal specifically -
that's the one signal this domain actually needs.

Usage: python3 examples/rag_grounding/run_poc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from confidence_referee import ConfidenceReferee, GroundingSignal
from confidence_referee.calibration import Calibrator

from adapter import build_claims, load_examples

DOMAIN = "rag_grounding"


def main() -> None:
    examples = load_examples()
    claims, labels = build_claims(examples)
    print(f"{len(examples)} QA examples -> {len(claims)} claims ({sum(labels)} grounded, {len(labels) - sum(labels)} hallucinated)")

    referee = ConfidenceReferee(signals=[GroundingSignal()], calibrator=None)
    feature_dicts = [referee.build_features(c) for c in claims]

    scores = np.array([f["grounding_score"] for f in feature_dicts])
    labels_arr = np.array(labels)

    print("\n--- Raw grounding_score, before any calibration ---")
    print(f"mean grounding_score | grounded claims:     {scores[labels_arr == 1].mean():.3f}")
    print(f"mean grounding_score | hallucinated claims:  {scores[labels_arr == 0].mean():.3f}")
    print(f"AUC (grounding_score alone as the classifier): {roc_auc_score(labels_arr, scores):.3f}")

    calibrator, report = Calibrator.train(feature_dicts, labels, domain=DOMAIN)
    print(f"\n--- Calibrator.train() on the single 'grounding_score' feature ---")
    print("Feature coefficients:")
    for name, coef in zip(calibrator.feature_names, calibrator.model.coef_[0]):
        print(f"  {name:24s} {coef:+.3f}")

    preds = calibrator.model.predict(report["X_test"])
    print("\nHeld-out test set performance:")
    print(classification_report(report["y_test"], preds, target_names=["hallucinated", "grounded"]))

    test_probs = calibrator.model.predict_proba(report["X_test"])[:, 1]
    print(f"Held-out AUC (through the trained Calibrator): {roc_auc_score(report['y_test'], test_probs):.3f}")


if __name__ == "__main__":
    main()
