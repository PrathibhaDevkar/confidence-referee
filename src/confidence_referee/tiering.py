"""Named, swappable tiering strategies for turning a continuous calibrated
score into a discrete tier label - ported unchanged from
confidence-aware-meeting-intelligence's confidence/tiering.py, which was
already fully domain-agnostic.

- fixed_threshold_tier: absolute cutoffs, meant for a user-facing "High/
  Medium/Low" badge where "High" should mean something stable across runs.
- quantile_tiers: top/middle/bottom third BY RANK within a given batch of
  scores - guarantees balanced groups for evaluation, regardless of where
  the raw score distribution happens to sit.
"""
from __future__ import annotations

import numpy as np

DEFAULT_HIGH_THRESHOLD = 0.66
DEFAULT_MEDIUM_THRESHOLD = 0.33


def fixed_threshold_tier(
    score: float, high: float = DEFAULT_HIGH_THRESHOLD, medium: float = DEFAULT_MEDIUM_THRESHOLD
) -> str:
    if score >= high:
        return "High"
    if score >= medium:
        return "Medium"
    return "Low"


def quantile_tiers(scores) -> list[str]:
    """Top/middle/bottom third by rank within `scores` itself - not
    comparable across different batches of scores, unlike fixed_threshold_tier."""
    scores = np.asarray(scores)
    lower, upper = np.percentile(scores, [33.33, 66.67])
    labels = []
    for s in scores:
        if s >= upper:
            labels.append("Top third")
        elif s >= lower:
            labels.append("Middle third")
        else:
            labels.append("Bottom third")
    return labels
