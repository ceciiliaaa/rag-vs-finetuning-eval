"""Weighted additive utility (Nutzwertanalyse) and gap decomposition.

The utility of system i is ``N_i = sum_j w_j * s_ij`` with weights ``w`` summing to one and
scores ``s`` in [0, 1]. The gap between the best and the second-best system is decomposed into
per-criterion contributions ``w_j * (s_best,j - s_second,j)`` whose shares sum to one; negative
shares mark criteria on which the runner-up is ahead.
"""

from __future__ import annotations

import numpy as np

from .metrics import round_half_up
from .schema import Rounding, Scores, UtilityResult, WeightsResult


def _round_array(values: np.ndarray, ndigits: int) -> np.ndarray:
    return np.vectorize(lambda x: round_half_up(x, ndigits), otypes=[float])(values)


def utility(scores: Scores, weights: WeightsResult) -> UtilityResult:
    """Aggregate scores with weights into one utility value per system."""
    if scores.metrics != weights.metrics:
        raise ValueError(
            f"metric order differs between scores {scores.metrics} and weights {weights.metrics}"
        )
    if len(scores.systems) < 2:
        raise ValueError("need at least two systems to compare")

    w = weights.weight_vector()
    s = scores.matrix()
    if scores.rounding is Rounding.THESIS:
        w = _round_array(w, 4)
        contributions = _round_array(w * s, 4)
    else:
        contributions = w * s
    totals = contributions.sum(axis=1)

    order = np.argsort(-totals, kind="stable")
    best, second = int(order[0]), int(order[1])
    gap_by_metric = contributions[best] - contributions[second]
    gap = float(gap_by_metric.sum())
    gap_share = gap_by_metric / gap if gap > 0 else np.full_like(gap_by_metric, np.nan)

    return UtilityResult(
        systems=scores.systems,
        metrics=scores.metrics,
        weights=tuple(float(x) for x in w),
        scores=tuple(tuple(float(x) for x in row) for row in s),
        contributions=tuple(tuple(float(x) for x in row) for row in contributions),
        totals=tuple(float(x) for x in totals),
        rounding=scores.rounding,
        winner=scores.systems[best],
        runner_up=scores.systems[second],
        gap=gap,
        gap_by_metric=tuple(float(x) for x in gap_by_metric),
        gap_share=tuple(float(x) for x in gap_share),
    )
