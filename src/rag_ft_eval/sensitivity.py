"""Sensitivity of the utility ranking to the criterion weights.

Three complementary analyses are provided:

* ``leave_one_out``: drop one criterion, renormalise the remaining weights proportionally and
  recompute the utilities. Shows whether a single criterion carries the result.
* ``perturb_weights``: sample weight vectors from a Dirichlet distribution centred on the expert
  weights and report how often each system wins. Shows robustness to moderate disagreement
  about priorities. This analysis always uses exact arithmetic.
* ``rank_reversal_threshold``: for each criterion, the weight at which the top two systems would
  swap if that weight were changed and the others rescaled proportionally. Because the utility is
  linear in the weight, the threshold has a closed form.
"""

from __future__ import annotations

import numpy as np

from .schema import (
    LeaveOneOutRow,
    PerturbationResult,
    RankReversalRow,
    Scores,
    SensitivityResult,
    UtilityResult,
    WeightsResult,
)
from .utility import utility


def _subset_weights(weights: WeightsResult, keep: list[int]) -> WeightsResult:
    w = weights.weight_vector()[keep]
    w = w / w.sum()
    return WeightsResult(
        metrics=tuple(weights.metrics[k] for k in keep),
        mean=tuple(weights.mean[k] for k in keep),
        std=tuple(weights.std[k] for k in keep),
        weight=tuple(float(x) for x in w),
        kendall=weights.kendall,
    )


def _subset_scores(scores: Scores, keep: list[int]) -> Scores:
    matrix = scores.matrix()[:, keep]
    return Scores(
        systems=scores.systems,
        metrics=tuple(scores.metrics[k] for k in keep),
        values=tuple(tuple(float(x) for x in row) for row in matrix),
        rounding=scores.rounding,
    )


def leave_one_out(
    scores: Scores, weights: WeightsResult, baseline: UtilityResult | None = None
) -> tuple[LeaveOneOutRow, ...]:
    """Recompute utilities with each criterion removed in turn.

    ``delta`` is the utility of the baseline winner minus that of the baseline runner-up under
    the reduced weight set, so a negative value marks a rank reversal.
    """
    base = baseline or utility(scores, weights)
    i_win = scores.systems.index(base.winner)
    i_run = scores.systems.index(base.runner_up)
    rows = []
    for j, metric in enumerate(scores.metrics):
        keep = [k for k in range(len(scores.metrics)) if k != j]
        result = utility(_subset_scores(scores, keep), _subset_weights(weights, keep))
        rows.append(
            LeaveOneOutRow(
                dropped_metric=metric,
                totals=result.totals,
                delta=float(result.totals[i_win] - result.totals[i_run]),
                winner=result.winner,
            )
        )
    return tuple(rows)


def perturb_weights(
    scores: Scores,
    weights: WeightsResult,
    n_samples: int = 10_000,
    concentration: float = 50.0,
    seed: int = 0,
    baseline: UtilityResult | None = None,
) -> PerturbationResult:
    """Sample weight vectors around the expert weights and count the winners.

    Weights are drawn from ``Dirichlet(concentration * w)``, whose mean is the expert weight
    vector ``w``; larger ``concentration`` means smaller perturbations. ``delta`` statistics
    describe the utility advantage of the baseline winner over the baseline runner-up.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if concentration <= 0:
        raise ValueError("concentration must be positive")
    base = baseline or utility(scores, weights)
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(concentration * weights.weight_vector(), size=n_samples)
    totals = samples @ scores.matrix().T
    winners = np.argmax(totals, axis=1)
    win_fraction = {system: float(np.mean(winners == i)) for i, system in enumerate(scores.systems)}
    i_win = scores.systems.index(base.winner)
    i_run = scores.systems.index(base.runner_up)
    delta = totals[:, i_win] - totals[:, i_run]
    return PerturbationResult(
        n_samples=int(n_samples),
        concentration=float(concentration),
        seed=int(seed),
        win_fraction=win_fraction,
        delta_mean=float(delta.mean()),
        delta_p05=float(np.percentile(delta, 5)),
        delta_p95=float(np.percentile(delta, 95)),
        delta_min=float(delta.min()),
        delta_max=float(delta.max()),
    )


def rank_reversal_threshold(
    scores: Scores, weights: WeightsResult, metric: str, baseline: UtilityResult | None = None
) -> RankReversalRow:
    """Weight of ``metric`` at which the baseline winner and runner-up would swap.

    With the weight of the chosen criterion set to ``t`` and the other weights rescaled to sum to
    ``1 - t``, the utility difference between the top two systems is linear in ``t``. The
    threshold is its root if it lies in [0, 1]; otherwise no reversal is possible by changing
    this weight alone and ``threshold_weight`` is ``None``.
    """
    if metric not in scores.metrics:
        raise ValueError(f"unknown metric {metric!r}")
    base = baseline or utility(scores, weights)
    j = scores.metrics.index(metric)
    w = weights.weight_vector()
    s = scores.matrix()
    i_win = scores.systems.index(base.winner)
    i_run = scores.systems.index(base.runner_up)
    others = [k for k in range(len(scores.metrics)) if k != j]

    rest_weight = 1.0 - w[j]
    if rest_weight <= 0:
        rest = np.zeros(len(scores.systems))
    else:
        rest = (s[:, others] @ w[others]) / rest_weight
    own_diff = s[i_win, j] - s[i_run, j]
    rest_diff = rest[i_win] - rest[i_run]
    denominator = rest_diff - own_diff

    threshold: float | None = None
    direction: str | None = None
    if abs(denominator) > 1e-12:
        root = rest_diff / denominator
        if 0.0 <= root <= 1.0:
            threshold = float(root)
            direction = "decrease" if root < w[j] else "increase"
    return RankReversalRow(
        metric=metric,
        current_weight=float(w[j]),
        threshold_weight=threshold,
        direction=direction,
    )


def sensitivity(
    scores: Scores,
    weights: WeightsResult,
    n_samples: int = 10_000,
    concentration: float = 50.0,
    seed: int = 0,
) -> SensitivityResult:
    """Run all three sensitivity analyses."""
    base = utility(scores, weights)
    return SensitivityResult(
        baseline_winner=base.winner,
        baseline_runner_up=base.runner_up,
        leave_one_out=leave_one_out(scores, weights, base),
        perturbation=perturb_weights(scores, weights, n_samples, concentration, seed, base),
        rank_reversal=tuple(
            rank_reversal_threshold(scores, weights, metric, base) for metric in scores.metrics
        ),
    )
