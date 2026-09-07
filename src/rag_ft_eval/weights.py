"""Criterion weights from expert importance rankings and Kendall's W.

Experts rank the criteria with a forced ranking: each expert assigns every integer from 1 to n
exactly once, where n is the most important criterion. Weights are the mean rank of each
criterion divided by the sum of all mean ranks, so they sum to one.

Agreement between experts is quantified with Kendall's coefficient of concordance W
(Kendall and Babington Smith, 1939). Because n is small in typical stakeholder studies, the usual
chi-square approximation is complemented by a Monte-Carlo permutation test that permutes every
expert's ranking independently under the null hypothesis of no agreement.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .schema import KendallResult, WeightsResult


def validate_rankings(rankings: object) -> np.ndarray:
    """Return ``rankings`` as a float array of shape (raters, items) after validation.

    Every row must be a permutation of 1..n. Ties are rejected because the concordance formula
    used here has no tie correction and forced rankings do not produce ties.
    """
    arr = np.asarray(rankings, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"rankings must be two-dimensional, got shape {arr.shape}")
    m, n = arr.shape
    if m < 2 or n < 2:
        raise ValueError("need at least two raters and two items")
    expected = np.arange(1, n + 1, dtype=float)
    for i, row in enumerate(arr):
        if not np.array_equal(np.sort(row), expected):
            raise ValueError(f"rater {i} did not assign a permutation of 1..{n}: {row.tolist()}")
    return arr


def rankings_matrix(frame: pd.DataFrame, metrics: Sequence[str]) -> np.ndarray:
    """Pivot a long table with columns ``expert_id, metric, rank`` into a (raters, items) array.

    Columns are ordered as in ``metrics``; rows follow the order of first appearance of each
    expert. Missing or duplicated cells raise ``ValueError``.
    """
    required = {"expert_id", "metric", "rank"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"rankings table lacks columns: {sorted(missing)}")
    unknown = set(frame["metric"]).difference(metrics)
    if unknown:
        raise ValueError(f"rankings table contains unknown metrics: {sorted(unknown)}")
    experts = list(dict.fromkeys(frame["expert_id"]))
    pivot = frame.pivot(index="expert_id", columns="metric", values="rank")
    if pivot.isna().any().any() or set(pivot.columns) != set(metrics):
        raise ValueError("every expert must rank every metric exactly once")
    return pivot.loc[experts, list(metrics)].to_numpy(dtype=float)


def _concordance(arr: np.ndarray) -> float:
    m, n = arr.shape
    column_sums = arr.sum(axis=0)
    s = float(((column_sums - column_sums.mean()) ** 2).sum())
    return 12.0 * s / (m**2 * (n**3 - n))


def kendalls_w(rankings: object, n_perm: int = 20_000, seed: int | None = 0) -> KendallResult:
    """Kendall's W with a chi-square approximation and a permutation test.

    The chi-square statistic is m(n-1)W with n-1 degrees of freedom. The permutation p-value is
    (b + 1) / (n_perm + 1), where b counts null samples with W at least as large as observed.
    Pass ``n_perm=0`` to skip the permutation test.
    """
    arr = validate_rankings(rankings)
    m, n = arr.shape
    w = _concordance(arr)
    chi2 = m * (n - 1) * w
    df = n - 1
    p_chi2 = float(stats.chi2.sf(chi2, df))

    p_perm: float | None = None
    if n_perm > 0:
        rng = np.random.default_rng(seed)
        keys = rng.random((n_perm, m, n))
        null_rankings = np.argsort(keys, axis=2) + 1
        column_sums = null_rankings.sum(axis=1)
        s = ((column_sums - column_sums.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        w_null = 12.0 * s / (m**2 * (n**3 - n))
        exceed = int(np.count_nonzero(w_null >= w - 1e-12))
        p_perm = (exceed + 1) / (n_perm + 1)

    return KendallResult(
        w=float(w),
        chi2=float(chi2),
        df=int(df),
        p_chi2=p_chi2,
        p_perm=p_perm,
        n_raters=int(m),
        n_items=int(n),
        n_perm=int(n_perm),
    )


def compute_weights(
    rankings: object,
    metrics: Sequence[str],
    n_perm: int = 20_000,
    seed: int | None = 0,
) -> WeightsResult:
    """Derive normalised criterion weights from a (raters, items) ranking matrix.

    ``metrics`` names the columns. The standard deviation uses one degree of freedom (sample
    standard deviation across experts).
    """
    arr = validate_rankings(rankings)
    if len(metrics) != arr.shape[1]:
        raise ValueError(f"{len(metrics)} metric names for {arr.shape[1]} ranked items")
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1)
    weight = mean / mean.sum()
    return WeightsResult(
        metrics=tuple(metrics),
        mean=tuple(float(x) for x in mean),
        std=tuple(float(x) for x in std),
        weight=tuple(float(x) for x in weight),
        kendall=kendalls_w(arr, n_perm=n_perm, seed=seed),
    )
