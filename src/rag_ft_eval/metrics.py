"""Normalisation of measured and literature-based criteria to scores in [0, 1].

Three kinds of criteria are supported:

* ``measured_binary``: each test question is judged 0 or 1 per system; the score is the mean.
* ``measured_latency``: response times per question; the score is the ratio-to-best
  ``t_min / t_system`` computed on the per-system means, so the fastest system scores 1.
* ``literature``: binary sub-criteria judged from published evidence; the score is the mean.

Every function accepts a :class:`Rounding` mode. ``EXACT`` never rounds; ``THESIS`` reproduces
the rounding of intermediate values used in the original case-study write-up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import numpy as np
import pandas as pd

from .schema import MetricKind, MetricSpec, Rounding, Scores


def round_half_up(value: float, ndigits: int) -> float:
    """Round like a spreadsheet: halves go away from zero (0.725 -> 0.73).

    Python's built-in ``round`` uses round-half-to-even on binary floats, which turns 0.725 into
    0.72. The original case-study write-up used spreadsheet rounding, so the thesis mode must
    reproduce that convention exactly.
    """
    quantum = Decimal(1).scaleb(-ndigits)
    return float(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def _round(value: float, ndigits: int, rounding: Rounding) -> float:
    return round_half_up(value, ndigits) if rounding is Rounding.THESIS else float(value)


def _validate_binary(values: np.ndarray, what: str) -> None:
    if values.size == 0:
        raise ValueError(f"{what}: no values")
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{what}: values must be 0 or 1, got {np.unique(values).tolist()}")


def binary_mean(hits: Sequence[int], rounding: Rounding = Rounding.EXACT) -> float:
    """Proportion of positive judgements (rounded to three decimals in thesis mode)."""
    arr = np.asarray(hits)
    _validate_binary(arr, "binary judgements")
    return _round(arr.mean(), 3, rounding)


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    """Sample statistics of response times in seconds."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        raise ValueError("need at least two latency measurements")
    if (arr <= 0).any():
        raise ValueError("latencies must be positive")
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def ratio_to_best(
    means: Mapping[str, float],
    lower_is_better: bool = True,
    rounding: Rounding = Rounding.EXACT,
) -> dict[str, float]:
    """Ratio-to-best normalisation of one summary value per system.

    With ``lower_is_better`` the score is ``best / value``; otherwise ``value / best``. In thesis
    mode the summary values are rounded to one decimal before the ratio and the ratio to two.
    """
    if not means:
        raise ValueError("no systems to normalise")
    prepared = {name: _round(value, 1, rounding) for name, value in means.items()}
    if any(value <= 0 for value in prepared.values()):
        raise ValueError("ratio-to-best needs strictly positive values")
    best = min(prepared.values()) if lower_is_better else max(prepared.values())
    return {
        name: _round(best / value if lower_is_better else value / best, 2, rounding)
        for name, value in prepared.items()
    }


def subcriteria_mean(scores: Sequence[int], rounding: Rounding = Rounding.EXACT) -> float:
    """Mean of binary sub-criteria (rounded to two decimals in thesis mode)."""
    arr = np.asarray(scores)
    _validate_binary(arr, "sub-criteria")
    return _round(arr.mean(), 2, rounding)


def build_score_matrix(
    measurements: pd.DataFrame,
    criteria: Mapping[str, Sequence[Mapping[str, Any]]],
    metric_specs: Sequence[MetricSpec],
    systems: Sequence[str],
    rounding: Rounding = Rounding.EXACT,
) -> Scores:
    """Compute the systems x metrics score matrix from raw inputs.

    ``measurements`` is a long table with a ``system`` column and one column per measured
    criterion (named in each :class:`MetricSpec`). ``criteria`` maps literature-based metric ids
    to their sub-criteria, each a mapping with ``id``, ``statement``, ``scores`` (per system) and
    ``sources``. The ``details`` of the returned :class:`Scores` record how every score was
    derived so that the report can show the underlying counts and statistics.
    """
    if "system" not in measurements.columns:
        raise ValueError("measurements need a 'system' column")
    unknown_systems = set(measurements["system"]).difference(systems)
    if unknown_systems:
        raise ValueError(f"measurements refer to unknown systems: {sorted(unknown_systems)}")

    per_metric: dict[str, dict[str, float]] = {}
    details: dict[str, Any] = {}
    for spec in metric_specs:
        if spec.kind is MetricKind.MEASURED_BINARY:
            if spec.column is None or spec.column not in measurements.columns:
                raise ValueError(f"metric {spec.id}: measurement column {spec.column!r} missing")
            hits = {
                system: measurements.loc[measurements["system"] == system, spec.column].to_numpy()
                for system in systems
            }
            per_metric[spec.id] = {
                system: binary_mean(values, rounding) for system, values in hits.items()
            }
            details[spec.id] = {
                "kind": spec.kind.value,
                "column": spec.column,
                "positive": {system: int(values.sum()) for system, values in hits.items()},
                "n": {system: int(values.size) for system, values in hits.items()},
            }
        elif spec.kind is MetricKind.MEASURED_LATENCY:
            if spec.column is None or spec.column not in measurements.columns:
                raise ValueError(f"metric {spec.id}: measurement column {spec.column!r} missing")
            stats = {
                system: latency_stats(
                    measurements.loc[measurements["system"] == system, spec.column].to_numpy()
                )
                for system in systems
            }
            per_metric[spec.id] = ratio_to_best(
                {system: stats[system]["mean"] for system in systems},
                lower_is_better=True,
                rounding=rounding,
            )
            details[spec.id] = {"kind": spec.kind.value, "column": spec.column, "stats": stats}
        elif spec.kind is MetricKind.LITERATURE:
            if spec.id not in criteria:
                raise ValueError(f"metric {spec.id}: no literature criteria provided")
            items = list(criteria[spec.id])
            per_metric[spec.id] = {
                system: subcriteria_mean([int(item["scores"][system]) for item in items], rounding)
                for system in systems
            }
            details[spec.id] = {
                "kind": spec.kind.value,
                "criteria": [
                    {
                        "id": item["id"],
                        "statement": item["statement"],
                        "scores": {system: int(item["scores"][system]) for system in systems},
                        "sources": list(item.get("sources", [])),
                    }
                    for item in items
                ],
            }
        else:  # pragma: no cover - guarded by the enum
            raise ValueError(f"unsupported metric kind {spec.kind}")

    values = tuple(
        tuple(per_metric[spec.id][system] for spec in metric_specs) for system in systems
    )
    return Scores(
        systems=tuple(systems),
        metrics=tuple(spec.id for spec in metric_specs),
        values=values,
        rounding=rounding,
        details=details,
    )
