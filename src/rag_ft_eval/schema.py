"""Typed containers shared by the evaluation modules.

All result objects are frozen dataclasses so that a computed result cannot be mutated
accidentally between the computation and the report. ``to_dict`` produces plain JSON-serialisable
structures; ``as_frame`` helpers return pandas views for tabular inspection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


class Rounding(StrEnum):
    """Rounding convention applied to intermediate values.

    ``EXACT`` keeps full floating-point precision and is the default.

    ``THESIS`` reproduces the convention used in the original write-up of the bundled case
    study, where intermediate values were rounded before aggregation: latency means to one
    decimal, ratio scores to two, literature sub-criteria means to two, binary proportions to
    three, weights to four, and weighted contributions to four decimals before summation.
    """

    EXACT = "exact"
    THESIS = "thesis"


class MetricKind(StrEnum):
    """How a criterion score is obtained."""

    MEASURED_BINARY = "measured_binary"
    MEASURED_LATENCY = "measured_latency"
    LITERATURE = "literature"


@dataclass(frozen=True)
class MetricSpec:
    """Declaration of one evaluation criterion."""

    id: str
    label: str
    kind: MetricKind
    column: str | None = None
    description: str = ""


@dataclass(frozen=True)
class KendallResult:
    """Kendall's coefficient of concordance with two significance tests."""

    w: float
    chi2: float
    df: int
    p_chi2: float
    p_perm: float | None
    n_raters: int
    n_items: int
    n_perm: int


@dataclass(frozen=True)
class WeightsResult:
    """Criterion weights derived from expert importance rankings."""

    metrics: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    weight: tuple[float, ...]
    kendall: KendallResult

    def weight_vector(self) -> np.ndarray:
        return np.asarray(self.weight, dtype=float)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"mean": self.mean, "std": self.std, "weight": self.weight},
            index=pd.Index(self.metrics, name="metric"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Scores:
    """Normalised scores in [0, 1]; rows are systems, columns are metrics."""

    systems: tuple[str, ...]
    metrics: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    rounding: Rounding
    details: dict[str, Any] = field(default_factory=dict)

    def matrix(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.matrix(),
            index=pd.Index(self.systems, name="system"),
            columns=list(self.metrics),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rounding"] = self.rounding.value
        return data


@dataclass(frozen=True)
class UtilityResult:
    """Weighted utility per system with the decomposition of the gap between the top two."""

    systems: tuple[str, ...]
    metrics: tuple[str, ...]
    weights: tuple[float, ...]
    scores: tuple[tuple[float, ...], ...]
    contributions: tuple[tuple[float, ...], ...]
    totals: tuple[float, ...]
    rounding: Rounding
    winner: str
    runner_up: str
    gap: float
    gap_by_metric: tuple[float, ...]
    gap_share: tuple[float, ...]

    def total_of(self, system: str) -> float:
        return self.totals[self.systems.index(system)]

    def as_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            np.asarray(self.contributions, dtype=float),
            index=pd.Index(self.systems, name="system"),
            columns=list(self.metrics),
        )
        frame["total"] = self.totals
        return frame

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rounding"] = self.rounding.value
        return data


@dataclass(frozen=True)
class LeaveOneOutRow:
    dropped_metric: str
    totals: tuple[float, ...]
    delta: float
    winner: str


@dataclass(frozen=True)
class PerturbationResult:
    """Outcome of sampling weight vectors around the expert weights."""

    n_samples: int
    concentration: float
    seed: int
    win_fraction: dict[str, float]
    delta_mean: float
    delta_p05: float
    delta_p95: float
    delta_min: float
    delta_max: float


@dataclass(frozen=True)
class RankReversalRow:
    metric: str
    current_weight: float
    threshold_weight: float | None
    direction: str | None


@dataclass(frozen=True)
class SensitivityResult:
    baseline_winner: str
    baseline_runner_up: str
    leave_one_out: tuple[LeaveOneOutRow, ...]
    perturbation: PerturbationResult
    rank_reversal: tuple[RankReversalRow, ...]

    def leave_one_out_frame(self, systems: tuple[str, ...]) -> pd.DataFrame:
        rows = []
        for row in self.leave_one_out:
            record: dict[str, Any] = {"dropped_metric": row.dropped_metric}
            record.update(dict(zip(systems, row.totals, strict=True)))
            record["delta"] = row.delta
            record["winner"] = row.winner
            rows.append(record)
        return pd.DataFrame(rows).set_index("dropped_metric")

    def rank_reversal_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(row) for row in self.rank_reversal]).set_index("metric")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
