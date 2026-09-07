"""Loading and validation of an evaluation configuration and its input files.

A configuration is a YAML file that names the systems, the ordered list of criteria and the
input tables (expert rankings, test questions, measurements, literature criteria). All paths in
the configuration are resolved relative to the configuration file. Every input is validated on
load so that a typo in a CSV fails early with a precise message instead of silently shifting a
score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .schema import MetricKind, MetricSpec, Rounding
from .weights import rankings_matrix, validate_rankings


@dataclass(frozen=True)
class SystemSpec:
    id: str
    label: str


@dataclass(frozen=True)
class EvaluationInputs:
    """Everything needed to run one evaluation, fully validated."""

    config_path: Path
    case_study: dict[str, Any]
    systems: tuple[SystemSpec, ...]
    metrics: tuple[MetricSpec, ...]
    expert_ids: tuple[str, ...]
    rankings: np.ndarray
    questions: pd.DataFrame
    measurements: pd.DataFrame
    criteria: dict[str, list[dict[str, Any]]]
    references: dict[str, dict[str, Any]]
    rounding: Rounding
    kendall: dict[str, Any]
    sensitivity: dict[str, Any]
    output_dir: Path

    @property
    def system_ids(self) -> tuple[str, ...]:
        return tuple(system.id for system in self.systems)

    @property
    def metric_ids(self) -> tuple[str, ...]:
        return tuple(metric.id for metric in self.metrics)

    @property
    def system_labels(self) -> dict[str, str]:
        return {system.id: system.label for system in self.systems}

    @property
    def metric_labels(self) -> dict[str, str]:
        return {metric.id: metric.label for metric in self.metrics}


class ConfigError(ValueError):
    """Raised when a configuration or one of its input files is invalid."""


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _parse_systems(raw: Any) -> tuple[SystemSpec, ...]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise ConfigError("config: 'systems' must list at least two systems")
    systems = []
    for entry in raw:
        if isinstance(entry, str):
            systems.append(SystemSpec(id=entry, label=entry))
        else:
            systems.append(
                SystemSpec(
                    id=str(_require(entry, "id", "systems")),
                    label=str(entry.get("label", entry["id"])),
                )
            )
    ids = [system.id for system in systems]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"config: duplicate system ids in {ids}")
    return tuple(systems)


def _parse_metrics(raw: Any) -> tuple[MetricSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("config: 'metrics' must be a non-empty list")
    specs = []
    for entry in raw:
        metric_id = str(_require(entry, "id", "metrics"))
        try:
            kind = MetricKind(str(_require(entry, "kind", f"metric {metric_id}")))
        except ValueError as exc:
            raise ConfigError(
                f"metric {metric_id}: unknown kind {entry['kind']!r}; "
                f"expected one of {[k.value for k in MetricKind]}"
            ) from exc
        column = entry.get("column")
        if kind is not MetricKind.LITERATURE and not column:
            raise ConfigError(f"metric {metric_id}: measured metrics need a 'column'")
        specs.append(
            MetricSpec(
                id=metric_id,
                label=str(entry.get("label", metric_id)),
                kind=kind,
                column=str(column) if column else None,
                description=str(entry.get("description", "")),
            )
        )
    ids = [spec.id for spec in specs]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"config: duplicate metric ids in {ids}")
    return tuple(specs)


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise ConfigError(f"input file not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8")
    missing = required.difference(frame.columns)
    if missing:
        raise ConfigError(f"{path.name}: missing columns {sorted(missing)}")
    return frame


def _validate_measurements(
    measurements: pd.DataFrame,
    questions: pd.DataFrame,
    systems: tuple[str, ...],
    metrics: tuple[MetricSpec, ...],
) -> None:
    question_ids = set(questions["question_id"])
    if len(question_ids) != len(questions):
        raise ConfigError("test_questions: question_id must be unique")
    unknown_q = set(measurements["question_id"]).difference(question_ids)
    if unknown_q:
        raise ConfigError(f"measurements: unknown question ids {sorted(unknown_q)}")
    unknown_s = set(measurements["system"]).difference(systems)
    if unknown_s:
        raise ConfigError(f"measurements: unknown systems {sorted(unknown_s)}")
    counts = measurements.groupby(["question_id", "system"]).size()
    if (counts != 1).any():
        raise ConfigError("measurements: every (question, system) pair must appear exactly once")
    for system in systems:
        answered = set(measurements.loc[measurements["system"] == system, "question_id"])
        if answered != question_ids:
            raise ConfigError(f"measurements: system {system!r} lacks answers for some questions")
    for spec in metrics:
        if spec.kind is MetricKind.LITERATURE:
            continue
        if spec.column not in measurements.columns:
            raise ConfigError(f"measurements: column {spec.column!r} for metric {spec.id} missing")
        values = measurements[spec.column]
        if values.isna().any():
            raise ConfigError(f"measurements: column {spec.column!r} has missing values")
        if spec.kind is MetricKind.MEASURED_BINARY and not values.isin([0, 1]).all():
            raise ConfigError(f"measurements: column {spec.column!r} must contain only 0 or 1")
        if spec.kind is MetricKind.MEASURED_LATENCY and (values <= 0).any():
            raise ConfigError(f"measurements: column {spec.column!r} must be positive")


def _validate_criteria(
    raw: dict[str, Any], systems: tuple[str, ...], metrics: tuple[MetricSpec, ...]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    references = dict(raw.get("references") or {})
    criteria: dict[str, list[dict[str, Any]]] = {}
    declared = raw.get("metrics") or {}
    for spec in metrics:
        if spec.kind is not MetricKind.LITERATURE:
            continue
        if spec.id not in declared:
            raise ConfigError(f"literature_criteria: no entry for metric {spec.id!r}")
        items = declared[spec.id].get("criteria") or []
        if not items:
            raise ConfigError(f"literature_criteria: metric {spec.id!r} has no criteria")
        for item in items:
            for key in ("id", "statement", "scores"):
                if key not in item:
                    raise ConfigError(f"literature_criteria: {spec.id} criterion lacks {key!r}")
            for system in systems:
                if system not in item["scores"]:
                    raise ConfigError(
                        f"literature_criteria: {spec.id}/{item['id']} has no score for {system!r}"
                    )
                if item["scores"][system] not in (0, 1):
                    raise ConfigError(
                        f"literature_criteria: {spec.id}/{item['id']} score for {system!r} "
                        "must be 0 or 1"
                    )
            for key in item.get("sources", []):
                if key not in references:
                    raise ConfigError(
                        f"literature_criteria: {spec.id}/{item['id']} cites unknown source {key!r}"
                    )
        criteria[spec.id] = [dict(item) for item in items]
    return criteria, references


def load_config(path: str | Path) -> EvaluationInputs:
    """Load and validate a configuration file and all inputs it references."""
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    base = config_path.parent

    systems = _parse_systems(_require(raw, "systems", "config"))
    metrics = _parse_metrics(_require(raw, "metrics", "config"))
    inputs = _require(raw, "inputs", "config")
    system_ids = tuple(system.id for system in systems)
    metric_ids = tuple(metric.id for metric in metrics)

    rankings_frame = _read_csv(
        base / _require(inputs, "expert_rankings", "inputs"), {"expert_id", "metric", "rank"}
    )
    try:
        rankings = validate_rankings(rankings_matrix(rankings_frame, metric_ids))
    except ValueError as exc:
        raise ConfigError(f"expert_rankings: {exc}") from exc
    expert_ids = tuple(dict.fromkeys(rankings_frame["expert_id"]))

    questions = _read_csv(
        base / _require(inputs, "test_questions", "inputs"), {"question_id", "category"}
    )
    measurements = _read_csv(
        base / _require(inputs, "measurements", "inputs"), {"question_id", "system"}
    )
    _validate_measurements(measurements, questions, system_ids, metrics)

    criteria_path = base / _require(inputs, "literature_criteria", "inputs")
    if not criteria_path.exists():
        raise ConfigError(f"input file not found: {criteria_path}")
    with criteria_path.open(encoding="utf-8") as handle:
        criteria_raw = yaml.safe_load(handle) or {}
    criteria, references = _validate_criteria(criteria_raw, system_ids, metrics)

    try:
        rounding = Rounding(str(raw.get("rounding", "exact")))
    except ValueError as exc:
        raise ConfigError(f"config: unknown rounding mode {raw.get('rounding')!r}") from exc

    kendall = {"n_permutations": 20_000, "seed": 0}
    kendall.update(raw.get("kendall") or {})
    sensitivity = {"n_samples": 10_000, "concentration": 50.0, "seed": 0}
    sensitivity.update(raw.get("sensitivity") or {})

    output_dir = (base / str(raw.get("output_dir", "results"))).resolve()

    return EvaluationInputs(
        config_path=config_path,
        case_study=dict(raw.get("case_study") or {}),
        systems=systems,
        metrics=metrics,
        expert_ids=expert_ids,
        rankings=rankings,
        questions=questions,
        measurements=measurements,
        criteria=criteria,
        references=references,
        rounding=rounding,
        kendall=kendall,
        sensitivity=sensitivity,
        output_dir=output_dir,
    )
