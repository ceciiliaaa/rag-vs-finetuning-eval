"""End-to-end evaluation: inputs -> weights -> scores -> utility -> sensitivity -> files."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .io import EvaluationInputs
from .metrics import build_score_matrix
from .report import render_report
from .schema import Rounding, Scores, SensitivityResult, UtilityResult, WeightsResult
from .sensitivity import sensitivity as run_sensitivity
from .utility import utility
from .weights import compute_weights


@dataclass(frozen=True)
class EvaluationRun:
    inputs: EvaluationInputs
    rounding: Rounding
    weights: WeightsResult
    scores: Scores
    utility: UtilityResult
    sensitivity: SensitivityResult


def run_evaluation(inputs: EvaluationInputs, rounding: Rounding | None = None) -> EvaluationRun:
    """Compute every result for the given inputs.

    ``rounding`` overrides the mode declared in the configuration. Weight perturbation always
    uses exact arithmetic; the other steps honour the mode.
    """
    mode = rounding or inputs.rounding
    weights = compute_weights(
        inputs.rankings,
        inputs.metric_ids,
        n_perm=int(inputs.kendall["n_permutations"]),
        seed=inputs.kendall["seed"],
    )
    scores = build_score_matrix(
        inputs.measurements, inputs.criteria, inputs.metrics, inputs.system_ids, mode
    )
    result = utility(scores, weights)
    sens = run_sensitivity(
        scores,
        weights,
        n_samples=int(inputs.sensitivity["n_samples"]),
        concentration=float(inputs.sensitivity["concentration"]),
        seed=int(inputs.sensitivity["seed"]),
    )
    return EvaluationRun(
        inputs=inputs,
        rounding=mode,
        weights=weights,
        scores=scores,
        utility=result,
        sensitivity=sens,
    )


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars, tuples and NaN into plain JSON values with stable precision."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 10)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_outputs(run: EvaluationRun, output_dir: Path, plots: bool = True) -> list[Path]:
    """Write JSON results, the Markdown report and (optionally) figures; return written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    meta = {
        "tool": "rag-ft-eval",
        "version": __version__,
        "config": run.inputs.config_path.name,
        "rounding": run.rounding.value,
        "systems": [s.id for s in run.inputs.systems],
        "metrics": list(run.inputs.metric_ids),
    }
    files: dict[str, Any] = {
        "weights.json": {**meta, **run.weights.to_dict(), "experts": list(run.inputs.expert_ids)},
        "scores.json": {**meta, **run.scores.to_dict()},
        "utility.json": {**meta, **run.utility.to_dict()},
        "sensitivity.json": {**meta, **run.sensitivity.to_dict()},
    }
    for name, payload in files.items():
        path = output_dir / name
        _write_json(path, payload)
        written.append(path)

    report_path = output_dir / "report.md"
    report_path.write_text(render_report(run), encoding="utf-8")
    written.append(report_path)

    if plots:
        from .plots import write_figures

        written.extend(write_figures(run, output_dir / "figures"))
    return written
