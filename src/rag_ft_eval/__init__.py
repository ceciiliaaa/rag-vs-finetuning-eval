"""Stakeholder-weighted multi-criteria evaluation of competing systems.

The package turns expert importance rankings into criterion weights, normalises measured and
literature-scored criteria onto a common scale, aggregates them into a weighted utility per
system, and reports how far the resulting ranking survives changes to the weights.

Typical use::

    from rag_ft_eval import load_config, run_evaluation

    run = run_evaluation(load_config("data/case_study/config.yaml"))
    print(run.utility.winner, run.utility.total_of("rag"))
"""

from ._version import __version__
from .evaluate import EvaluationRun, run_evaluation, write_outputs
from .io import ConfigError, EvaluationInputs, load_config
from .metrics import binary_mean, build_score_matrix, ratio_to_best, subcriteria_mean
from .report import render_report
from .schema import (
    KendallResult,
    MetricKind,
    MetricSpec,
    Rounding,
    Scores,
    SensitivityResult,
    UtilityResult,
    WeightsResult,
)
from .sensitivity import (
    leave_one_out,
    perturb_weights,
    rank_reversal_threshold,
    sensitivity,
)
from .utility import utility
from .weights import compute_weights, kendalls_w

__all__ = [
    "ConfigError",
    "EvaluationInputs",
    "EvaluationRun",
    "KendallResult",
    "MetricKind",
    "MetricSpec",
    "Rounding",
    "Scores",
    "SensitivityResult",
    "UtilityResult",
    "WeightsResult",
    "__version__",
    "binary_mean",
    "build_score_matrix",
    "compute_weights",
    "kendalls_w",
    "leave_one_out",
    "load_config",
    "perturb_weights",
    "rank_reversal_threshold",
    "ratio_to_best",
    "render_report",
    "run_evaluation",
    "sensitivity",
    "subcriteria_mean",
    "utility",
    "write_outputs",
]
