"""The documented public API must stay importable and behave as the README shows."""

from pathlib import Path

import pytest

import rag_ft_eval
from rag_ft_eval import load_config, run_evaluation

CONFIG = Path(__file__).resolve().parents[1] / "data" / "case_study" / "config.yaml"


def test_everything_in_all_is_importable():
    missing = [name for name in rag_ft_eval.__all__ if not hasattr(rag_ft_eval, name)]
    assert not missing, f"declared in __all__ but not exported: {missing}"


def test_readme_example_runs():
    run = run_evaluation(load_config(CONFIG))
    assert run.utility.winner == "rag"
    assert run.utility.total_of("rag") == pytest.approx(0.8461, abs=5e-4)
    assert run.weights.kendall.w == pytest.approx(0.792, abs=5e-4)
    dropped = {row.dropped_metric for row in run.sensitivity.leave_one_out}
    assert dropped == set(run.scores.metrics)


def test_version_is_exposed():
    assert rag_ft_eval.__version__ == rag_ft_eval._version.__version__
