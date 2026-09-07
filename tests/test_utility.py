import numpy as np
import pytest

from rag_ft_eval.schema import Rounding, Scores, WeightsResult
from rag_ft_eval.utility import utility


def test_exact_utilities(scores_exact, weights):
    result = utility(scores_exact, weights)
    assert result.winner == "rag"
    assert result.runner_up == "finetuned"
    assert result.total_of("rag") == pytest.approx(0.8461, abs=5e-4)
    assert result.total_of("finetuned") == pytest.approx(0.4171, abs=5e-4)
    assert result.gap == pytest.approx(0.429, abs=1e-3)
    assert result.rounding is Rounding.EXACT


def test_thesis_rounding_reproduces_original_write_up(scores_thesis, weights):
    result = utility(scores_thesis, weights)
    assert result.total_of("rag") == pytest.approx(0.8453, abs=5e-4)
    assert result.total_of("finetuned") == pytest.approx(0.4167, abs=5e-4)
    # contributions are rounded to four decimals before summation in this mode
    assert all(round(c, 4) == c for row in result.contributions for c in row)


def test_gap_decomposition(scores_exact, weights):
    result = utility(scores_exact, weights)
    share = dict(zip(result.metrics, result.gap_share, strict=True))
    assert share["transparency"] == pytest.approx(0.65, abs=0.01)
    assert share["recency"] == pytest.approx(0.23, abs=0.01)
    assert share["correctness"] == pytest.approx(0.10, abs=0.01)
    assert share["latency"] < 0  # the runner-up is faster
    assert sum(result.gap_share) == pytest.approx(1.0)
    frame = result.as_frame()
    np.testing.assert_allclose(frame["total"], result.totals)


def test_metric_order_mismatch_is_rejected(scores_exact, weights):
    shuffled = WeightsResult(
        metrics=tuple(reversed(weights.metrics)),
        mean=weights.mean,
        std=weights.std,
        weight=weights.weight,
        kendall=weights.kendall,
    )
    with pytest.raises(ValueError):
        utility(scores_exact, shuffled)


def test_tie_yields_nan_shares(weights):
    tied = Scores(
        systems=("a", "b"),
        metrics=weights.metrics,
        values=((0.5,) * 5, (0.5,) * 5),
        rounding=Rounding.EXACT,
    )
    result = utility(tied, weights)
    assert result.gap == 0
    assert all(np.isnan(result.gap_share))
