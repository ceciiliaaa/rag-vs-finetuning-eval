import numpy as np
import pytest

from rag_ft_eval.metrics import (
    binary_mean,
    latency_stats,
    ratio_to_best,
    round_half_up,
    subcriteria_mean,
)
from rag_ft_eval.schema import Rounding

from .conftest import LATENCY, METRICS, SYSTEMS


def test_round_half_up_differs_from_builtin_round():
    assert round(0.725, 2) == 0.72  # binary float, round-half-to-even
    assert round_half_up(0.725, 2) == 0.73
    assert round_half_up(2 / 3, 2) == 0.67
    assert round_half_up(1 / 3, 2) == 0.33
    assert round_half_up(4.6 / 15, 4) == 0.3067


def test_binary_mean_exact_and_thesis():
    hits = [1] * 13 + [0] * 2
    assert binary_mean(hits) == pytest.approx(13 / 15)
    assert binary_mean(hits, Rounding.THESIS) == 0.867
    with pytest.raises(ValueError):
        binary_mean([0, 2])


def test_latency_stats_match_case_study():
    rag = latency_stats(LATENCY["rag"])
    ft = latency_stats(LATENCY["finetuned"])
    assert rag["mean"] == pytest.approx(3.96)
    assert rag["std"] == pytest.approx(1.272, abs=1e-3)
    assert ft["mean"] == pytest.approx(2.94)
    assert ft["std"] == pytest.approx(0.947, abs=1e-3)
    assert rag["n"] == ft["n"] == 15


def test_ratio_to_best_exact_and_thesis():
    means = {"rag": np.mean(LATENCY["rag"]), "finetuned": np.mean(LATENCY["finetuned"])}
    exact = ratio_to_best(means)
    assert exact["finetuned"] == 1.0
    assert exact["rag"] == pytest.approx(2.94 / 3.96)
    thesis = ratio_to_best(means, rounding=Rounding.THESIS)
    assert thesis == {"rag": 0.73, "finetuned": 1.0}


def test_ratio_to_best_higher_is_better():
    assert ratio_to_best({"a": 2.0, "b": 4.0}, lower_is_better=False) == {"a": 0.5, "b": 1.0}
    with pytest.raises(ValueError):
        ratio_to_best({"a": 0.0, "b": 1.0})


def test_subcriteria_mean():
    assert subcriteria_mean([1, 1, 1, 0]) == 0.75
    assert subcriteria_mean([1, 0, 1]) == pytest.approx(2 / 3)
    assert subcriteria_mean([1, 0, 1], Rounding.THESIS) == 0.67
    assert subcriteria_mean([0, 1, 0], Rounding.THESIS) == 0.33


def test_score_matrix_exact(scores_exact):
    assert scores_exact.systems == SYSTEMS
    assert scores_exact.metrics == METRICS
    frame = scores_exact.as_frame()
    np.testing.assert_allclose(
        frame.loc["rag"], [13 / 15, 1.0, 0.75, 2.94 / 3.96, 2 / 3], rtol=1e-6
    )
    np.testing.assert_allclose(frame.loc["finetuned"], [11 / 15, 0.0, 0.25, 1.0, 1 / 3], rtol=1e-6)
    assert scores_exact.details["correctness"]["positive"] == {"rag": 13, "finetuned": 11}
    assert scores_exact.details["latency"]["stats"]["rag"]["n"] == 15
    assert len(scores_exact.details["recency"]["criteria"]) == 4


def test_score_matrix_thesis(scores_thesis):
    frame = scores_thesis.as_frame()
    np.testing.assert_allclose(frame.loc["rag"], [0.867, 1.0, 0.75, 0.73, 0.67])
    np.testing.assert_allclose(frame.loc["finetuned"], [0.733, 0.0, 0.25, 1.0, 0.33])
