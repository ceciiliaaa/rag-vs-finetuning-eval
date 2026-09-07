import numpy as np
import pandas as pd
import pytest

from rag_ft_eval.weights import compute_weights, kendalls_w, rankings_matrix, validate_rankings

from .conftest import METRICS, RANKINGS


def test_means_std_and_weights(weights):
    assert weights.metrics == METRICS
    np.testing.assert_allclose(weights.mean, [4.6, 4.2, 3.0, 1.6, 1.6])
    np.testing.assert_allclose(weights.std, [0.548, 0.837, 0.707, 0.548, 0.894], atol=1e-3)
    np.testing.assert_allclose(weights.weight, [0.3067, 0.28, 0.20, 0.1067, 0.1067], atol=5e-5)
    assert weights.weight_vector().sum() == pytest.approx(1.0)


def test_kendalls_w_matches_closed_form(rankings):
    result = kendalls_w(rankings, n_perm=0)
    # S = 198 for column sums (23, 21, 15, 8, 8): W = 12 * 198 / (25 * 120)
    assert result.w == pytest.approx(0.792)
    assert result.chi2 == pytest.approx(15.84)
    assert result.df == 4
    assert result.p_chi2 == pytest.approx(0.00324, abs=2e-4)
    assert result.p_perm is None


def test_permutation_p_value_is_small_and_reproducible(rankings):
    first = kendalls_w(rankings, n_perm=5_000, seed=1)
    second = kendalls_w(rankings, n_perm=5_000, seed=1)
    assert first.p_perm == second.p_perm
    assert 0 < first.p_perm < 0.01


def test_perfect_agreement_gives_one():
    identical = np.tile([1, 2, 3, 4, 5], (4, 1))
    assert kendalls_w(identical, n_perm=0).w == pytest.approx(1.0)


def test_reversed_rankings_give_zero():
    opposite = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    assert kendalls_w(opposite, n_perm=0).w == pytest.approx(0.0)


@pytest.mark.parametrize(
    "bad",
    [
        [[1, 1, 2, 3, 4], [1, 2, 3, 4, 5]],  # tie
        [[1, 2, 3, 4, 6], [1, 2, 3, 4, 5]],  # out of range
        [[1, 2, 3, 4, 5]],  # single rater
        [1, 2, 3],  # one-dimensional
    ],
)
def test_invalid_rankings_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_rankings(bad)


def test_metric_count_must_match(rankings):
    with pytest.raises(ValueError):
        compute_weights(rankings, METRICS[:-1], n_perm=0)


def test_rankings_matrix_from_long_table():
    frame = pd.DataFrame(
        [
            {"expert_id": f"expert_{i + 1}", "metric": metric, "rank": int(RANKINGS[i, j])}
            for i in range(RANKINGS.shape[0])
            for j, metric in enumerate(METRICS)
        ]
    )
    np.testing.assert_array_equal(rankings_matrix(frame, METRICS), RANKINGS)
    with pytest.raises(ValueError):
        rankings_matrix(frame.iloc[:-1], METRICS)
