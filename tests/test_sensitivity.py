import pytest

from rag_ft_eval.sensitivity import (
    leave_one_out,
    perturb_weights,
    rank_reversal_threshold,
    sensitivity,
)


def test_leave_one_out_exact(scores_exact, weights):
    rows = {row.dropped_metric: row for row in leave_one_out(scores_exact, weights)}
    assert set(rows) == set(scores_exact.metrics)
    assert rows["transparency"].delta == pytest.approx(0.207, abs=2e-3)
    assert all(row.winner == "rag" for row in rows.values())
    assert all(row.delta > 0 for row in rows.values())


def test_leave_one_out_thesis(scores_thesis, weights):
    rows = {row.dropped_metric: row for row in leave_one_out(scores_thesis, weights)}
    assert rows["transparency"].delta == pytest.approx(0.206, abs=2e-3)


def test_perturbation_is_seeded_and_robust(scores_exact, weights):
    first = perturb_weights(scores_exact, weights, n_samples=2_000, seed=3)
    second = perturb_weights(scores_exact, weights, n_samples=2_000, seed=3)
    assert first == second
    assert first.win_fraction["rag"] == 1.0
    assert first.delta_min > 0
    assert first.delta_p05 < first.delta_mean < first.delta_p95
    assert sum(first.win_fraction.values()) == pytest.approx(1.0)


def test_perturbation_rejects_bad_parameters(scores_exact, weights):
    with pytest.raises(ValueError):
        perturb_weights(scores_exact, weights, n_samples=0)
    with pytest.raises(ValueError):
        perturb_weights(scores_exact, weights, concentration=0)


def test_rank_reversal_thresholds(scores_exact, weights):
    transparency = rank_reversal_threshold(scores_exact, weights, "transparency")
    assert transparency.threshold_weight is None  # RAG leads even without transparency
    latency = rank_reversal_threshold(scores_exact, weights, "latency")
    assert latency.threshold_weight is not None
    assert latency.direction == "increase"
    assert latency.threshold_weight == pytest.approx(0.665, abs=5e-3)
    with pytest.raises(ValueError):
        rank_reversal_threshold(scores_exact, weights, "nonexistent")


def test_sensitivity_bundle(scores_exact, weights):
    result = sensitivity(scores_exact, weights, n_samples=500, seed=0)
    assert result.baseline_winner == "rag"
    assert len(result.leave_one_out) == 5
    assert len(result.rank_reversal) == 5
    frame = result.leave_one_out_frame(scores_exact.systems)
    assert list(frame.columns) == ["rag", "finetuned", "delta", "winner"]
    assert "threshold_weight" in result.rank_reversal_frame().columns
    assert isinstance(result.to_dict()["perturbation"]["win_fraction"], dict)
