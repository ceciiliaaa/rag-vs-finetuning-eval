"""Shared fixtures: the raw inputs of the bundled case study, typed directly for unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rag_ft_eval.metrics import build_score_matrix
from rag_ft_eval.schema import MetricKind, MetricSpec, Rounding, Scores
from rag_ft_eval.weights import compute_weights

METRICS = ("correctness", "transparency", "recency", "latency", "cost_efficiency")
SYSTEMS = ("rag", "finetuned")

# Five experts, forced ranking 1..5 with 5 = most important.
RANKINGS = np.array(
    [
        [4, 5, 3, 2, 1],
        [5, 4, 2, 1, 3],
        [5, 4, 3, 2, 1],
        [4, 5, 3, 2, 1],
        [5, 3, 4, 1, 2],
    ]
)

LATENCY = {
    "rag": [3.9, 7.2, 3.5, 3.9, 3.3, 3.2, 2.2, 3.4, 3.9, 5.4, 4.7, 2.2, 5.1, 3.3, 4.2],
    "finetuned": [3.8, 2.2, 2.1, 2.1, 3.2, 3.8, 1.2, 2.1, 4.6, 3.7, 4.0, 2.1, 3.2, 2.8, 3.2],
}
CORRECT = {
    "rag": [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    "finetuned": [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
}
TRANSPARENT = {"rag": [1] * 15, "finetuned": [0] * 15}

CRITERIA = {
    "recency": [
        {
            "id": "R1",
            "statement": "update without retraining",
            "scores": {"rag": 1, "finetuned": 0},
        },
        {"id": "R2", "statement": "available during updates", "scores": {"rag": 1, "finetuned": 0}},
        {"id": "R3", "statement": "low cost per update", "scores": {"rag": 1, "finetuned": 0}},
        {"id": "R4", "statement": "stable after updates", "scores": {"rag": 0, "finetuned": 1}},
    ],
    "cost_efficiency": [
        {"id": "C1", "statement": "low initial cost", "scores": {"rag": 1, "finetuned": 0}},
        {"id": "C2", "statement": "low inference cost", "scores": {"rag": 0, "finetuned": 1}},
        {"id": "C3", "statement": "low scaling cost", "scores": {"rag": 1, "finetuned": 0}},
    ],
}

SPECS = (
    MetricSpec("correctness", "Answer correctness", MetricKind.MEASURED_BINARY, "correct"),
    MetricSpec("transparency", "Transparency", MetricKind.MEASURED_BINARY, "transparent"),
    MetricSpec("recency", "Recency", MetricKind.LITERATURE),
    MetricSpec("latency", "Response time", MetricKind.MEASURED_LATENCY, "latency_s"),
    MetricSpec("cost_efficiency", "Cost efficiency", MetricKind.LITERATURE),
)


def measurements_frame() -> pd.DataFrame:
    rows = []
    for system in SYSTEMS:
        for q in range(15):
            rows.append(
                {
                    "question_id": f"q{q + 1:02d}",
                    "system": system,
                    "latency_s": LATENCY[system][q],
                    "correct": CORRECT[system][q],
                    "transparent": TRANSPARENT[system][q],
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def rankings() -> np.ndarray:
    return RANKINGS.copy()


@pytest.fixture(scope="session")
def weights():
    return compute_weights(RANKINGS, METRICS, n_perm=2_000, seed=0)


@pytest.fixture(scope="session")
def measurements() -> pd.DataFrame:
    return measurements_frame()


def scores_for(rounding: Rounding) -> Scores:
    return build_score_matrix(measurements_frame(), CRITERIA, SPECS, SYSTEMS, rounding)


@pytest.fixture(scope="session")
def scores_exact() -> Scores:
    return scores_for(Rounding.EXACT)


@pytest.fixture(scope="session")
def scores_thesis() -> Scores:
    return scores_for(Rounding.THESIS)
