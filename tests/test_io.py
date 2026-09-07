import shutil
from pathlib import Path

import pytest
import yaml

from rag_ft_eval.io import ConfigError, load_config
from rag_ft_eval.schema import MetricKind, Rounding

REPO = Path(__file__).resolve().parents[1]
CASE_STUDY = REPO / "data" / "case_study"


def test_case_study_loads_and_is_consistent():
    inputs = load_config(CASE_STUDY / "config.yaml")
    assert inputs.system_ids == ("rag", "finetuned")
    assert inputs.metric_ids == (
        "correctness",
        "transparency",
        "recency",
        "latency",
        "cost_efficiency",
    )
    assert inputs.rankings.shape == (5, 5)
    assert inputs.expert_ids == tuple(f"expert_{i}" for i in range(1, 6))
    assert len(inputs.questions) == 15
    assert len(inputs.measurements) == 30
    assert inputs.rounding is Rounding.EXACT
    assert set(inputs.criteria) == {"recency", "cost_efficiency"}
    assert inputs.output_dir == REPO / "results"
    assert [m.kind for m in inputs.metrics].count(MetricKind.LITERATURE) == 2


def test_measurements_keep_german_text():
    inputs = load_config(CASE_STUDY / "config.yaml")
    assert inputs.measurements["answer"].str.contains("ä").any()
    assert inputs.questions["question_de"].str.contains("ü").any()


@pytest.fixture
def case_copy(tmp_path: Path) -> Path:
    target = tmp_path / "case_study"
    shutil.copytree(CASE_STUDY, target)
    return target


def _edit_yaml(path: Path, mutate) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_non_permutation_ranking_is_rejected(case_copy: Path):
    rankings = case_copy / "expert_rankings.csv"
    text = rankings.read_text(encoding="utf-8").replace(
        "expert_1,correctness,4", "expert_1,correctness,5"
    )
    rankings.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="permutation"):
        load_config(case_copy / "config.yaml")


def test_unknown_citation_key_is_rejected(case_copy: Path):
    def mutate(data):
        data["metrics"]["recency"]["criteria"][0]["sources"].append("nobody2099")

    _edit_yaml(case_copy / "literature_criteria.yaml", mutate)
    with pytest.raises(ConfigError, match="unknown source"):
        load_config(case_copy / "config.yaml")


def test_non_binary_judgement_is_rejected(case_copy: Path):
    measurements = case_copy / "measurements.csv"
    text = measurements.read_text(encoding="utf-8").replace(
        "q01,rag,3.9,1,1,", "q01,rag,3.9,2,1,", 1
    )
    measurements.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="0 or 1"):
        load_config(case_copy / "config.yaml")


def test_missing_measurement_row_is_rejected(case_copy: Path):
    measurements = case_copy / "measurements.csv"
    lines = measurements.read_text(encoding="utf-8").splitlines(keepends=True)
    measurements.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(ConfigError, match="lacks answers"):
        load_config(case_copy / "config.yaml")


def test_unknown_metric_kind_is_rejected(case_copy: Path):
    def mutate(data):
        data["metrics"][0]["kind"] = "measured_fuzzy"

    _edit_yaml(case_copy / "config.yaml", mutate)
    with pytest.raises(ConfigError, match="unknown kind"):
        load_config(case_copy / "config.yaml")


def test_missing_config_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config(CASE_STUDY / "does_not_exist.yaml")
