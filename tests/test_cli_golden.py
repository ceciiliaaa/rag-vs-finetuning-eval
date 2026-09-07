"""Golden test: the committed results/ directory must equal a fresh run of the CLI.

The README quotes numbers from results/report.md. This test (and the same check in CI) makes
sure those files are the output of the code in the repository and not hand-edited.
"""

import json
import math
from pathlib import Path

import pytest

from rag_ft_eval.cli import main

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "data" / "case_study" / "config.yaml"
JSON_FILES = ("weights.json", "scores.json", "utility.json", "sensitivity.json")


def _assert_close(expected, actual, where: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), where
        assert expected.keys() == actual.keys(), where
        for key in expected:
            _assert_close(expected[key], actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), where
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            _assert_close(e, a, f"{where}[{i}]")
    elif isinstance(expected, float) and not isinstance(expected, bool):
        assert isinstance(actual, int | float), where
        assert math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-12), (
            f"{where}: {expected} != {actual}"
        )
    else:
        assert expected == actual, f"{where}: {expected!r} != {actual!r}"


@pytest.mark.parametrize(
    ("rounding", "committed"),
    [("exact", REPO / "results"), ("thesis", REPO / "results" / "thesis_rounding")],
)
def test_committed_results_match_fresh_run(tmp_path: Path, rounding: str, committed: Path):
    assert committed.exists(), "run `rag-ft-eval run data/case_study/config.yaml` first"
    exit_code = main(
        ["run", str(CONFIG), "--rounding", rounding, "--output", str(tmp_path), "--no-plots"]
    )
    assert exit_code == 0
    for name in JSON_FILES:
        expected = json.loads((committed / name).read_text(encoding="utf-8"))
        actual = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        _assert_close(expected, actual, name)
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == (
        committed / "report.md"
    ).read_text(encoding="utf-8")


def test_validate_and_weights_commands(capsys):
    assert main(["validate", str(CONFIG)]) == 0
    assert "5 experts" in capsys.readouterr().out
    assert main(["weights", str(CONFIG)]) == 0
    out = capsys.readouterr().out
    assert "Kendall's W = 0.792" in out


def test_sensitivity_command(capsys):
    assert main(["sensitivity", str(CONFIG), "--rounding", "thesis"]) == 0
    out = capsys.readouterr().out
    assert "Rank-reversal thresholds" in out


def test_invalid_config_returns_error_code(tmp_path: Path, capsys):
    bad = tmp_path / "config.yaml"
    bad.write_text("systems: [a]\n", encoding="utf-8")
    assert main(["validate", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err
