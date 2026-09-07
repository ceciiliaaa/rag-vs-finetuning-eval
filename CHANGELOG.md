# Changelog

## 0.1.1 (unreleased)

- README rewritten around the thesis context: research question, method figures, results,
  limitations, repository layout.
- Thesis figures added under `docs/figures/` (German labels; English versions to follow).
- `requirements.txt` / `requirements-dev.txt` exported from `uv.lock` for pip users.

## 0.1.0 (2026-09-07)

- Evaluation framework: expert weights with Kendall's W (chi-square and permutation test),
  criterion normalisation with `exact` and `thesis` rounding modes, weighted utility with gap
  decomposition, sensitivity analysis (leave-one-out, Dirichlet weight perturbation,
  rank-reversal thresholds).
- CLI `rag-ft-eval run | weights | sensitivity | validate`, deterministic JSON and Markdown
  output, figures.
- Case-study inputs (five expert rankings, 15 test questions with both systems' answers,
  literature criteria with sources) and committed results in both rounding modes.
- 43 offline tests including a golden test; CI regenerates `results/` and fails on any drift.
