# Changelog

## 0.2.0 (2026-09-07)

Presentation and information architecture, no change to any measurement, judgement or result.

- README repositioned as a research artifact: research question, study design, results,
  robustness, scope conditions. Tutorial-level explanations of RAG and fine-tuning removed.
- Claims tightened. The result is stated as conditional on the application context, the
  operationalised criteria, the elicited stakeholder profile and the sample. The transparency
  score is described as a property of the evaluated prototype rather than of fine-tuned systems in
  general, and the sensitivity analysis is described as testing robustness to the weight model
  only.
- Contributions of the submitted thesis and of this repository separated in a dedicated section
  and in `docs/methodology.md`.
- `docs/system-design.md` added: configuration of both prototypes and the measurement instruments.
- Diagrams used in the documentation redrawn in English as SVG under `docs/figures/`. The original
  German thesis figures moved to `docs/thesis_figures/` with an index for provenance.
- `docs/limitations.md` restructured by sample, evidence quality and model assumptions.
- Badge row removed; repository description and topics set.

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
