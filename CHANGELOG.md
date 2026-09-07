# Changelog

## 0.2.2 (2026-09-07)

- The package architecture diagram is removed from the README; the module table below it carried
  the same information in less space.

## 0.2.1 (2026-09-07)

Presentation of the result figures. No change to any measurement or numerical result.

- One muted palette across every figure: a deep-blue to pale-rose ramp for the criteria and two
  colours from the same family for the systems. The SVG diagrams follow the same palette, so the
  fine-tuning pipeline is no longer drawn in a saturated red.
- The rank-reversal panel is dropped from the sensitivity figure; it was mostly empty space and the
  thresholds are stated in the report and the README. The figure is now a single leave-one-out
  panel.
- The target-profile radar returns to the README, where it shows the same result as a profile
  against what the stakeholders asked for.

## 0.2.0 (2026-09-07)

Presentation, information architecture and public API. No change to any measurement, judgement or
numerical result.

- Public API exposed from the package root (`from rag_ft_eval import load_config, run_evaluation`),
  with `__all__` and a test that pins the exported surface.
- README restructured around the released software: an architecture figure of the evaluation
  package, the module responsibilities, the tech stack and a programmatic usage example come before
  the study design.
- Sections `Key contributions`, `Repository extension beyond the thesis` and `Scope and limitations`
  removed from the README. The provenance content moved to `docs/provenance.md`; the methodological
  constraints stay in `docs/limitations.md`, linked from the robustness discussion.
- Badges restored (CI, Python, license).

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
- 46 offline tests including a golden test; CI regenerates `results/` and fails on any drift.
