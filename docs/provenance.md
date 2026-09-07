# Provenance

The case study in this repository comes from a project thesis submitted at DHBW Stuttgart in 2025.
The analysis was originally carried out in a spreadsheet. This repository is a later
reimplementation as tested software, and it extends the robustness analysis. Nothing about the
study itself was changed in the process: the criteria, the prototypes, the elicited weights, every
measurement and every human judgement are the ones from the study, committed unaltered under
[`../data/case_study/`](../data/case_study).

| | Submitted thesis | This repository |
|---|---|---|
| Criteria, prototypes, measurements | yes | unchanged, committed as raw inputs |
| Expert weights, Kendall's W | yes, chi-square approximation | plus a seeded Monte-Carlo permutation test |
| Additive utility, gap decomposition | yes | unchanged, recomputed from the inputs |
| Arithmetic | rounded intermediate values | exact by default, thesis rounding reproducible on request |
| Leave-one-criterion-out | for one criterion | for all five criteria |
| Weight perturbation | no | Dirichlet sampling around the elicited weights |
| Rank-reversal thresholds | no | closed form, per criterion |
| Implementation | spreadsheet | Python package, CLI, deterministic reports, tests, CI |

## Rounding

The thesis rounded intermediate values before aggregation, in the way a spreadsheet does, and
reported utilities of 0.8453 and 0.4167. Computing the same model in exact arithmetic gives 0.8461
and 0.4171. Both are reproducible:

```bash
rag-ft-eval run data/case_study/config.yaml                     # exact, the default
rag-ft-eval run data/case_study/config.yaml --rounding thesis   # the thesis convention
```

The `thesis` mode rounds latency means to one decimal and the resulting ratio to two, literature
sub-criteria means to two, binary proportions to three, weights to four, and weighted contributions
to four decimals before summation, using round-half-up rather than Python's round-half-to-even.
Ranking, gap decomposition and every sensitivity conclusion are the same in both modes.

## Figures

The diagrams in [`figures/`](figures) are English redraws. The original German thesis figures are
kept in [`thesis_figures/`](thesis_figures) with an index that maps each one to its counterpart
here or to the committed data it was made from.
