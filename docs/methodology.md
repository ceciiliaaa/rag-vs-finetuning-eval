# Methodology

This document defines every quantity the package computes. Notation: `m` experts, `n` criteria,
`k` systems; `r_ij` is the rank expert `i` gives criterion `j` (forced ranking, `n` = most
important); `s_cj` is the normalised score of system `c` on criterion `j`; `w_j` is the weight of
criterion `j`.

## 1. Criterion weights from expert rankings

Each expert distributes the ranks `1..n` over the `n` criteria exactly once. The forced ranking
avoids the tendency to rate everything as important that interval scales invite (Ishizaka and
Nemery, 2013). The weight of a criterion is its mean rank normalised to sum to one:

```
w_j = mean_i(r_ij) / sum_j mean_i(r_ij)
```

The reported standard deviation is the sample standard deviation across experts (`ddof = 1`).
Rank-based weighting of this kind is discussed in Chergui and Jiménez-Martín (2024).

## 2. Agreement between experts: Kendall's W

Kendall's coefficient of concordance (Kendall and Babington Smith, 1939) measures how similar the
`m` rankings are:

```
R_j = sum_i r_ij                      column sums
S   = sum_j (R_j - mean(R))^2
W   = 12 S / (m^2 (n^3 - n))          W in [0, 1]
```

`W = 1` means identical rankings, `W = 0` no agreement. No tie correction is needed because forced
rankings contain no ties (the loader rejects them).

Two significance tests are reported:

* Chi-square approximation: `chi2 = m (n - 1) W` with `n - 1` degrees of freedom. The
  approximation is coarse for small `n` (Legendre, 2005), which is the usual situation in
  stakeholder studies.
* Monte-Carlo permutation test: every expert's ranking is permuted independently, `W` is
  recomputed, and the p-value is `(b + 1) / (N + 1)` where `b` counts null samples with `W` at
  least as large as observed and `N` is the number of samples (default 20 000, seeded).

## 3. Normalised criterion scores

All scores lie in `[0, 1]` with 1 = best.

* Binary judgements (correctness, transparency): share of test questions judged positively.
* Response time: ratio-to-best on the per-system mean latency, `s_c = t_min / t_c`, so the
  fastest system scores 1 and a system twice as slow scores 0.5.
* Literature-based criteria (recency, cost efficiency): each sub-criterion is scored 0 or 1 per
  system from published evidence; the score is the share of sub-criteria in favour of the
  system. Sub-criteria and sources are listed in `data/case_study/literature_criteria.yaml`.

### Rounding modes

`exact` (default) keeps full precision. `thesis` reproduces the convention of the original
write-up of the bundled case study, where intermediate values were rounded with spreadsheet
(round-half-up) rounding: latency means to one decimal, ratio scores to two, literature
sub-criteria means to two, binary proportions to three, weights to four and weighted contributions
to four decimals before summation. Both modes are computed and committed under `results/`.

## 4. Weighted utility

Additive utility (Nutzwertanalyse; Kühnapfel, 2021; Belton and Stewart, 2002):

```
N_c = sum_j w_j s_cj
```

The gap between the best system `a` and the runner-up `b` is decomposed per criterion:

```
gap_j       = w_j (s_aj - s_bj)
gap_share_j = gap_j / sum_j gap_j
```

Shares sum to one; a negative share marks a criterion on which the runner-up is ahead.

## 5. Sensitivity analysis

### Leave one criterion out

Criterion `j` is removed and the remaining weights are rescaled to sum to one. The table reports
the utilities, the difference between the baseline winner and runner-up, and the winner.

### Random weight perturbation

Weight vectors are sampled from a Dirichlet distribution centred on the expert weights,
`w ~ Dirichlet(alpha * w_expert)`, whose mean is the expert vector; the concentration `alpha`
controls the spread (`alpha = 50` gives component standard deviations of roughly 0.05 to 0.06 for
weights between 0.1 and 0.3). The report gives the share of samples won by each system and the
distribution of the utility advantage. This analysis always uses exact arithmetic.

### Rank-reversal thresholds

For criterion `j`, set its weight to `t` and rescale the other weights proportionally to
`1 - t`. With `rest_c = sum_{l != j} w_l s_cl / (1 - w_j)`, the utility of system `c` becomes

```
N_c(t) = t s_cj + (1 - t) rest_c
```

which is linear in `t`. The difference between the baseline winner `a` and runner-up `b` is zero at

```
t* = (rest_a - rest_b) / ((rest_a - rest_b) - (s_aj - s_bj))
```

If `t*` lies in `[0, 1]` the ranking reverses when the weight of criterion `j` crosses `t*`;
otherwise no change of that single weight can reverse it.

## References

* Belton, V. and Stewart, T. (2002). *Multiple Criteria Decision Analysis: An Integrated Approach.*
  Springer.
* Chergui, Z. and Jiménez-Martín, A. (2024). On ordinal information-based weighting methods and
  comparison analyses. *Information* 15(9), 527. doi:10.3390/info15090527
* Ishizaka, A. and Nemery, P. (2013). *Multi-Criteria Decision Analysis: Methods and Software.*
  Wiley.
* Kendall, M. G. and Babington Smith, B. (1939). The problem of m rankings. *Annals of
  Mathematical Statistics* 10(3), 275-287. doi:10.1214/aoms/1177732186
* Kühnapfel, J. B. (2021). *Scoring und Nutzwertanalysen: Ein Leitfaden für die Praxis.* Springer
  Gabler.
* Legendre, P. (2005). Species associations: the Kendall coefficient of concordance revisited.
  *Journal of Agricultural, Biological, and Environmental Statistics* 10(2), 226-245.
  doi:10.1198/108571105X46642
