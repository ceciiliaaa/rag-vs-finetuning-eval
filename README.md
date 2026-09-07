# RAG vs. Fine-Tuning Evaluation

**A reproducible stakeholder-weighted evaluation of retrieval-augmented generation and supervised
fine-tuning for domain-specific question answering**

> Project thesis by **Cecilia Nothstein**<br>
> DHBW Stuttgart, Business Information Systems<br>
> Conducted in cooperation with **Mercedes-AMG** @ Portfolio Strategy & Market Intelligence

A department that wants a question-answering assistant over its customer feedback has to pick a
domain-adaptation strategy, and accuracy alone does not decide it. Whether an analyst can see
where an answer came from, how quickly last week's feedback reaches the system, how long a query
takes and what the setup costs over its lifetime all matter, and they trade off against each
other. The literature reviewed for this project emphasised technical performance; the
organisational trade-off was less systematically operationalised.

This repository is the evaluation instrument of a Design Science Research thesis that compared two
widely used domain-adaptation strategies, retrieval-augmented generation and supervised
fine-tuning, on that trade-off:

> **RQ** Which of the two strategies for domain specialisation of large language models,
> retrieval-augmented generation or supervised fine-tuning, achieves the higher economic value in
> the context of analysing social-media customer data?

Economic value is treated as a multi-criteria construct, weighted by the people who would use the
system and aggregated in a utility analysis. Two matched prototypes were built on the same corpus
and evaluated on the same 15 questions.

**The result: RAG reached a utility of ≈ 0.85 against ≈ 0.42 for fine-tuning under the elicited
stakeholder profile.** The more informative part is *why*. Two thirds of the gap come from a
single criterion, transparency, where the difference is architectural rather than a matter of
answer quality: the fine-tuning prototype had no retrieval layer and therefore could show no
per-answer sources. And the framework can say exactly how far stakeholder priorities would have to
move before the ranking flips, namely response time rising from a weight of 0.11 to 0.66. The
verdict is conditional on this application context, these operationalised criteria, this
stakeholder profile and this sample, and the repository is built so that a reader can change any
of them and recompute.

## Key contributions

- **Matched prototypes.** RAG and supervised fine-tuning evaluated on the same corpus, interface,
  generation model and question set, so measured differences are attributable to the
  knowledge-integration step.
- **A stakeholder-weighted MCDA framework** over answer correctness, transparency (per-answer
  source traceability), recency, response time and cost efficiency, with two scoring routes:
  measured on the prototypes, or scored from published evidence with cited sources.
- **Weights elicited from five practitioners** in Portfolio Strategy & Market Intelligence by
  forced ranking, with rater agreement quantified rather than assumed.
- **A reproducible Python implementation**: deterministic reports, a golden test and CI that
  regenerates every committed result and fails on drift, so no number in this README is typed by
  hand.
- **Robustness analysis** by leave-one-criterion-out, Dirichlet weight perturbation and
  closed-form rank-reversal thresholds, which turns "RAG wins" into "RAG wins unless priorities
  shift this far".
- **A context-specific answer**, stated with its scope conditions rather than as a general claim
  about the two strategies.

---

## Study design

![Conceptual framework: two systems, five criteria, two scoring routes, stakeholder weighting, weighted aggregation](docs/figures/evaluation-framework.svg)

| Criterion | Definition | Scoring |
|---|---|---|
| Answer correctness | Share of answers that are factually and contextually right | 15 test questions, binary human judgement |
| Transparency | Whether the origin of an answer can be traced by the user | 15 test questions, binary judgement: were supporting sources shown |
| Recency | How quickly new or changed information reaches the answers | 4 binary sub-criteria from published evidence |
| Response time | Seconds from question to complete answer | Measured per question; fastest mean divided by system mean |
| Cost efficiency | Initial, inference and scaling cost over the lifecycle | 3 binary sub-criteria from published evidence |

Recency and cost efficiency were not measured on the prototypes. Both depend heavily on
deployment scale and update frequency, which a 15-question prototype study cannot represent, so
they are scored against the literature on binary sub-criteria that each carry their sources in
[`literature_criteria.yaml`](data/case_study/literature_criteria.yaml). That is a weaker form of
evidence than the measured criteria, and it is marked as such everywhere it appears.

![Prototype architecture: a Streamlit interface passes the question to either the RAG pipeline or the fine-tuning pipeline, both generating with GPT-3.5-Turbo](docs/figures/prototype-architecture.svg)

Both variants share the corpus, the interface, the generation model (GPT-3.5-Turbo, temperature
0.7) and the question set. The RAG variant embeds the posts with `all-MiniLM-L6-v2`, retrieves
the five most similar ones from a Pinecone index and shows them next to the answer. The
fine-tuning variant trains a GPT-3.5-Turbo instance on question-answer pairs derived from the
same posts and answers without retrieval. Full configuration in
[`docs/system-design.md`](docs/system-design.md).

The 15 questions cover three areas of a portfolio strategist's work: market and competition,
customer sentiment, product and technology. Questions, answers, response times and judgements are
committed verbatim in [`measurements.csv`](data/case_study/measurements.csv), so any individual
judgement can be disputed and the result recomputed.

### Weights

Five practitioners with at least three years in portfolio and market strategy ranked the five
criteria in semi-structured interviews. A forced ranking was used rather than a rating scale,
which prevents the "everything is important" pattern.

| Criterion | Mean rank | SD | Weight |
|---|---|---|---|
| Answer correctness | 4.60 | 0.55 | 0.3067 |
| Transparency | 4.20 | 0.84 | 0.2800 |
| Recency | 3.00 | 0.71 | 0.2000 |
| Response time | 1.60 | 0.55 | 0.1067 |
| Cost efficiency | 1.60 | 0.89 | 0.1067 |

Agreement across the five rankings: **Kendall's W = 0.792**. The chi-square approximation gives
p = 0.0032, but it is coarse at five raters, so a Monte-Carlo permutation test over 20,000
resamples is reported alongside it: p = 0.0001. Agreement was highest on answer correctness and
response time (SD 0.55) and lowest on cost efficiency (SD 0.89).

---

## Results

| Criterion | Weight | RAG | Fine-tuning |
|---|---|---|---|
| Answer correctness | 0.3067 | 0.867 (13/15) | 0.733 (11/15) |
| Transparency | 0.2800 | 1.000 (15/15) | 0.000 (0/15) |
| Recency | 0.2000 | 0.750 (3/4) | 0.250 (1/4) |
| Response time | 0.1067 | 0.742 (mean 3.96 s) | 1.000 (mean 2.94 s) |
| Cost efficiency | 0.1067 | 0.667 (2/3) | 0.333 (1/3) |
| **Weighted utility** | | **0.8461** | **0.4171** |

![Weighted utility by criterion for both systems](results/figures/utility_contributions.png)

Fine-tuning answers faster, 2.94 s against 3.96 s, because there is no retrieval step, and it is
cheaper per request. It loses on the criteria the stakeholders ranked higher. The gap of 0.4290
decomposes as:

| Criterion | Contribution to gap | Share |
|---|---|---|
| Transparency | 0.2800 | 65.3 % |
| Recency | 0.1000 | 23.3 % |
| Answer correctness | 0.0409 | 9.5 % |
| Cost efficiency | 0.0356 | 8.3 % |
| Response time | -0.0275 | -6.4 % |

The single largest term is not a quality difference. The fine-tuning prototype scored 0 on
transparency because it had no retrieval layer to expose, so the criterion the stakeholders
ranked second was decided by an architectural choice. That makes the robustness analysis the more
important half of the evaluation, not an appendix to it.

> **Reproducibility note.** The repository implementation yields 0.8461 and 0.4171. The thesis
> reported 0.8453 and 0.4167 because intermediate values were rounded before aggregation; that
> convention is reproducible with `--rounding thesis`. Ranking and interpretation are unchanged.

## Robustness

![Leave-one-out utilities and the weight at which fine-tuning would overtake RAG](results/figures/sensitivity.png)

- **Leave one criterion out.** RAG stays ahead in all five cases. The smallest remaining gap is
  0.2069, when transparency is removed and the other weights are rescaled proportionally.
- **Perturb the weights.** Over 10,000 weight vectors drawn from a Dirichlet distribution centred
  on the elicited weights, RAG wins 100 % of the time; its advantage never falls below 0.2022
  (5th percentile 0.3321, mean 0.4289).
- **Rank reversal.** Response time is the only criterion whose weight can flip the ranking, and it
  would have to rise from 0.1067 to **0.6648**. No change to any other single weight reverses the
  result.

Within this utility model and case study, fine-tuning therefore overtakes RAG only for a
stakeholder profile that weights response time at roughly two thirds of the total, far outside the
range the interviews produced. The analysis tests robustness to the specified weight model. It
does not establish robustness to measurement error, to a different set or operationalisation of
criteria, to a different sample of experts, or to different model providers.

## Repository extension beyond the thesis

The submitted thesis contributed the case study, the criteria, the elicited weights, the
measurements and the additive utility analysis, computed in a spreadsheet. This repository is a
later reimplementation that adds the engineering and the parts of the robustness analysis marked
below, so the two should not be read as one artifact.

| | Submitted thesis | This repository |
|---|---|---|
| Criteria, prototypes, measurements | yes | unchanged, committed as raw inputs |
| Expert weights, Kendall's W | yes, chi-square approximation | plus a seeded Monte-Carlo permutation test |
| Additive utility, gap decomposition | yes | unchanged, recomputed from the inputs |
| Arithmetic | rounded intermediate values | exact by default, thesis rounding reproducible on request |
| Leave-one-criterion-out | for transparency only | for all five criteria |
| Weight perturbation | no | Dirichlet sampling around the elicited weights |
| Rank-reversal thresholds | no | closed-form, per criterion |
| Implementation | spreadsheet | tested Python package, CLI, deterministic reports, CI |

No measurement, judgement or result was changed while porting the analysis.

## Scope and limitations

Five experts and 15 questions per system are a small sample, sized for an exploratory study in one
department, and the weights express that department's priorities rather than a general preference
profile. Answer correctness and transparency were judged by a single annotator, so no
inter-annotator agreement exists. Recency and cost efficiency were scored from published evidence
rather than measured, which is weaker evidence than the other three criteria. The prototypes
answered over a largely synthetic corpus, so correctness was judged against corpus content rather
than against ground truth about the real market. Response times are one observation per question
against hosted provider APIs and depend on provider load at the time, and both prototypes depend
on external services whose behaviour can change. The additive model assumes the criteria are
preferentially independent and does not represent interactions between them. The sensitivity
analysis probes robustness to the weights, not to any of the above. Detail in
[`docs/limitations.md`](docs/limitations.md).

## Reproducing

Every table and figure in this README is generated from the committed inputs. CI regenerates them
on Python 3.11 and 3.12 and fails if the committed results drift.

```bash
uv sync --extra plots
uv run rag-ft-eval run data/case_study/config.yaml                     # exact arithmetic -> results/
uv run rag-ft-eval run data/case_study/config.yaml --rounding thesis --output results/thesis_rounding
uv run pytest                                                           # 43 offline tests, no API keys
```

Without `uv`:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -e .
rag-ft-eval run data/case_study/config.yaml
```

`rag-ft-eval weights CONFIG` prints the weights and agreement statistics, `rag-ft-eval sensitivity
CONFIG` the robustness tables, and `rag-ft-eval validate CONFIG` checks a configuration and its
inputs without computing anything.

### Applying it to another comparison

The package is not specific to this case study. Point a configuration at your own inputs;
[`config.yaml`](data/case_study/config.yaml) documents the schema. Expert rankings are a long
table of `expert_id, metric, rank`; measurements hold one row per question and system; literature
criteria are binary sub-criteria with citation keys. The loader rejects non-permutation rankings,
values outside {0, 1}, incomplete question grids and unresolved citation keys with a precise
message. The Python API mirrors the pipeline: `compute_weights`, `build_score_matrix`, `utility`,
`sensitivity`.

## Repository layout

```
src/rag_ft_eval/
  weights.py       expert rankings -> weights; Kendall's W with chi-square and permutation test
  metrics.py       binary, ratio-to-best and literature scores; exact / thesis rounding
  utility.py       weighted additive utility and gap decomposition
  sensitivity.py   leave-one-out, Dirichlet perturbation, rank-reversal thresholds
  io.py            configuration and input loading with validation
  evaluate.py      end-to-end run and output writing
  report.py        deterministic Markdown report
  plots.py         result figures
  cli.py           rag-ft-eval run | weights | sensitivity | validate
data/case_study/   the raw research inputs: rankings, questions, measurements, literature criteria
results/           committed output, exact arithmetic; results/thesis_rounding/ for the thesis convention
tests/             43 offline tests, including the golden test that pins results/ to the code
docs/
  methodology.md   every formula the package computes, with references
  system-design.md configuration of the two prototypes and what was measured how
  limitations.md   what this study can and cannot support
  figures/         diagrams used in the documentation
  thesis_figures/  the original German thesis figures, kept for provenance
```

## Contact

If you work on LLM evaluation, human-AI interaction, or decision support for AI system selection,
I am happy to discuss the method, the study design or the implementation.

Cecilia Nothstein, <Cecilia.Nothstein@gmail.com>

Project thesis, DHBW Stuttgart, 2025. Released under the MIT license; cite via
[`CITATION.cff`](CITATION.cff).
