# Limitations

The evaluation framework is general. The bundled case study is small and should be read as a
worked application of the method, not as a general verdict on retrieval-augmented generation
versus supervised fine-tuning.

## Sample and judgement

* **Five experts.** The weights come from five practitioners in one department, sized for an
  exploratory study. Kendall's W is reported with a permutation test because the chi-square
  approximation is coarse at this size, but agreement among five raters does not make the profile
  representative of other organisations.
* **Fifteen questions per system.** The question set was designed with the stakeholders to cover
  their actual work, not to be a statistically powered benchmark.
* **One annotator.** Answer correctness and transparency were judged binarily by a single
  domain-familiar annotator against a written rule set. No inter-annotator agreement is available.
  The answers and judgements are committed verbatim so that individual calls can be contested.

## Evidence quality

* **Two criteria are not measured.** Recency and cost efficiency were scored from published
  evidence on binary sub-criteria, each with cited sources. This is weaker evidence than the three
  measured criteria and is marked as such wherever it appears.
* **Synthetic corpus.** The prototypes answered over a corpus of social-media-style posts about
  Mercedes-Benz and AMG vehicles that was largely template-generated, with a small share of
  authentic public posts. Correctness was judged against corpus content, not against ground truth
  about the real market. The corpus is not part of this release.
* **One latency observation per question.** Response times were measured once per question and
  system on a developer machine against hosted APIs, so they carry provider load and network
  conditions at the time of measurement.
* **Provider dependence.** Both prototypes used hosted models and a managed vector index. Model or
  service changes alter the measured behaviour and cannot be pinned by this repository.

## Model assumptions

* **Additive utility.** The aggregation assumes the criteria are preferentially independent.
  Interactions, for instance between transparency and perceived correctness, are not represented.
* **Architectural scores.** The fine-tuning prototype scored 0 on transparency because it had no
  retrieval layer to expose. This is a property of the evaluated artifact. Fine-tuned systems can
  be given provenance mechanisms; that variant was not built and therefore not measured.
* **Scope of the sensitivity analysis.** Leave-one-out, weight perturbation and rank-reversal
  thresholds test how far the result survives changes to the *weights*. They say nothing about
  robustness to measurement error, to a different set or operationalisation of criteria, to a
  different expert sample, or to different model providers.
* **Context-specific result.** The ranking holds for this application context, these criteria and
  this stakeholder profile. It is not a general statement about the two strategies.
