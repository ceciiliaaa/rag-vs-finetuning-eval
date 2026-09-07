# Limitations of the case study

The evaluation framework is general; the bundled case study is small and should be read as a
worked example of the method, not as a general verdict on RAG versus fine-tuning.

* **Sample sizes.** Five experts ranked the criteria and 15 test questions were answered per
  system. Kendall's W is reported with a permutation test because the chi-square approximation is
  coarse at this size, but the weights still reflect one department's priorities.
* **Judgements.** Answer correctness and transparency were judged binarily (0/1) by a single
  domain-familiar annotator against the expert-agreed question catalogue. No inter-annotator
  agreement is available.
* **Two criteria are literature-based.** Recency and cost efficiency were not measured on the
  prototypes; they were scored from published evidence on binary sub-criteria. The sources are
  listed with each sub-criterion so that the judgement can be contested.
* **Synthetic corpus.** The prototypes answered questions over a corpus of social-media-style
  posts about Mercedes-Benz and AMG vehicles that was largely template-generated, with a small
  share of authentic public posts. Correctness was judged against the corpus content, not against ground truth about
  the real market. The corpus and the prototypes are not part of this release.
* **Latency.** Each question was timed once per system on a developer machine against hosted
  APIs; the values depend on provider load and network conditions at the time.
* **Provider dependence.** Both prototypes used hosted models (GPT-3.5-Turbo and a fine-tuned
  variant) and a managed vector index. Model or service updates change the measured behaviour.
* **Stakeholder profile.** The utility ranking holds for the elicited weight profile. The
  sensitivity analysis shows how far the weights can move before the ranking changes; different
  stakeholders may sit outside that range.
* **Additive model.** The utility is additive and assumes preferential independence of the
  criteria. Interactions (for example between transparency and correctness) are not modelled.
