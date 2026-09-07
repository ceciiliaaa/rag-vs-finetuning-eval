# Evaluation report: RAG vs. fine-tuning for a domain-specific question-answering assistant

Context: Mercedes-AMG, portfolio strategy and market intelligence department, 2025
Rounding mode: `thesis` (rounding convention of the original write-up (intermediate values rounded)).

## 1. Criterion weights from expert rankings

5 experts ranked 5 criteria (forced ranking, higher rank = more important). Weights are mean ranks normalised to sum to one; the standard deviation is the sample standard deviation across experts.

| Criterion | Mean rank | SD | Weight | Weight (%) |
|---|---|---|---|---|
| Answer correctness | 4.60 | 0.55 | 0.3067 | 30.67 % |
| Transparency | 4.20 | 0.84 | 0.2800 | 28.00 % |
| Recency | 3.00 | 0.71 | 0.2000 | 20.00 % |
| Response time | 1.60 | 0.55 | 0.1067 | 10.67 % |
| Cost efficiency | 1.60 | 0.89 | 0.1067 | 10.67 % |

Agreement: Kendall's W = 0.792; chi-square = 15.84 with 4 degrees of freedom, p = 0.0032, permutation test p = 0.0001 (20000 samples).

## 2. Normalised criterion scores

| Criterion | Kind | RAG score | RAG basis | Fine-tuning score | Fine-tuning basis |
|---|---|---|---|---|---|
| Answer correctness | measured_binary | 0.867 | 13/15 | 0.733 | 11/15 |
| Transparency | measured_binary | 1.000 | 15/15 | 0.000 | 0/15 |
| Recency | literature | 0.750 | 3/4 sub-criteria | 0.250 | 1/4 sub-criteria |
| Response time | measured_latency | 0.730 | mean 3.96 s, sd 1.27 s, n = 15 | 1.000 | mean 2.94 s, sd 0.95 s, n = 15 |
| Cost efficiency | literature | 0.670 | 2/3 sub-criteria | 0.330 | 1/3 sub-criteria |

Binary criteria are the share of positive judgements. Response time uses ratio-to-best normalisation (fastest mean divided by the system mean). Literature-based criteria are the share of sub-criteria in favour of the system (section 5).

## 3. Weighted utility

| Criterion | Weight | RAG | Fine-tuning |
|---|---|---|---|
| Answer correctness | 0.3067 | 0.2659 | 0.2248 |
| Transparency | 0.2800 | 0.2800 | 0.0000 |
| Recency | 0.2000 | 0.1500 | 0.0500 |
| Response time | 0.1067 | 0.0779 | 0.1067 |
| Cost efficiency | 0.1067 | 0.0715 | 0.0352 |
| **Total utility** | 1.0001 | **0.8453** | **0.4167** |

RAG reaches the higher utility (0.8453 versus 0.4167 for Fine-tuning); the gap is 0.4286.

### Decomposition of the gap

| Criterion | Contribution to gap | Share of gap |
|---|---|---|
| Answer correctness | 0.0411 | 9.6 % |
| Transparency | 0.2800 | 65.3 % |
| Recency | 0.1000 | 23.3 % |
| Response time | -0.0288 | -6.7 % |
| Cost efficiency | 0.0363 | 8.5 % |

Negative shares mark criteria on which the runner-up is ahead.

## 4. Sensitivity to the weights

### Leave one criterion out (remaining weights renormalised)

| Dropped criterion | RAG | Fine-tuning | Delta | Winner |
|---|---|---|---|---|
| Answer correctness | 0.8355 | 0.2767 | 0.5588 | RAG |
| Transparency | 0.7849 | 0.5787 | 0.2062 | RAG |
| Recency | 0.8689 | 0.4583 | 0.4106 | RAG |
| Response time | 0.8589 | 0.3470 | 0.5119 | RAG |
| Cost efficiency | 0.8661 | 0.4270 | 0.4391 | RAG |

### Random weight perturbation

10000 weight vectors drawn from Dirichlet(concentration 50.0 x expert weights), seed 0, exact arithmetic. Win share: RAG 100.0 %, Fine-tuning 0.0 %. Utility advantage of RAG: mean 0.4284, 5th to 95th percentile 0.3314 to 0.5249, minimum 0.1992.

### Rank-reversal thresholds

Weight a single criterion would need (others rescaled proportionally) for Fine-tuning to overtake RAG.

| Criterion | Current weight | Reversal at weight | Direction |
|---|---|---|---|
| Answer correctness | 0.3067 | none in [0, 1] | no reversal possible |
| Transparency | 0.2800 | none in [0, 1] | no reversal possible |
| Recency | 0.2000 | none in [0, 1] | no reversal possible |
| Response time | 0.1067 | 0.6547 | increase |
| Cost efficiency | 0.1067 | none in [0, 1] | no reversal possible |

## 5. Literature-based criteria

### Recency

| Id | Sub-criterion | RAG | Fine-tuning | Sources |
|---|---|---|---|---|
| R1 | New knowledge becomes effective without retraining the model | 1 | 0 | lewis2020, gao2024, petroni2019, shi2024, wu2025 |
| R2 | The system stays available while knowledge is updated | 1 | 0 | gao2024, shi2024 |
| R3 | Low cost per knowledge update | 1 | 0 | gao2024, yang2025, mombaerts2024 |
| R4 | Model behaviour is stable after an update | 0 | 1 | gao2024, wu2025 |

### Cost efficiency

| Id | Sub-criterion | RAG | Fine-tuning | Sources |
|---|---|---|---|---|
| C1 | Low initial cost (setup and domain integration) | 1 | 0 | huang2024, han2024, wu2025 |
| C2 | Low inference cost per request | 0 | 1 | huang2024, wu2025, shi2024 |
| C3 | Predictable cost when data volume or usage grows | 1 | 0 | huang2024, gao2024, shi2024 |

### References

- `lewis2020`: Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems 33. https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
- `gao2024`: Gao, Y. et al. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv 2312.10997. doi:10.48550/arXiv.2312.10997
- `petroni2019`: Petroni, F. et al. (2019). Language Models as Knowledge Bases?. EMNLP-IJCNLP 2019. doi:10.18653/v1/D19-1250
- `shi2024`: Shi, T. et al. (2024). Preliminary Study on Incremental Learning for Large Language Model-based Recommender Systems. arXiv 2312.15599. doi:10.48550/arXiv.2312.15599
- `wu2025`: Wu, X.-K. et al. (2025). LLM Fine-Tuning: Concepts, Opportunities, and Challenges. Big Data and Cognitive Computing 9(4), 87. doi:10.3390/bdcc9040087
- `yang2025`: Yang, Z. et al. (2025). An Empirical Study of Retrieval-Augmented Code Generation: Challenges and Opportunities. ACM Transactions on Software Engineering and Methodology 34(7). doi:10.1145/3717061
- `mombaerts2024`: Mombaerts, L. et al. (2024). Meta Knowledge for Retrieval Augmented Large Language Models. arXiv 2408.09017. doi:10.48550/arXiv.2408.09017
- `huang2024`: Huang, Y. and Huang, J. (2024). A Survey on Retrieval-Augmented Text Generation for Large Language Models. arXiv 2404.10981. doi:10.48550/arXiv.2404.10981
- `han2024`: Han, Z. et al. (2024). Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey. arXiv 2403.14608. doi:10.48550/arXiv.2403.14608

---
Generated by rag-ft-eval 0.1.0 from `config.yaml` (15 test questions, 5 experts).
