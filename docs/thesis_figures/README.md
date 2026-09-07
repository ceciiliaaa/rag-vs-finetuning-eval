# Original thesis figures

These are the figures as they appear in the submitted project thesis. Their labels are German,
which is the language the thesis was written in. They are kept here for provenance: the
public-facing documents use redrawn English versions in [`../figures/`](../figures).

| File | Shows | English counterpart |
|---|---|---|
| `prototype-architecture.png` | System architecture of the chatbot prototype | `../figures/prototype-architecture.svg` |
| `rag-pipeline.png` | RAG pipeline of the prototype | `../figures/rag-pipeline.svg` |
| `evaluation-framework.png` | Conceptual framework of the evaluation | `../figures/evaluation-framework.svg` |
| `supervised-fine-tuning.png` | Schematic of supervised fine-tuning | none, textbook content |
| `rag-process.png` | Generic RAG process: indexing, retrieval, generation | none, textbook content |
| `rag-vs-finetuning-flow.jpg` | Both strategies answering the same question | none, textbook content |
| `interview-guide.png` | Interview guide used for the weight elicitation | `../../data/case_study/expert_rankings.csv` holds the resulting ranks |
| `expert-weighting.png` | Weighting table as computed in the thesis | reproduced in `../../results/report.md` |
| `test-question-catalogue.jpg` | All 15 questions with both systems' answers and judgements | `../../data/case_study/measurements.csv` |
| `decision-matrix.png` | Decision matrix with the thesis rounding | reproduced in `../../results/thesis_rounding/report.md` |
| `utility-by-criterion.png` | Utility by criterion, thesis version | regenerated in `../../results/figures/utility_contributions.png` |
| `stakeholder-alignment-radar.png` | Target versus measured profile, thesis version | regenerated in `../../results/figures/profile_radar.png` |

The last four are superseded: the same content is regenerated from the committed inputs by
`rag-ft-eval run`, so the code output is authoritative and these images only document what the
thesis itself printed.
