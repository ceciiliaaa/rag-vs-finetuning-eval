# Sample corpus

`sample_posts.csv` holds 24 short posts, twelve German and twelve English, written for this
repository so that the pipelines in `src/rag_ft_eval/pipelines/` can be run and tested without
credentials or a download.

**This is not the corpus the study used.** It is not a sample of it either. Every row was
written by hand as a plausible-looking piece of automotive customer feedback, which is why the
`source` column says `synthetic` for all of them. Nothing here came from Reddit, YouTube, a
forum, or any other platform, and no real person's text is included.

The corpus the prototypes actually answered over is not part of this release, for the reasons in
[`../../docs/limitations.md`](../../docs/limitations.md).

| Column | Meaning |
|---|---|
| `id` | unique row id |
| `text` | the post |
| `language` | `de` or `en` |
| `source` | always `synthetic` here; the pipelines only use it as metadata |

Use it to see the pipelines work end to end, not to measure retrieval quality. Twenty-four rows
on a hashing embedder tell you the wiring is correct and nothing more.
