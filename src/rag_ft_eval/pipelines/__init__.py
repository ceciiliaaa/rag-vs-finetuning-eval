"""Reference implementations of the two domain-adaptation pipelines that were evaluated.

These modules reconstruct the prototypes described in ``docs/system-design.md``: a
retrieval-augmented assistant and a supervised fine-tuning workflow, both over the same corpus
of social-media posts.

They are deliberately kept out of the package's public API in ``rag_ft_eval/__init__.py``. The
evaluation framework has no dependency on them, and importing it must not pull in
sentence-transformers, Pinecone or the OpenAI client. Every heavy or network-bound dependency is
imported lazily, inside the function or constructor that needs it, so the offline path runs with
nothing beyond NumPy and pandas.

Two implementations of each moving part are provided:

* embeddings: a sentence-transformer model, or a deterministic hashing embedder for tests
* vector store: a managed Pinecone index, or an in-memory NumPy store
* generation: the OpenAI chat API, or an extractive fallback that quotes the retrieved posts

The first of each pair is what the study used. The second exists so that the pipeline can be run
and tested end to end without credentials, which is what ``rag_ft_eval.pipelines.demo`` does.
"""

__all__: list[str] = []
