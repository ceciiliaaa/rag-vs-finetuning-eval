"""The retrieval-augmented assistant: index a corpus, then answer questions from it."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..common import Answer, RetrievedDocument, detect_language, load_corpus
from .embedding import Embedder
from .generation import ExtractiveGenerator, Generator
from .stores import VectorStore

TOP_K = 5
DISTANCE_THRESHOLD = 1.0


class RagAssistant:
    """Ties an embedder, a vector store and a generator into one question-answering pipeline.

    Defaults follow the evaluated prototype: five nearest posts, a language filter on the
    detected query language, and posts kept only below a cosine distance of 1.0. The retrieved
    posts are returned alongside the answer, which is what made the prototype auditable.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        generator: Generator | None = None,
        top_k: int = TOP_K,
        distance_threshold: float = DISTANCE_THRESHOLD,
        filter_by_language: bool = True,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.generator = generator or ExtractiveGenerator()
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self.filter_by_language = filter_by_language

    def index_corpus(self, corpus: pd.DataFrame | str | Path, batch_size: int = 100) -> int:
        """Embed every row and write it to the store. Returns the number of rows indexed."""
        frame = corpus if isinstance(corpus, pd.DataFrame) else load_corpus(corpus)
        total = 0
        for start in range(0, len(frame), batch_size):
            chunk = frame.iloc[start : start + batch_size]
            texts = chunk["text"].tolist()
            vectors = self.embedder.encode(texts)
            self.store.upsert(
                ids=[str(v) for v in chunk["id"].tolist()],
                vectors=vectors,
                metadata=[
                    {
                        "text": row.text,
                        "language": str(row.language),
                        "source": str(row.source),
                    }
                    for row in chunk.itertuples()
                ],
            )
            total += len(chunk)
        return total

    def retrieve(self, question: str, language: str | None = None) -> list[RetrievedDocument]:
        """Return the retained nearest posts for a question, closest first."""
        lang = language or detect_language(question)
        vector = self.embedder.encode([question])[0]
        found = self.store.query(
            vector,
            top_k=self.top_k,
            language=lang if self.filter_by_language else None,
        )
        return [doc for doc in found if doc.distance < self.distance_threshold]

    def ask(self, question: str, language: str | None = None) -> Answer:
        """Retrieve, then generate. The retrieved posts travel with the answer."""
        started = time.perf_counter()
        lang = language or detect_language(question)
        documents = self.retrieve(question, lang)
        text = self.generator.generate(question, documents, lang)
        return Answer(
            question=question,
            language=lang,
            answer=text,
            sources=tuple(documents),
            generator=self.generator.name,
            elapsed_seconds=time.perf_counter() - started,
        )
