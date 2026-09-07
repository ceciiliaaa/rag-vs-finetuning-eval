"""Pieces shared by both pipelines: the corpus schema, language detection and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

CORPUS_COLUMNS = ("id", "text", "language", "source")

# Function words used to tell German from English. This is the heuristic the prototypes used:
# it is cheap, has no dependencies and is good enough for one-sentence questions. It is not a
# general-purpose language identifier.
GERMAN_MARKERS = (
    "wie",
    "ist",
    "der",
    "die",
    "das",
    "mein",
    "bei",
    "und",
    "mit",
    "was",
    "halten",
    "kunden",
    "nicht",
    "für",
    "sind",
    "haben",
    "wird",
)
ENGLISH_MARKERS = (
    "how",
    "is",
    "the",
    "my",
    "with",
    "and",
    "what",
    "which",
    "customers",
    "think",
    "are",
    "does",
    "do",
    "of",
)


def detect_language(text: str, default: str = "en") -> str:
    """Return ``"de"`` or ``"en"`` by counting function words.

    Ties fall back to ``default``. The comparison is on whole words, so "the" in "there" does
    not count.
    """
    words = {w.strip(".,!?;:()\"'") for w in text.lower().split()}
    german = len(words & set(GERMAN_MARKERS))
    english = len(words & set(ENGLISH_MARKERS))
    if german == english:
        return default
    return "de" if german > english else "en"


def load_corpus(path: str | Path) -> pd.DataFrame:
    """Read a corpus CSV and validate the columns the pipelines rely on.

    Required columns are ``id``, ``text``, ``language`` and ``source``. Rows with an empty text
    are dropped, because neither pipeline can do anything with them.
    """
    frame = pd.read_csv(Path(path), encoding="utf-8")
    missing = set(CORPUS_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"corpus is missing columns {sorted(missing)}")
    frame = frame.copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    dropped = frame["text"] == ""
    if dropped.any():
        frame = frame.loc[~dropped]
    if frame.empty:
        raise ValueError("corpus contains no usable rows")
    if frame["id"].duplicated().any():
        raise ValueError("corpus ids must be unique")
    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class RetrievedDocument:
    """One post returned by the retriever.

    ``distance`` is ``1 - cosine_similarity``, so smaller means more similar and the range is
    [0, 2]. The prototypes kept documents below a distance threshold of 1.0.
    """

    id: str
    text: str
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


@dataclass(frozen=True)
class Answer:
    """What an assistant returns for one question."""

    question: str
    language: str
    answer: str
    sources: tuple[RetrievedDocument, ...]
    generator: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "language": self.language,
            "answer": self.answer,
            "generator": self.generator,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "sources": [
                {
                    "id": d.id,
                    "text": d.text,
                    "distance": round(d.distance, 6),
                    "metadata": d.metadata,
                }
                for d in self.sources
            ],
        }
