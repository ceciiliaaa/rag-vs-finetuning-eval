"""Text embedding, with the model the study used and a dependency-free stand-in.

``SentenceTransformerEmbedder`` is the real one: ``all-MiniLM-L6-v2``, 384 dimensions, the model
named in the thesis. It needs the ``rag`` extra and downloads weights on first use.

``HashingEmbedder`` is a deterministic bag-of-words hasher with no model behind it. It exists so
that the retrieval path can be exercised in tests and in the offline demo without a download. It
captures lexical overlap only, so its neighbours are not semantic neighbours. Do not use it to
draw conclusions about retrieval quality.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN = re.compile(r"[\w'-]+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into unit-length vectors of a fixed width."""

    @property
    def dimension(self) -> int: ...

    @property
    def name(self) -> str: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so that a dot product is the cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SentenceTransformerEmbedder:
    """The embedding model used in the study.

    The import is deferred to construction time so that importing this module costs nothing.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - needs the optional extra
            raise RuntimeError(
                "SentenceTransformerEmbedder needs the 'rag' extra: uv sync --extra rag"
            ) from exc
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self._model.encode(list(texts)), dtype=np.float32)
        return _normalise(vectors)


class HashingEmbedder:
    """Deterministic hashing embedder: lexical overlap only, no model, no network.

    Each token is hashed into one of ``dimension`` buckets with a signed weight, which is the
    hashing-trick construction. The result is stable across runs and machines because it uses
    BLAKE2b rather than Python's randomised ``hash``.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"hashing-{self._dimension}"

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if value & 1 else -1.0
        return value % self._dimension, sign

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                index, sign = self._bucket(token)
                vectors[row, index] += sign
        return _normalise(vectors)
