"""Vector stores: the managed index the study used, and an in-memory one for offline runs.

Both return distances rather than similarities, using ``distance = 1 - cosine_similarity`` so
that the two are directly comparable and the threshold from the prototype carries over.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..common import RetrievedDocument


@runtime_checkable
class VectorStore(Protocol):
    """Stores vectors with metadata and answers nearest-neighbour queries."""

    def upsert(
        self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]
    ) -> None: ...

    def query(
        self, vector: np.ndarray, top_k: int, language: str | None = None
    ) -> list[RetrievedDocument]: ...

    def count(self) -> int: ...


class InMemoryVectorStore:
    """Exact cosine search over a NumPy matrix.

    Suitable for the corpus sizes in this study (thousands of rows). It is exact, so it is also
    the reference against which an approximate index can be checked.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._matrix = np.zeros((0, dimension), dtype=np.float32)

    def upsert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"expected {self._dimension}-dimensional vectors, got {vectors.shape[1]}"
            )
        if not (len(ids) == vectors.shape[0] == len(metadata)):
            raise ValueError("ids, vectors and metadata must have the same length")
        known = {doc_id: i for i, doc_id in enumerate(self._ids)}
        fresh_rows, fresh_ids, fresh_meta = [], [], []
        for doc_id, vector, meta in zip(ids, vectors, metadata, strict=True):
            if doc_id in known:
                self._matrix[known[doc_id]] = vector
                self._metadata[known[doc_id]] = meta
            else:
                fresh_ids.append(doc_id)
                fresh_rows.append(vector)
                fresh_meta.append(meta)
        if fresh_rows:
            self._matrix = np.vstack([self._matrix, np.asarray(fresh_rows, dtype=np.float32)])
            self._ids.extend(fresh_ids)
            self._metadata.extend(fresh_meta)

    def query(
        self, vector: np.ndarray, top_k: int, language: str | None = None
    ) -> list[RetrievedDocument]:
        if self._matrix.shape[0] == 0:
            return []
        similarities = self._matrix @ np.asarray(vector, dtype=np.float32).ravel()
        candidates = range(len(self._ids))
        if language is not None:
            candidates = [i for i in candidates if self._metadata[i].get("language") == language]
            if not candidates:
                return []
        ordered = sorted(candidates, key=lambda i: -float(similarities[i]))[:top_k]
        return [
            RetrievedDocument(
                id=self._ids[i],
                text=str(self._metadata[i].get("text", "")),
                distance=float(1.0 - similarities[i]),
                metadata={k: v for k, v in self._metadata[i].items() if k != "text"},
            )
            for i in ordered
        ]

    def count(self) -> int:
        return len(self._ids)


class PineconeVectorStore:
    """The managed serverless index used in the study.

    Mirrors the prototype: an index named ``mercedes-insights`` with cosine similarity, created
    on demand, upserts in batches of 100, and an optional metadata filter on the language. The
    post text travels in the metadata so a query returns it without a second lookup.
    """

    def __init__(
        self,
        dimension: int,
        index_name: str = "mercedes-insights",
        api_key: str | None = None,
        cloud: str = "aws",
        region: str = "us-east-1",
        create_if_missing: bool = True,
    ) -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:  # pragma: no cover - needs the optional extra
            raise RuntimeError(
                "PineconeVectorStore needs the 'rag' extra: uv sync --extra rag"
            ) from exc

        key = api_key or os.getenv("PINECONE_API_KEY")
        if not key:
            raise RuntimeError("set PINECONE_API_KEY, or pass api_key")

        self._dimension = dimension
        self.index_name = index_name
        client = Pinecone(api_key=key)
        existing = {idx["name"] for idx in client.list_indexes()}
        if index_name not in existing:
            if not create_if_missing:
                raise RuntimeError(f"index {index_name!r} does not exist")
            client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self._index = client.Index(index_name)

    def upsert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        batch = 100
        for start in range(0, len(ids), batch):
            stop = start + batch
            self._index.upsert(
                vectors=[
                    {"id": doc_id, "values": vec.tolist(), "metadata": meta}
                    for doc_id, vec, meta in zip(
                        ids[start:stop], vectors[start:stop], metadata[start:stop], strict=True
                    )
                ]
            )

    def query(
        self, vector: np.ndarray, top_k: int, language: str | None = None
    ) -> list[RetrievedDocument]:
        response = self._index.query(
            vector=np.asarray(vector, dtype=np.float32).ravel().tolist(),
            top_k=top_k,
            include_metadata=True,
            filter={"language": {"$eq": language}} if language else None,
        )
        documents = []
        for match in response["matches"]:
            meta = dict(match.get("metadata") or {})
            documents.append(
                RetrievedDocument(
                    id=match["id"],
                    text=str(meta.pop("text", "")),
                    distance=float(1.0 - match["score"]),
                    metadata=meta,
                )
            )
        return documents

    def count(self) -> int:
        return int(self._index.describe_index_stats().get("total_vector_count", 0))
