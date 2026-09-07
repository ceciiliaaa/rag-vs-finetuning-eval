"""Retrieval-augmented generation pipeline."""

from .assistant import RagAssistant
from .embedding import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from .generation import ExtractiveGenerator, Generator, OpenAIGenerator, build_prompt
from .stores import InMemoryVectorStore, PineconeVectorStore, VectorStore

__all__ = [
    "Embedder",
    "ExtractiveGenerator",
    "Generator",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "OpenAIGenerator",
    "PineconeVectorStore",
    "RagAssistant",
    "SentenceTransformerEmbedder",
    "VectorStore",
    "build_prompt",
]
