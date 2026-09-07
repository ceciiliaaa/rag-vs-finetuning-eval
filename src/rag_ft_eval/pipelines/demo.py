"""End-to-end run of both pipelines with no credentials and no downloads.

    python -m rag_ft_eval.pipelines.demo

Uses the hashing embedder, the in-memory store and the extractive generator, so it exercises the
real wiring while needing nothing beyond NumPy and pandas. The generation and fine-tuning steps
that call a provider are described but not run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_corpus
from .finetuning.dataset import build_training_examples
from .rag import ExtractiveGenerator, HashingEmbedder, InMemoryVectorStore, RagAssistant

DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / "data" / "sample_corpus" / "sample_posts.csv"

QUESTIONS = (
    "Was sagen Kundinnen und Kunden über den Verbrauch beim GLE?",
    "What do customers say about service costs?",
    "Wie bewerten Kunden den Innenraum der C-Klasse?",
)


def run(corpus_path: Path, top_k: int = 3) -> int:
    corpus = load_corpus(corpus_path)
    embedder = HashingEmbedder(dimension=384)
    store = InMemoryVectorStore(dimension=embedder.dimension)
    assistant = RagAssistant(
        embedder=embedder,
        store=store,
        generator=ExtractiveGenerator(max_quotes=2),
        top_k=top_k,
    )

    indexed = assistant.index_corpus(corpus)
    print(f"corpus            {corpus_path}")
    print(f"indexed           {indexed} posts with {embedder.name}")
    print(f"store             {type(store).__name__}, {store.count()} vectors\n")

    print("=== RAG pipeline ===")
    for question in QUESTIONS:
        answer = assistant.ask(question)
        print(f"\nQ ({answer.language})  {question}")
        print(f"retrieved   {len(answer.sources)} posts in {answer.elapsed_seconds * 1000:.1f} ms")
        for doc in answer.sources:
            print(f"  - {doc.id}  distance {doc.distance:.3f}  {doc.text[:70]}...")
        print("answer      " + answer.answer.replace("\n", "\n            "))

    print("\n\n=== Fine-tuning pipeline ===")
    examples = build_training_examples(corpus, max_examples=6, seed=0)
    print(f"built {len(examples)} training examples from the same corpus\n")
    first = examples[0]
    print(f"  system     {first.system}")
    print(f"  user       {first.user}")
    print(f"  assistant  {first.assistant}")
    print(
        "\nThe assistant turn restates the source post, because the targets are built by rule "
        "rather than written by hand.\nSee dataset.py for what that means for a model trained "
        "this way."
    )
    print("\nUploading the file and starting a job need OPENAI_API_KEY and are not run here.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    return run(args.corpus, args.top_k)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
