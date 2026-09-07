"""Offline tests for both reference pipelines.

Everything here runs without credentials, without a network and without a model download: the
hashing embedder, the in-memory store and the extractive generator cover the whole path. The
provider-backed classes are only checked for the error they raise when their extra is missing or
no key is set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rag_ft_eval.pipelines.common import (
    Answer,
    RetrievedDocument,
    detect_language,
    load_corpus,
)
from rag_ft_eval.pipelines.finetuning.dataset import (
    build_training_examples,
    rule_based_answer,
    rule_based_question,
    write_jsonl,
)
from rag_ft_eval.pipelines.rag import (
    ExtractiveGenerator,
    HashingEmbedder,
    InMemoryVectorStore,
    RagAssistant,
    build_prompt,
)

REPO = Path(__file__).resolve().parents[1]
SAMPLE_CORPUS = REPO / "data" / "sample_corpus" / "sample_posts.csv"


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return load_corpus(SAMPLE_CORPUS)


@pytest.fixture
def assistant(corpus: pd.DataFrame) -> RagAssistant:
    embedder = HashingEmbedder(dimension=128)
    store = InMemoryVectorStore(dimension=embedder.dimension)
    bot = RagAssistant(embedder=embedder, store=store, generator=ExtractiveGenerator())
    bot.index_corpus(corpus)
    return bot


# --------------------------------------------------------------------------- corpus and language


def test_sample_corpus_is_declared_synthetic(corpus: pd.DataFrame):
    assert len(corpus) == 24
    assert set(corpus["language"]) == {"de", "en"}
    assert set(corpus["source"]) == {"synthetic"}


def test_load_corpus_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("id,text\n1,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_corpus(path)


def test_load_corpus_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "dup.csv"
    path.write_text("id,text,language,source\na,x y z,en,s\na,q r s,en,s\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_corpus(path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Was halten die Kunden vom GLE?", "de"),
        ("What do customers think about the GLE?", "en"),
        ("Wie ist der Verbrauch bei dem Modell?", "de"),
    ],
)
def test_detect_language(text: str, expected: str):
    assert detect_language(text) == expected


def test_detect_language_falls_back_on_a_tie():
    assert detect_language("GLE 2024", default="de") == "de"


# --------------------------------------------------------------------------- embedding and store


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dimension=64)
    first = embedder.encode(["the quick brown fox", "a different sentence"])
    second = embedder.encode(["the quick brown fox", "a different sentence"])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert first.shape == (2, 64)


def test_hashing_embedder_ranks_lexical_overlap_highest():
    embedder = HashingEmbedder(dimension=256)
    vectors = embedder.encode(
        ["fuel consumption is too high", "fuel consumption above the figure", "the boot is small"]
    )
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])


def test_in_memory_store_returns_nearest_first_and_filters_language():
    embedder = HashingEmbedder(dimension=128)
    store = InMemoryVectorStore(dimension=128)
    texts = ["service costs are high", "der Verbrauch ist hoch", "the boot is small"]
    store.upsert(
        ids=["a", "b", "c"],
        vectors=embedder.encode(texts),
        metadata=[
            {"text": texts[0], "language": "en", "source": "synthetic"},
            {"text": texts[1], "language": "de", "source": "synthetic"},
            {"text": texts[2], "language": "en", "source": "synthetic"},
        ],
    )
    assert store.count() == 3

    hits = store.query(embedder.encode(["service costs"])[0], top_k=3)
    assert hits[0].id == "a"
    assert all(hits[i].distance <= hits[i + 1].distance for i in range(len(hits) - 1))
    assert all(0.0 <= h.distance <= 2.0 for h in hits)

    german = store.query(embedder.encode(["Verbrauch"])[0], top_k=3, language="de")
    assert [h.id for h in german] == ["b"]
    assert store.query(embedder.encode(["x"])[0], top_k=3, language="fr") == []


def test_in_memory_store_upsert_replaces_an_existing_id():
    embedder = HashingEmbedder(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    meta = [{"text": "first", "language": "en", "source": "synthetic"}]
    store.upsert(["a"], embedder.encode(["first"]), meta)
    store.upsert(["a"], embedder.encode(["second"]), [{**meta[0], "text": "second"}])
    assert store.count() == 1
    assert store.query(embedder.encode(["second"])[0], top_k=1)[0].text == "second"


def test_in_memory_store_rejects_wrong_dimension():
    store = InMemoryVectorStore(dimension=8)
    with pytest.raises(ValueError, match="dimensional"):
        store.upsert(["a"], np.zeros((1, 4), dtype=np.float32), [{}])


def test_empty_store_answers_nothing():
    store = InMemoryVectorStore(dimension=16)
    assert store.query(np.zeros(16, dtype=np.float32), top_k=5) == []


# --------------------------------------------------------------------------- retrieval and answers


def test_assistant_indexes_the_whole_corpus(assistant: RagAssistant, corpus: pd.DataFrame):
    assert assistant.store.count() == len(corpus)


def test_assistant_retrieves_in_the_question_language(assistant: RagAssistant):
    answer = assistant.ask("Wie ist der Verbrauch beim GLE?")
    assert answer.language == "de"
    assert answer.sources
    assert all(doc.metadata["language"] == "de" for doc in answer.sources)


def test_assistant_applies_the_distance_threshold(assistant: RagAssistant):
    assistant.distance_threshold = 0.0
    assert assistant.ask("Wie ist der Verbrauch beim GLE?").sources == ()


def test_answer_carries_its_sources_and_serialises(assistant: RagAssistant):
    answer = assistant.ask("What do customers say about service costs?")
    assert isinstance(answer, Answer)
    assert answer.generator == "extractive"
    assert answer.elapsed_seconds >= 0
    payload = answer.to_dict()
    assert payload["language"] == "en"
    assert len(payload["sources"]) == len(answer.sources)
    assert all(set(s) == {"id", "text", "distance", "metadata"} for s in payload["sources"])


def test_extractive_generator_quotes_rather_than_writes():
    docs = [RetrievedDocument(id="a", text="service is expensive", distance=0.2)]
    text = ExtractiveGenerator().generate("why?", docs, "en")
    assert "service is expensive" in text
    assert "1 matching customer posts" in text


def test_extractive_generator_reports_no_evidence():
    assert "nothing substantive" in ExtractiveGenerator().generate("q", [], "en")
    assert "nichts Belastbares" in ExtractiveGenerator().generate("q", [], "de")


def test_build_prompt_truncates_and_caps_the_context():
    docs = [RetrievedDocument(id=str(i), text="x" * 500, distance=0.1) for i in range(5)]
    system, user = build_prompt("Frage?", docs, "de")
    assert "ausschließlich" in system
    assert user.count("[") == 3  # only the first three documents
    assert "x" * 301 not in user  # each truncated to 300 characters


# --------------------------------------------------------------------------- fine-tuning dataset


def test_rule_based_question_uses_the_expected_template():
    assert rule_based_question("Der Service war teuer", "de").startswith("Wie sind die Erfahrungen")
    assert rule_based_question("There was a problem", "en").startswith("What problems")
    assert "GLE" in rule_based_question("Mein GLE faehrt gut", "de")
    assert rule_based_question("Nice weather", "en").startswith("What are the overall")


def test_rule_based_answer_restates_the_post():
    post = "Der Verbrauch ist zu hoch und der Service teuer."
    answer = rule_based_answer(post, "de")
    assert post in answer
    assert answer.startswith("Aus den Kundenbeiträgen:")


def test_rule_based_answer_truncates_long_posts():
    """The excerpt is capped; the closing sentence of the template follows it."""
    answer = rule_based_answer("wort " * 200, "en")
    excerpt = answer.removeprefix("From the customer posts: ").split("...")[0]
    assert len(excerpt) <= 200
    assert "..." in answer
    assert answer.endswith("This is a typical piece of feedback.")


def test_build_training_examples_is_balanced_and_seeded(corpus: pd.DataFrame):
    first = build_training_examples(corpus, max_examples=8, seed=7)
    second = build_training_examples(corpus, max_examples=8, seed=7)
    assert [e.user for e in first] == [e.user for e in second]
    assert len(first) == 8
    assert sum("Kundenbeiträge" in e.assistant for e in first) == 4  # four German, four English


def test_build_training_examples_skips_short_posts(corpus: pd.DataFrame):
    with pytest.raises(ValueError, match="at least"):
        build_training_examples(corpus, min_text_length=10_000)


def test_write_jsonl_emits_the_chat_format(corpus: pd.DataFrame, tmp_path: Path):
    import json

    path = write_jsonl(
        build_training_examples(corpus, max_examples=4, seed=0), tmp_path / "t.jsonl"
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    for line in lines:
        messages = json.loads(line)["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert all(m["content"].strip() for m in messages)


# --------------------------------------------------------------------------- provider-backed paths


def test_provider_classes_refuse_to_run_without_a_key(monkeypatch: pytest.MonkeyPatch):
    """Without credentials these must fail loudly rather than silently degrade."""
    from rag_ft_eval.pipelines.finetuning.assistant import FineTunedAssistant
    from rag_ft_eval.pipelines.finetuning.job import FineTuningJob
    from rag_ft_eval.pipelines.rag.generation import OpenAIGenerator
    from rag_ft_eval.pipelines.rag.stores import PineconeVectorStore

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    for factory in (
        lambda: OpenAIGenerator(),
        lambda: FineTuningJob(),
        lambda: FineTunedAssistant(model="ft:demo"),
        lambda: PineconeVectorStore(dimension=384),
    ):
        with pytest.raises(RuntimeError):
            factory()


def test_importing_the_package_does_not_pull_in_heavy_dependencies():
    """`import rag_ft_eval` must stay cheap; the pipelines are opt-in."""
    import subprocess
    import sys

    code = (
        "import sys, rag_ft_eval;"
        "heavy = {'torch', 'sentence_transformers', 'pinecone', 'openai'};"
        "loaded = heavy & set(sys.modules);"
        "print(sorted(loaded))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_demo_runs_offline(capsys: pytest.CaptureFixture[str]):
    from rag_ft_eval.pipelines.demo import main

    assert main(["--corpus", str(SAMPLE_CORPUS)]) == 0
    out = capsys.readouterr().out
    assert "indexed           24 posts" in out
    assert "=== RAG pipeline ===" in out
    assert "=== Fine-tuning pipeline ===" in out
