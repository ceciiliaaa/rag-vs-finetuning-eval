"""Building the supervised fine-tuning set from the corpus.

Fine-tuning needs prompt and completion pairs, but the corpus is a pile of unlabelled posts. The
study bridged that gap with rules rather than with human annotation, and this module reproduces
that construction so the training set can be rebuilt from the corpus.

Read the two builders below before drawing conclusions from a model trained this way:

* ``rule_based_question`` picks one of four question templates per language from keywords in the
  post, so many posts collapse onto the same question.
* ``rule_based_answer`` truncates the post to 200 characters and wraps it in one of three
  sentences chosen by a sentiment word list. The assistant turn is therefore a lightly rephrased
  excerpt of the post, not an independently written answer.

The consequence is that the model learns the register and vocabulary of the corpus rather than
verified question-answer behaviour. ``build_training_examples`` takes the two builders as
arguments so that a curated or model-assisted set can be substituted without touching the rest
of the pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..common import load_corpus

MIN_TEXT_LENGTH = 50
ANSWER_EXCERPT_CHARS = 200

SYSTEM_PROMPT = {
    "de": (
        "Du bist ein Analyst für Kundenfeedback zu Fahrzeugen. Du antwortest auf Basis von "
        "Kundenbeiträgen, die du im Training gesehen hast."
    ),
    "en": (
        "You are an analyst for automotive customer feedback. You answer from customer posts "
        "seen during training."
    ),
}

_MODEL_KEYWORDS = (
    "GLE",
    "GLC",
    "GLS",
    "GLA",
    "GLB",
    "C-Class",
    "E-Class",
    "S-Class",
    "A-Class",
    "CLA",
    "CLS",
    "AMG",
    "C-Klasse",
    "E-Klasse",
    "S-Klasse",
)

_QUESTIONS = {
    "de": {
        "service": "Wie sind die Erfahrungen mit dem Service?",
        "problem": "Welche Probleme berichten Kundinnen und Kunden?",
        "model": "Was sagen Kundinnen und Kunden über den {model}?",
        "general": "Wie fallen die Kundenerfahrungen insgesamt aus?",
    },
    "en": {
        "service": "What are the experiences with service?",
        "problem": "What problems do customers report?",
        "model": "What do customers say about the {model}?",
        "general": "What are the overall customer experiences?",
    },
}

_ANSWERS = {
    "de": {
        "positive": "Aus den Kundenbeiträgen: {excerpt} Das zeigt eine positive Erfahrung.",
        "negative": "Aus den Kundenbeiträgen: {excerpt} Das zeigt Verbesserungspotenzial.",
        "neutral": "Aus den Kundenbeiträgen: {excerpt} Das ist eine typische Rückmeldung.",
    },
    "en": {
        "positive": "From the customer posts: {excerpt} This reflects a positive experience.",
        "negative": "From the customer posts: {excerpt} This points to room for improvement.",
        "neutral": "From the customer posts: {excerpt} This is a typical piece of feedback.",
    },
}

_POSITIVE_WORDS = (
    "gut",
    "toll",
    "empfehlen",
    "zufrieden",
    "good",
    "great",
    "recommend",
    "satisfied",
)
_NEGATIVE_WORDS = ("schlecht", "problem", "teuer", "bad", "expensive", "issue", "fehler")


@dataclass(frozen=True)
class TrainingExample:
    """One chat-format example: a system, a user and an assistant turn."""

    system: str
    user: str
    assistant: str

    def to_chat(self) -> dict[str, list[dict[str, str]]]:
        return {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ]
        }


def rule_based_question(text: str, language: str) -> str:
    """Pick one of four question templates from keywords in the post."""
    lang = language if language in _QUESTIONS else "en"
    lowered = text.lower()
    templates = _QUESTIONS[lang]
    if "service" in lowered or "werkstatt" in lowered:
        return templates["service"]
    if any(word in lowered for word in ("problem", "issue", "fehler")):
        return templates["problem"]
    for model in _MODEL_KEYWORDS:
        if model.lower() in lowered:
            return templates["model"].format(model=model)
    return templates["general"]


def rule_based_answer(text: str, language: str) -> str:
    """Wrap a truncated excerpt of the post in one of three sentences.

    The excerpt is the post itself, so the assistant turn restates the input rather than
    answering it independently.
    """
    lang = language if language in _ANSWERS else "en"
    excerpt = " ".join(text.split())
    if len(excerpt) > ANSWER_EXCERPT_CHARS:
        excerpt = excerpt[:ANSWER_EXCERPT_CHARS].rstrip() + "..."
    lowered = excerpt.lower()
    if any(word in lowered for word in _POSITIVE_WORDS):
        tone = "positive"
    elif any(word in lowered for word in _NEGATIVE_WORDS):
        tone = "negative"
    else:
        tone = "neutral"
    return _ANSWERS[lang][tone].format(excerpt=excerpt)


def build_training_examples(
    corpus: pd.DataFrame | str | Path,
    max_examples: int = 500,
    min_text_length: int = MIN_TEXT_LENGTH,
    seed: int = 0,
    question_builder: Callable[[str, str], str] = rule_based_question,
    answer_builder: Callable[[str, str], str] = rule_based_answer,
) -> list[TrainingExample]:
    """Sample the corpus evenly across languages and build one example per post.

    Posts shorter than ``min_text_length`` characters are skipped. The sample is seeded, so
    rebuilding the set from the same corpus gives the same file; the original study run drew its
    sample without a seed and is therefore not reproducible from the corpus alone.
    """
    frame = corpus if isinstance(corpus, pd.DataFrame) else load_corpus(corpus)
    usable = frame[frame["text"].str.len() >= min_text_length]
    if usable.empty:
        raise ValueError(f"no posts of at least {min_text_length} characters")

    languages = sorted(usable["language"].unique())
    per_language = max(1, max_examples // max(len(languages), 1))

    examples: list[TrainingExample] = []
    for language in languages:
        subset = usable[usable["language"] == language]
        take = min(per_language, len(subset))
        sample = subset.sample(n=take, random_state=seed)
        prompt = SYSTEM_PROMPT.get(str(language), SYSTEM_PROMPT["en"])
        for row in sample.itertuples():
            examples.append(
                TrainingExample(
                    system=prompt,
                    user=question_builder(row.text, str(language)),
                    assistant=answer_builder(row.text, str(language)),
                )
            )
    return examples


def write_jsonl(examples: list[TrainingExample], path: str | Path) -> Path:
    """Write the examples in the chat JSONL format the fine-tuning API expects."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_chat(), ensure_ascii=False) + "\n")
    return target
