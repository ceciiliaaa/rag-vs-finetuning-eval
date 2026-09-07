"""Answer generation from retrieved posts.

``OpenAIGenerator`` reproduces the prototype: GPT-3.5-Turbo, temperature 0.7, 250 output tokens,
a system prompt that forbids going beyond the retrieved posts, and the retrieved posts pasted
into the user turn.

``ExtractiveGenerator`` writes no prose of its own. It states how many posts matched and quotes
them. It is the offline fallback, and it is also the honest baseline: anything the OpenAI
generator adds on top of this is model output, not retrieved evidence.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from ..common import RetrievedDocument

CONTEXT_CHARS = 300
CONTEXT_DOCUMENTS = 3

SYSTEM_PROMPT = {
    "de": (
        "Du bist ein Analyst für Kundenfeedback. Beantworte die Frage in vier bis sechs Sätzen "
        "und stütze dich ausschließlich auf die angegebenen Kundenbeiträge.\n"
        "Erfinde nichts, was nicht in den Beiträgen steht. Wenn die Beiträge die Frage nicht "
        "beantworten, sage das. Sprich nur über das in der Frage genannte Modell."
    ),
    "en": (
        "You are an analyst for customer feedback. Answer the question in four to six sentences "
        "and rely only on the customer posts provided.\n"
        "Do not invent anything that is not in the posts. If the posts do not answer the "
        "question, say so. Only discuss the model named in the question."
    ),
}

NO_EVIDENCE = {
    "de": "In den vorliegenden Kundenbeiträgen finde ich dazu nichts Belastbares.",
    "en": "The available customer posts contain nothing substantive on this.",
}

_USER_TEMPLATE = {
    "de": "Frage: {question}\n\nKundenbeiträge:\n{context}\n\nAntworte auf Basis dieser Beiträge.",
    "en": "Question: {question}\n\nCustomer posts:\n{context}\n\nAnswer from these posts.",
}


def build_prompt(
    question: str, documents: list[RetrievedDocument], language: str
) -> tuple[str, str]:
    """Return the system and user message for a question and its retrieved posts.

    Only the first ``CONTEXT_DOCUMENTS`` posts enter the prompt and each is truncated to
    ``CONTEXT_CHARS`` characters, which is what the prototype did.
    """
    lang = language if language in SYSTEM_PROMPT else "en"
    context = "\n\n".join(
        f"[{i}] {doc.text[:CONTEXT_CHARS]}"
        for i, doc in enumerate(documents[:CONTEXT_DOCUMENTS], start=1)
    )
    return SYSTEM_PROMPT[lang], _USER_TEMPLATE[lang].format(question=question, context=context)


@runtime_checkable
class Generator(Protocol):
    """Turns a question plus retrieved posts into an answer."""

    @property
    def name(self) -> str: ...

    def generate(self, question: str, documents: list[RetrievedDocument], language: str) -> str: ...


class ExtractiveGenerator:
    """Quotes the retrieved posts instead of writing prose. No model, no network."""

    def __init__(self, max_quotes: int = 3, quote_chars: int = 200) -> None:
        self.max_quotes = max_quotes
        self.quote_chars = quote_chars

    @property
    def name(self) -> str:
        return "extractive"

    def generate(self, question: str, documents: list[RetrievedDocument], language: str) -> str:
        lang = language if language in NO_EVIDENCE else "en"
        if not documents:
            return NO_EVIDENCE[lang]
        head = (
            f"{len(documents)} passende Kundenbeiträge:"
            if lang == "de"
            else f"{len(documents)} matching customer posts:"
        )
        quotes = [
            f'{i}. "{doc.text[: self.quote_chars]}" (distance {doc.distance:.3f})'
            for i, doc in enumerate(documents[: self.max_quotes], start=1)
        ]
        return "\n".join([head, *quotes])


class OpenAIGenerator:
    """GPT-3.5-Turbo with the retrieved posts in the prompt, as in the study."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 250,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - needs the optional extra
            raise RuntimeError(
                "OpenAIGenerator needs the 'openai' extra: uv sync --extra openai"
            ) from exc
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("set OPENAI_API_KEY, or pass api_key")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=key)

    @property
    def name(self) -> str:
        return self.model

    def generate(self, question: str, documents: list[RetrievedDocument], language: str) -> str:
        if not documents:
            return NO_EVIDENCE[language if language in NO_EVIDENCE else "en"]
        system, user = build_prompt(question, documents, language)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()
