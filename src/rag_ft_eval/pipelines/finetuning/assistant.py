"""Asking the fine-tuned model a question.

There is no retrieval step. The question goes straight to the model, and the answer comes back
with nothing attached to check it against. That absence is the whole of the transparency
difference between the two pipelines, and it is visible here: :class:`Answer` is returned with
an empty ``sources`` tuple.
"""

from __future__ import annotations

import os
import time

from ..common import Answer, detect_language

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


class FineTunedAssistant:
    """Chat against a fine-tuned model instance."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - needs the optional extra
            raise RuntimeError(
                "FineTunedAssistant needs the 'openai' extra: uv sync --extra openai"
            ) from exc
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("set OPENAI_API_KEY, or pass api_key")
        if not model:
            raise ValueError("a fine-tuned model id is required")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=key)

    def ask(self, question: str, language: str | None = None) -> Answer:
        """Answer a question. ``sources`` is always empty: nothing was retrieved."""
        started = time.perf_counter()
        lang = language or detect_language(question)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.get(lang, SYSTEM_PROMPT["en"])},
                {"role": "user", "content": question},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return Answer(
            question=question,
            language=lang,
            answer=(response.choices[0].message.content or "").strip(),
            sources=(),
            generator=self.model,
            elapsed_seconds=time.perf_counter() - started,
        )
