from __future__ import annotations

import os

from rag_os.core.types import RetrievalResult
from rag_os.generation.base import Generator, generator_registry

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided context. "
    "If the context doesn't contain the answer, say so plainly instead of guessing."
)


@generator_registry.register(
    "groq_chat",
    "Answer synthesis via Groq's OpenAI-compatible chat completions API (Llama 3.3 by "
    "default, very fast inference). Builds a numbered context block from the retrieved "
    "chunks and asks the model to answer grounded in that context, citing chunk numbers. "
    "Requires GROQ_API_KEY and the 'cloud' extra.",
)
class GroqChatGenerator(Generator):
    name = "groq_chat"

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_context_chunks: int = 8,
        temperature: float = 0.2,
    ):
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError(
                "groq package is not installed. Run: uv add --optional cloud groq"
            ) from e

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY in your environment or .env file."
            )

        self.model = model
        self.system_prompt = system_prompt
        self.max_context_chunks = max_context_chunks
        self.temperature = temperature
        self._client = Groq(api_key=key)

    def _build_context(self, results: list[RetrievalResult]) -> str:
        picked = results[: self.max_context_chunks]
        blocks = [f"[{i + 1}] {r.text}" for i, r in enumerate(picked)]
        return "\n\n".join(blocks)

    def generate(self, query: str, results: list[RetrievalResult]) -> str:
        context = self._build_context(results)
        user_prompt = (
            f"Context:\n{context}\n\nQuestion: {query}\n\n"
            "Answer the question using only the context above. Cite chunks as [n]."
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
