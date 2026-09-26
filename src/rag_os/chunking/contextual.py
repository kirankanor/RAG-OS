from __future__ import annotations

from collections.abc import Callable

from rag_os.chunking.base import Chunker, chunker_registry
from rag_os.chunking.recursive_char import RecursiveCharacterChunker
from rag_os.core.types import Chunk, Document

LlmFn = Callable[[str], str]  # prompt -> completion

_PROMPT_TEMPLATE = (
    "Document:\n{doc}\n\nHere is a chunk from this document:\n{chunk}\n\n"
    "Give a short (1-2 sentence) context describing where this chunk fits in the "
    "overall document, to improve search retrieval. Answer only with the context."
)


@chunker_registry.register(
    "contextual",
    "Wraps recursive_char chunking, then prepends an LLM-generated 1-2 sentence "
    "summary of each chunk's surrounding context before embedding (Anthropic's "
    "'contextual retrieval'). One extra LLM call per chunk at generation time - "
    "slower and costs tokens, but reduces retrieval misses on chunks that lose "
    "meaning out of context. NOTE: pass llm_fn explicitly, same pattern as "
    "semantic.py's embed_fn; chunking module deliberately doesn't import an LLM client.",
)
class ContextualChunker(Chunker):
    name = "contextual"

    def __init__(
        self,
        llm_fn: LlmFn | None = None,
        chunk_size: int = 1000,
        overlap: int = 150,
    ):
        self.llm_fn = llm_fn
        self._base = RecursiveCharacterChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, document: Document) -> list[Chunk]:
        if self.llm_fn is None:
            raise ValueError(
                "ContextualChunker requires an llm_fn. Construct it as "
                "ContextualChunker(llm_fn=your_llm_call) from the pipeline layer."
            )

        base_chunks = self._base.chunk(document)
        for c in base_chunks:
            prompt = _PROMPT_TEMPLATE.format(doc=document.text[:8000], chunk=c.text)
            context = self.llm_fn(prompt).strip()
            c.text = f"{context}\n\n{c.text}"
            c.chunker_name = self.name
            c.metadata["prepended_context"] = context
        return base_chunks