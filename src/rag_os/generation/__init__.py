"""
Importing this package registers every answer-generation strategy with
`generator_registry`. This is the final stage after retrieval: query + retrieved
chunks -> synthesized answer. Not yet wired into pipeline/dataset_generation.py
or any UI page - construct via `generator_registry.create("groq_chat", ...)` and
call `.generate(query, results)` with the RetrievalResults from a run's
retriever/pipeline.
"""
from rag_os.generation import groq_chat  # noqa: F401
from rag_os.generation.base import Generator, generator_registry

__all__ = ["Generator", "generator_registry"]
