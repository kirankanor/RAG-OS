"""
Importing this package registers every chunking strategy with `chunker_registry`.
"""
from rag_os.chunking import (  # noqa: F401
    fixed_size,
    markdown_aware,
    recursive_char,
    semantic,
    sentence_window,
)
from rag_os.chunking.base import Chunker, chunker_registry

__all__ = ["Chunker", "chunker_registry"]
