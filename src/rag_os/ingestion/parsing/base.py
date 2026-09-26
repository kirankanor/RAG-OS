from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from rag_os.core.registry import Registry
from rag_os.core.types import Document


class Parser(ABC):
    """Interface every parsing strategy must implement."""

    name: str = "base"

    @abstractmethod
    def parse(self, file_path: str | Path) -> Document:
        """Turn a file on disk into a Document (clean text + metadata)."""
        raise NotImplementedError

    @classmethod
    def supported_extensions(cls) -> tuple[str, ...]:
        """Used by the UI to only offer parsers that can handle the uploaded file type."""
        return ()


parser_registry: Registry[Parser] = Registry("parser")

DEFAULT_STRATEGY_BY_EXTENSION: dict[str, str] = {
    ".txt": "txt_plain", ".md": "txt_plain",
    ".pdf": "pdf_pymupdf",
    ".docx": "docx_python_docx",
    ".html": "html_bs4", ".htm": "html_bs4",
}

def parser_for_file(file_path: str | Path) -> Parser:
    ext = Path(file_path).suffix.lower()
    name = DEFAULT_STRATEGY_BY_EXTENSION.get(ext)
    if name is None:
        raise ValueError(f"No default parser configured for extension '{ext}'")
    return parser_registry.create(name)
