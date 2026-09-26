from __future__ import annotations

from pathlib import Path

from rag_os.core.types import Document
from rag_os.parsing.base import Parser, parser_registry


@parser_registry.register(
    "pdf_pymupdf",
    "Extracts text from PDFs using PyMuPDF (fitz). Fast, preserves reading order well, "
    "good general-purpose default. Requires the 'local' extra.",
)
class PyMuPdfParser(Parser):
    name = "pdf_pymupdf"

    @classmethod
    def supported_extensions(cls) -> tuple[str, ...]:
        return (".pdf",)

    def parse(self, file_path: str | Path) -> Document:
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "pymupdf is not installed. Run: uv add --optional local pymupdf"
            ) from e

        path = Path(file_path)
        text_parts: list[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
            num_pages = doc.page_count

        text = "\n\n".join(text_parts)
        return Document(
            source_filename=path.name,
            text=text,
            parser_name=self.name,
            metadata={"num_pages": num_pages, "num_chars": len(text)},
        )
