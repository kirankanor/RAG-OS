from __future__ import annotations

from pathlib import Path

from rag_os.core.types import Document
from rag_os.parsing.base import Parser, parser_registry


@parser_registry.register(
    "pdf_pypdf",
    "Extracts text from PDFs using pypdf. Pure-python, no extra system deps, "
    "sometimes slightly different (worse or better) extraction than PyMuPDF depending on the PDF.",
)
class PyPdfParser(Parser):
    name = "pdf_pypdf"

    @classmethod
    def supported_extensions(cls) -> tuple[str, ...]:
        return (".pdf",)

    def parse(self, file_path: str | Path) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("pypdf is not installed. Run: uv add pypdf") from e

        path = Path(file_path)
        reader = PdfReader(str(path))
        text_parts = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(text_parts)
        return Document(
            source_filename=path.name,
            text=text,
            parser_name=self.name,
            metadata={"num_pages": len(reader.pages), "num_chars": len(text)},
        )
