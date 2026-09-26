from __future__ import annotations

import re

from rag_os.chunking.base import Chunker, chunker_registry
from rag_os.core.types import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@chunker_registry.register(
    "markdown_aware",
    "Splits on Markdown headings first (so each chunk stays under one heading's section), "
    "then sub-splits any section that's still too long using simple size-based cuts. "
    "Best for structured docs / READMEs / wikis converted to markdown.",
)
class MarkdownAwareChunker(Chunker):
    name = "markdown_aware"

    def __init__(self, max_chunk_size: int = 1200):
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        matches = list(_HEADING_RE.finditer(text))

        sections: list[tuple[int, int, str]] = []
        if not matches:
            sections.append((0, len(text), text))
        else:
            for idx, m in enumerate(matches):
                start = m.start()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                sections.append((start, end, text[start:end]))

        chunks: list[Chunk] = []
        position = 0
        for start, end, section_text in sections:
            if len(section_text) <= self.max_chunk_size:
                chunks.append(self._make_chunk(document, section_text, position, start, end))
                position += 1
                continue

            offset = 0
            while offset < len(section_text):
                sub = section_text[offset : offset + self.max_chunk_size]
                sub_start = start + offset
                sub_end = sub_start + len(sub)
                chunks.append(self._make_chunk(document, sub, position, sub_start, sub_end))
                position += 1
                offset += self.max_chunk_size
        return chunks
