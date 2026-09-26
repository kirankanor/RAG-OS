"""
Importing this package registers every parsing strategy with `parser_registry`.
The pipeline/UI code should import `parser_registry` from here (or from
`rag_os.ingestion.parsing.base`) rather than importing individual strategy modules,
so adding a new strategy only means adding one file + a decorator.
"""
# Import strategies for their registration side-effects.
from rag_os.ingestion.parsing import (  # noqa: F401
    docx_parser,
    html_parser,
    pdf_pymupdf,
    pdf_pypdf,
    txt_parser,
)
from rag_os.ingestion.parsing.base import (
    DEFAULT_STRATEGY_BY_EXTENSION,
    Parser,
    parser_for_file,
    parser_registry,
)

__all__ = ["Parser", "parser_registry", "parser_for_file", "DEFAULT_STRATEGY_BY_EXTENSION"]
