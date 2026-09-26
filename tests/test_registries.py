"""Sanity tests: every strategy module registers itself correctly and is constructible
with default params where that's possible without network/API access."""
from rag_os.chunking import chunker_registry
from rag_os.core.types import Document
from rag_os.embedding import embedder_registry
from rag_os.parsing import parser_registry
from rag_os.retrieval import retriever_registry


def test_parser_registry_has_expected_strategies():
    names = parser_registry.names()
    for expected in ["txt_plain", "pdf_pymupdf", "pdf_pypdf", "docx_python_docx", "html_bs4"]:
        assert expected in names


def test_chunker_registry_has_expected_strategies():
    names = chunker_registry.names()
    for expected in ["fixed_size", "recursive_char", "sentence_window", "markdown_aware", "semantic" , "code_aware"]:
        assert expected in names


def test_embedder_registry_has_expected_strategies():
    names = embedder_registry.names()
    for expected in ["local_minilm", "openai_text_embedding_3_small", "cohere_embed_v3"]:
        assert expected in names


def test_retriever_registry_has_expected_strategies():
    names = retriever_registry.names()
    for expected in ["faiss_flat_l2", "qdrant", "hybrid_bm25_vector"]:
        assert expected in names


def test_fixed_size_chunker_produces_chunks():
    chunker = chunker_registry.create("fixed_size", chunk_size=20, overlap=5)
    doc = Document(text="a" * 100)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert all(c.document_id == doc.id for c in chunks)


def test_recursive_char_chunker_respects_paragraphs():
    chunker = chunker_registry.create("recursive_char", chunk_size=50, overlap=0)
    doc = Document(text="First paragraph.\n\nSecond paragraph that is a bit longer than the first one.")
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
