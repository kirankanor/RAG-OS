import streamlit as st

st.set_page_config(page_title="RAG-OS", page_icon="🧪", layout="wide")

st.title("🧪 RAG-OS")
st.caption("A modular monolith for experimenting with RAG pipeline strategies.")

st.markdown(
    """
RAG-OS lets you mix and match strategies at each stage of a RAG pipeline and compare
the results side by side.

**How to use it:**

1. **Upload Documents** — upload PDFs / Word docs / HTML / txt files and pick a parser,
   chunker, embedder, and retriever. This creates a "run".
2. **Parsing** — inspect the raw parsed text for any run.
3. **Chunking** — inspect the resulting chunks (size, boundaries, overlap) for any run.
4. **Embedding** — inspect embedding stats (dimension, sample vectors) for any run.
5. **Retrieval** — type a query, see the top-k results for any run, and rate them.
6. **Reports** — compare runs side by side, manually and (once you add ground-truth
   query/answer pairs) with automatic recall@k / precision@k / MRR metrics.

Use the sidebar to move between pages. Every run you create is saved to a local
SQLite database (`data/rag_os.sqlite3`), so you can close the app and come back later
without losing anything.
"""
)

st.info(
    "New here? Start on the **Upload Documents** page in the sidebar to create your "
    "first run.",
    icon="👈",
)
