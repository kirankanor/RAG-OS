import sys
from pathlib import Path

import streamlit as st

from rag_os.chunking import chunker_registry
from rag_os.core.types import RunConfig
from rag_os.embedding import embedder_registry
from rag_os.parsing.base import DEFAULT_STRATEGY_BY_EXTENSION
from rag_os.pipeline import run_dataset_generation
from rag_os.retrieval import retriever_registry
from rag_os.storage.file_store import save_upload

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib import strategy_picker

st.set_page_config(page_title="Upload Documents · RAG-OS", page_icon="📤", layout="wide")
st.title("📤 Upload Documents & Create a Run")

st.markdown(
    "Upload one or more files, pick a strategy for each stage, then click **Run pipeline**. "
    "This parses, chunks, and embeds every file and builds a retriever - all saved as one "
    "named **run** you can revisit later."
)

run_name = st.text_input("Run name (optional, helps you tell runs apart later)", value="")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "html", "htm", "txt", "md"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.caption("Parser auto-detected per file:")
    for f in uploaded_files:
        ext = Path(f.name).suffix.lower()
        st.caption(f"• {f.name} → `{DEFAULT_STRATEGY_BY_EXTENSION.get(ext, 'unsupported')}`")


st.subheader("Your decisions")
col1, col2, col3 = st.columns(3)
with col1:
    chunker_name, chunker_params = strategy_picker("chunking", chunker_registry, "chunker")
with col2:
    embedder_name, embedder_params = strategy_picker("embedding", embedder_registry, "embedder")
with col3:
    retriever_name, retriever_params = strategy_picker("retrieval", retriever_registry, "retriever")

if chunker_name == "semantic":
    st.info("Semantic chunking uses the embedder above and costs extra embedding calls.", icon="ℹ️")

with st.expander("📋 Review your run before executing", expanded=True):
    st.json({
        "chunker": {"name": chunker_name, "params": chunker_params},
        "embedder": {"name": embedder_name, "params": embedder_params},
        "retriever": {"name": retriever_name, "params": retriever_params},
    })

st.divider()

if st.button("▶️ Run pipeline", type="primary", disabled=not uploaded_files):
    config = RunConfig(
        name=run_name,
        parser_name="auto_by_extension",
        parser_params={},
        chunker_name=chunker_name,
        chunker_params=chunker_params,
        embedder_name=embedder_name,
        embedder_params=embedder_params,
        retriever_name=retriever_name,
        retriever_params=retriever_params,
    )

    with st.spinner("Saving uploads..."):
        saved_paths = [
            save_upload(f.getvalue(), f.name, config.id) for f in uploaded_files
        ]

    try:
        with st.spinner("Parsing → chunking → embedding → indexing..."):
            run_row, documents, chunks, retriever = run_dataset_generation(saved_paths, config)
    except Exception as e:  # noqa: BLE001
        st.error(f"Pipeline failed: {e}")
    else:
        st.success(
            f"Run '{run_row.name or run_row.id}' created: "
            f"{len(documents)} document(s), {len(chunks)} chunk(s)."
        )
        st.info(
            "Head to the Parsing / Chunking / Embedding / Retrieval pages to inspect this "
            "run, or Reports to compare it against others.",
            icon="👉",
        )
