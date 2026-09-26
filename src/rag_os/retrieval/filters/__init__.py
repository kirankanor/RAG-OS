"""
Importing this package registers every filter step with `pipeline_step_registry`.
Filters drop candidates that don't match a condition (currently: chunk metadata).
"""
from rag_os.retrieval.filters import metadata_filter  # noqa: F401

__all__: list[str] = []
