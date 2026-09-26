"""
Importing this package registers every fuser step with `pipeline_step_registry`.
Fusers merge candidates that came from 2+ generator steps into one deduped, scored
list. Only needed when a pipeline uses more than one generator.
"""
from rag_os.retrieval.fusers import fuse  # noqa: F401

__all__: list[str] = []
