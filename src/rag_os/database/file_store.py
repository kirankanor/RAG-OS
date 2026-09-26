"""
Plain-file storage for the things that don't belong in SQLite rows: raw uploaded
files, and (optionally) large numpy vector arrays if you'd rather not stuff them
into the DB as JSON for very large runs.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from rag_os.config.settings import get_settings


def save_upload(file_bytes: bytes, filename: str, run_id: str) -> Path:
    settings = get_settings()
    dest_dir = settings.uploads_dir / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_bytes(file_bytes)
    return dest_path


def uploads_for_run(run_id: str) -> list[Path]:
    settings = get_settings()
    run_dir = settings.uploads_dir / run_id
    if not run_dir.exists():
        return []
    return sorted(p for p in run_dir.iterdir() if p.is_file())


def clear_run_uploads(run_id: str) -> None:
    settings = get_settings()
    run_dir = settings.uploads_dir / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
