"""One-off migration: adds reranker_name / reranker_params columns to the runs table.
Safe to run multiple times - skips columns that already exist."""
import sqlite3

from rag_os.config.settings import get_settings


def column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def main() -> None:
    settings = get_settings()
    conn = sqlite3.connect(settings.db_path)
    cursor = conn.cursor()

    if not column_exists(cursor, "runs", "reranker_name"):
        cursor.execute("ALTER TABLE runs ADD COLUMN reranker_name TEXT DEFAULT ''")
        print("Added reranker_name")
    else:
        print("reranker_name already exists, skipping")

    if not column_exists(cursor, "runs", "reranker_params"):
        cursor.execute("ALTER TABLE runs ADD COLUMN reranker_params TEXT DEFAULT '{}'")
        print("Added reranker_params")
    else:
        print("reranker_params already exists, skipping")

    conn.commit()
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    main()