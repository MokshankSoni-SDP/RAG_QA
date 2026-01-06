import sqlite3
from pathlib import Path
from typing import List, Tuple

DB_PATH = Path("data") / "documents.db"


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_name TEXT NOT NULL,
        chunk_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def insert_chunk(
    document_name: str,
    chunk_id: int,
    chunk_text: str,
    embedding: bytes
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO document_chunks (document_name, chunk_id, chunk_text, embedding)
    VALUES (?, ?, ?, ?)
    """, (document_name, chunk_id, chunk_text, embedding))

    conn.commit()
    conn.close()


def fetch_all_chunks() -> List[Tuple]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT document_name, chunk_id, chunk_text, embedding
    FROM document_chunks
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows
