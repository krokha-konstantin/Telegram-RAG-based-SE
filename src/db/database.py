import sqlite3
import numpy as np

from ..data_classes import *
from ..config import DB_PATH

columns  = [
    ("channel", "TEXT"),
    ("message_id", "INTEGER"),
    ("datetime", "TEXT"),
    ("language", "TEXT"),
    ("content", "TEXT"),
    ("link", "TEXT"),
    ("embedding", "BLOB"),
]

PRIMARY_KEY = ("channel", "message_id")


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)

    columns_sql = ",\n".join([f"{col} {dtype}" for col, dtype in columns])
    primary_key_sql = f"PRIMARY KEY ({', '.join(PRIMARY_KEY)})"
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS posts (
            {columns_sql},
            {primary_key_sql}
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_posts (
            channel_username TEXT,
            message_id INTEGER,
            PRIMARY KEY (channel_username, message_id)
        )
    """)

    conn.commit()
    return conn


def save_post(conn, channel: Channel, message: Message):
    conn.execute(
        f"""
        INSERT OR IGNORE INTO posts ({', '.join(col for col, _ in columns)})
        VALUES ({', '.join(['?'] * len(columns))})
        """,
        (
            channel.link,
            message.id,
            message.date.isoformat(),
            channel.language,
            message.text,
            message.link,
            np.array(message.embedding, dtype=np.float32).tobytes(),
        ),
    )


def mark_processed(conn, channel: Channel, message: Message):
    conn.execute(
        """
        INSERT OR IGNORE INTO processed_posts (channel_username, message_id)
        VALUES (?, ?)
        """,
        (channel.username, message.id),
    )


def post_exists(conn, channel: Channel, message: Message) -> bool:
    cursor = conn.execute(
        """
        SELECT 1
        FROM processed_posts
        WHERE channel_username = ? AND message_id = ?
        LIMIT 1
        """,
        (channel.username, message.id),
    )
    return cursor.fetchone() is not None


def get_posts(conn):
    cursor = conn.execute("SELECT content, link, embedding FROM posts")

    texts = []
    links = []
    embeddings = []
    for content, link, blob in cursor:
        embedding = np.frombuffer(blob, dtype=np.float32)
        texts.append(content)
        links.append(link)
        embeddings.append(embedding)
    texts = np.array(texts, dtype=np.str_)
    links = np.array(links, dtype=np.str_)
    embeddings = np.array(embeddings, dtype=np.float32)

    return texts, links, embeddings


def delete_old_posts(conn):
    cursor = conn.execute(
        """
        DELETE FROM posts
        WHERE datetime < datetime('now', '-1 month')
        """
    )
    conn.commit()
    return cursor.rowcount