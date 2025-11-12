import sqlite3
import json
import time
from pathlib import Path
from typing import Any, Optional


class SQLiteCache:
    def __init__(self, db_path: str = "./storage/cache.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()

    def _init(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[tuple[Any, float]]:
        cur = self.conn.cursor()
        cur.execute("SELECT value, updated_at FROM cache WHERE key=?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        value, updated_at = row
        return json.loads(value), updated_at

    def set(self, key: str, value: Any) -> None:
        cur = self.conn.cursor()
        now = time.time()
        cur.execute(
            """
            INSERT INTO cache(key, value, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              value=excluded.value,
              updated_at=excluded.updated_at
            """,
            (key, json.dumps(value, default=str), now),
        )
        self.conn.commit()

    def delete(self, key: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM cache WHERE key=?", (key,))
        self.conn.commit()
