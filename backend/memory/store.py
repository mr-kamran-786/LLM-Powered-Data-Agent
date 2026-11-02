"""Persistent session store (SQLite): sessions, datasets, connections and every turn.

Every answer is stored with its plan, the executed code, intermediate tables and charts,
so the UI can always show the work behind a number (requirement F7) and follow-up
questions can build on earlier turns (F8).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from backend.config import get_settings

if TYPE_CHECKING:
    from backend.data.connectors import DatabaseConnection
    from backend.memory.session import Dataset

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, title TEXT, created_at REAL, updated_at REAL,
    tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0,
    summary TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS datasets (
    session_id TEXT, name TEXT, source TEXT, parquet_path TEXT, filename TEXT,
    rows INTEGER, columns INTEGER, profile_text TEXT, profile_json TEXT,
    PRIMARY KEY (session_id, name)
);
CREATE TABLE IF NOT EXISTS connections (
    session_id TEXT, name TEXT, kind TEXT, target TEXT, PRIMARY KEY (session_id, name)
);
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY, session_id TEXT, idx INTEGER, created_at REAL, model TEXT,
    question TEXT, status TEXT, answer_md TEXT, short_answer TEXT,
    plan_json TEXT, steps_json TEXT, usage_json TEXT
);
CREATE INDEX IF NOT EXISTS turns_by_session ON turns(session_id, idx);
"""


class Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            path = get_settings().db_path
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)
        return self._conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            with self.conn:
                yield self.conn

    # ------------------------------------------------------------ sessions

    def create_session(self, title: str) -> str:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        with self.tx() as c:
            c.execute("INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)", (sid, title, now, now))
        return sid

    def session_exists(self, sid: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone() is not None

    def get_session(self, sid: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM turns t WHERE t.session_id=s.id) AS turn_count "
            "FROM sessions s ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def rename_session(self, sid: str, title: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE sessions SET title=? WHERE id=?", (title[:120], sid))

    def set_summary(self, sid: str, summary: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE sessions SET summary=? WHERE id=?", (summary, sid))

    def add_usage(self, sid: str, tokens_in: int, tokens_out: int, cost: float) -> None:
        with self.tx() as c:
            c.execute("UPDATE sessions SET tokens_in=tokens_in+?, tokens_out=tokens_out+?, cost_usd=cost_usd+?, "
                      "updated_at=? WHERE id=?", (tokens_in, tokens_out, cost, time.time(), sid))

    def delete_session(self, sid: str) -> None:
        with self.tx() as c:
            for table in ("turns", "datasets", "connections"):
                c.execute(f"DELETE FROM {table} WHERE session_id=?", (sid,))
            c.execute("DELETE FROM sessions WHERE id=?", (sid,))

    # ------------------------------------------------------------ datasets & connections

    def save_dataset(self, sid: str, ds: "Dataset") -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO datasets VALUES (?,?,?,?,?,?,?,?,?)", (
                sid, ds.name, ds.source, str(ds.parquet_path) if ds.parquet_path else None,
                ds.original_filename, ds.rows, ds.columns, ds.profile_text, json.dumps(ds.profile, default=str)))

    def load_datasets(self, sid: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM datasets WHERE session_id=?", (sid,)).fetchall()
        return [{**dict(r), "profile": json.loads(r["profile_json"] or "{}")} for r in rows]

    def save_connection(self, sid: str, conn: "DatabaseConnection") -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO connections VALUES (?,?,?,?)", (sid, conn.name, conn.kind, conn.target))

    def load_connections(self, sid: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM connections WHERE session_id=?", (sid,)).fetchall()]

    # ------------------------------------------------------------ turns

    def save_turn(self, sid: str, turn: dict[str, Any]) -> str:
        tid = turn.get("id") or uuid.uuid4().hex[:12]
        idx = self.conn.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM turns WHERE session_id=?", (sid,)).fetchone()[0]
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                tid, sid, idx, time.time(), turn.get("model"), turn.get("question"), turn.get("status"),
                turn.get("answer_md"), turn.get("short_answer"), json.dumps(turn.get("plan")),
                json.dumps(turn.get("steps", []), default=str), json.dumps(turn.get("usage", {}))))
            c.execute("UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), sid))
        return tid

    def list_turns(self, sid: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM turns WHERE session_id=? ORDER BY idx", (sid,)).fetchall()
        return [_turn(r) for r in rows]


def _turn(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["plan"] = json.loads(d.pop("plan_json") or "null")
    d["steps"] = json.loads(d.pop("steps_json") or "[]")
    d["usage"] = json.loads(d.pop("usage_json") or "{}")
    return d


store = Store()
