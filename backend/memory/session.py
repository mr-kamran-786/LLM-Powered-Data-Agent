"""Runtime session: datasets, database connections and the sandbox of one conversation."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import get_settings
from backend.data import ingestion, profiler
from backend.data.connectors import DatabaseConnection
from backend.memory.store import store
from backend.sandbox.manager import Sandbox, pool


@dataclass
class Dataset:
    name: str
    source: str                  # "file" or "db:<connection>"
    parquet_path: Path | None    # files only
    original_filename: str | None
    rows: int
    columns: int
    profile_text: str
    profile: dict[str, Any] = field(default_factory=dict)

    def public_info(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source, "filename": self.original_filename,
                "rows": self.rows, "columns": self.columns}


class Session:
    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self.dir = get_settings().upload_dir / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.datasets: dict[str, Dataset] = {}
        self.connections: dict[str, DatabaseConnection] = {}
        self.lock = asyncio.Lock()  # one agent turn at a time per session

    @property
    def sandbox(self) -> Sandbox:
        return pool.get(self.id, self.dir)

    # ------------------------------------------------------------ files

    def add_file(self, raw_path: Path, original_name: str) -> Dataset:
        df = ingestion.load_dataframe(raw_path)
        name = ingestion.table_name_from_filename(original_name, taken=set(self.datasets))
        parquet = ingestion.save_as_parquet(df, self.dir / "data" / f"{name}.parquet")
        ds = Dataset(
            name=name, source="file", parquet_path=parquet, original_filename=original_name,
            rows=len(df), columns=df.shape[1], profile_text=profiler.profile(df, name),
            profile=profiler.profile_dict(df, name),
        )
        self.datasets[name] = ds
        reply = self.sandbox.load_dataset(name, parquet)
        if not reply.get("ok"):
            raise RuntimeError(f"Sandbox could not load {name}: {reply.get('error')}")
        store.save_dataset(self.id, ds)
        return ds

    def frame(self, name: str, limit: int | None = None) -> pd.DataFrame:
        ds = self.datasets[name]
        if ds.parquet_path is not None:
            df = pd.read_parquet(ds.parquet_path)
            return df.head(limit) if limit else df
        conn = self.connections[ds.source.split(":", 1)[1]]
        return conn.table_frame(name, limit=limit)

    # ------------------------------------------------------------ databases

    def add_connection(self, name: str, kind: str, target: str, profile_rows: int = 5000) -> DatabaseConnection:
        conn = DatabaseConnection(name=name, kind=kind, target=target)
        tables = conn.test()
        self.connections[name] = conn
        for table in tables[:50]:  # profile from a sample; large schemas are retrieved per question
            df = conn.table_frame(table, limit=profile_rows)
            total = conn.row_count(table)
            text = profiler.profile(df, table)
            if total > len(df):
                text = text.replace(f"{len(df):,} rows", f"{total:,} rows (profiled on the first {len(df):,})", 1)
            text = f"[database `{name}` ({kind}); query with run_sql(target='{name}')]\n" + text
            self.datasets[table] = Dataset(
                name=table, source=f"db:{name}", parquet_path=None, original_filename=None,
                rows=total, columns=df.shape[1], profile_text=text, profile=profiler.profile_dict(df, table),
            )
        store.save_connection(self.id, conn)
        return conn

    # ------------------------------------------------------------ restore after server restart

    def restore(self) -> None:
        for row in store.load_datasets(self.id):
            path = Path(row["parquet_path"]) if row["parquet_path"] else None
            if path is None or not path.exists():
                continue
            ds = Dataset(name=row["name"], source=row["source"], parquet_path=path,
                         original_filename=row["filename"], rows=row["rows"], columns=row["columns"],
                         profile_text=row["profile_text"], profile=row["profile"])
            self.datasets[ds.name] = ds
            self.sandbox.datasets[ds.name] = path  # loaded when the worker starts
        for row in store.load_connections(self.id):
            try:
                self.add_connection(row["name"], row["kind"], row["target"])
            except Exception:  # database moved or offline; the user can reconnect
                continue

    def schema_text(self, question: str | None = None, max_tables: int = 8) -> str:
        """Profiles for the prompt. With many tables, only the ones relevant to the question."""
        items = list(self.datasets.values())
        if question and len(items) > max_tables:
            items = rank_tables(question, items)[:max_tables]
        if not items:
            return "No datasets loaded yet."
        return "\n\n".join(d.profile_text for d in items)


def rank_tables(question: str, tables: list[Dataset]) -> list[Dataset]:
    """Schema retrieval for large databases: rank tables by word overlap with the question.

    Lexical scoring keeps the system dependency-free; an embedding index (Chroma/pgvector)
    can replace it behind the same function.
    """
    import re

    words = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2}

    def score(ds: Dataset) -> float:
        vocab = set(re.findall(r"[a-z0-9]+", (ds.name + " " + " ".join(
            c["name"] for c in ds.profile.get("column_profiles", []))).lower()))
        stems = {v.rstrip("s") for v in vocab}
        return sum(1.0 for w in words if w in vocab or w.rstrip("s") in stems)

    return sorted(tables, key=score, reverse=True)


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Session:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                if not store.session_exists(session_id):
                    raise KeyError(session_id)
                sess = Session(session_id)
                sess.restore()
                self._sessions[session_id] = sess
            return sess

    def create(self, title: str = "New analysis") -> Session:
        session_id = store.create_session(title)
        sess = Session(session_id)
        with self._lock:
            self._sessions[session_id] = sess
        return sess

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
        pool.drop(session_id)
        store.delete_session(session_id)


registry = SessionRegistry()
