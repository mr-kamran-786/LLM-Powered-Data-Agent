"""Read-only database connectors: SQLite, DuckDB and PostgreSQL (requirement F2).

Three layers keep the user's database safe:
1. the driver connection itself is read-only (SQLite mode=ro, DuckDB read_only,
   PostgreSQL default_transaction_read_only);
2. every query passes `ensure_read_only` (sqlglot parser);
3. results are capped at `max_rows`.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from backend.data.sql_guard import ensure_read_only

SUPPORTED_KINDS = ("sqlite", "duckdb", "postgres")


@dataclass
class DatabaseConnection:
    name: str
    kind: str              # sqlite | duckdb | postgres
    target: str            # file path for sqlite/duckdb, DSN for postgres
    tables: list[str] = field(default_factory=list)

    @property
    def dialect(self) -> str:
        return {"sqlite": "sqlite", "duckdb": "duckdb", "postgres": "postgres"}[self.kind]

    # ---------- connection ----------

    def _connect(self) -> Any:
        if self.kind == "sqlite":
            uri = Path(self.target).resolve().as_uri() + "?mode=ro"
            return sqlite3.connect(uri, uri=True, check_same_thread=False)
        if self.kind == "duckdb":
            return duckdb.connect(self.target, read_only=True)
        if self.kind == "postgres":
            import psycopg  # optional dependency, imported lazily

            conn = psycopg.connect(self.target, options="-c default_transaction_read_only=on", connect_timeout=10)
            conn.read_only = True
            return conn
        raise ValueError(f"Unknown database kind '{self.kind}'. Supported: {', '.join(SUPPORTED_KINDS)}")

    def test(self) -> list[str]:
        """Connect, list tables, and remember them."""
        self.tables = self.list_tables()
        return self.tables

    def list_tables(self) -> list[str]:
        if self.kind == "sqlite":
            q = "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
        else:
            q = ("SELECT table_name FROM information_schema.tables "
                 "WHERE table_schema NOT IN ('information_schema','pg_catalog') ORDER BY table_name")
        return [str(r) for r in self._raw_query(q)["__col0__" if False else 0].tolist()] if False else \
            self._raw_query(q).iloc[:, 0].astype(str).tolist()

    def table_frame(self, table: str, limit: int | None = None) -> pd.DataFrame:
        """Fetch a table (or its first rows) for profiling."""
        if table not in (self.tables or self.list_tables()):
            raise ValueError(f"Unknown table '{table}'.")
        quoted = '"' + table.replace('"', '""') + '"'
        q = f"SELECT * FROM {quoted}" + (f" LIMIT {int(limit)}" if limit else "")
        return self._raw_query(q)

    def row_count(self, table: str) -> int:
        quoted = '"' + table.replace('"', '""') + '"'
        return int(self._raw_query(f"SELECT COUNT(*) FROM {quoted}").iloc[0, 0])

    # ---------- queries ----------

    def query(self, sql: str, max_rows: int = 100_000) -> pd.DataFrame:
        """Run a user/agent query after the read-only check."""
        ensure_read_only(sql, dialect=self.dialect)
        df = self._raw_query(sql)
        return df.head(max_rows)

    def _raw_query(self, sql: str) -> pd.DataFrame:
        conn = self._connect()
        try:
            if self.kind == "duckdb":
                return conn.execute(sql).df()
            cur = conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=cols)
        finally:
            conn.close()

    def public_info(self) -> dict[str, Any]:
        """What the API returns; never includes passwords from the DSN."""
        return {"name": self.name, "kind": self.kind, "target": _redact(self.target), "tables": self.tables}


def _redact(target: str) -> str:
    if "://" in target and "@" in target:
        scheme, rest = target.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return target
