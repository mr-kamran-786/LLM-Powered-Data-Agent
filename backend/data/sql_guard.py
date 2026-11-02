"""Read-only SQL enforcement (requirement F2, risk "agent modifies the database").

A parser check, not a regex: sqlglot parses the query and every statement must be a
pure SELECT (or UNION/CTE of SELECTs). This is the second layer; connections are also
opened read-only at the driver level.
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

_FORBIDDEN = tuple(
    getattr(exp, name)
    for name in ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "Merge", "Command", "Into",
                 "TruncateTable", "Pragma", "Set", "Use", "Attach", "Detach", "Copy", "Install", "Grant")
    if hasattr(exp, name)  # node names vary slightly between sqlglot versions
)
# Table functions that read files or other databases from inside a SELECT.
_FORBIDDEN_FUNCS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto", "read_text",
    "read_blob", "glob", "sqlite_scan", "postgres_scan", "httpfs", "load_extension",
    "pg_read_file", "pg_ls_dir", "lo_import", "dblink", "query_table",
}


_FILE_LIKE = re.compile(r"[/\\]|\.(csv|tsv|parquet|json|jsonl|txt|xlsx?|db|sqlite3?|duckdb)$", re.I)


class UnsafeSQLError(ValueError):
    """The query is not a read-only SELECT."""


def ensure_read_only(query: str, dialect: str | None = None) -> str:
    """Return the query unchanged if it is a read-only SELECT; raise UnsafeSQLError otherwise."""
    if not query or not query.strip():
        raise UnsafeSQLError("Empty SQL query.")
    try:
        statements = [s for s in sqlglot.parse(query, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise UnsafeSQLError(f"SQL could not be parsed: {str(exc).splitlines()[0]}") from exc
    if not statements:
        raise UnsafeSQLError("Empty SQL query.")
    if len(statements) > 1:
        raise UnsafeSQLError("Only one SQL statement per call is allowed.")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.With, exp.Subquery)):
        raise UnsafeSQLError(f"Only SELECT queries are allowed (got {stmt.key.upper()}).")
    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN):
            raise UnsafeSQLError(f"Statement contains a forbidden operation: {node.key.upper()}.")
        if isinstance(node, (exp.Anonymous, exp.Func)):
            fname = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if fname in _FORBIDDEN_FUNCS:
                raise UnsafeSQLError(f"Function {fname}() is not allowed; query the registered tables instead.")
        if isinstance(node, exp.Table) and (
            isinstance(node.this, exp.Literal) or _FILE_LIKE.search(node.name or "")
        ):
            raise UnsafeSQLError("Reading files by path is not allowed; query the registered tables instead.")
    return query
