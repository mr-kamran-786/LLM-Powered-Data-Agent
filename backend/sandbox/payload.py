"""Serialise a DataFrame the same way the sandbox worker does (used for database query results)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

MAX_TABLE_ROWS = 50


def table_payload(obj: pd.DataFrame | pd.Series, title: str | None = None) -> dict[str, Any]:
    df = obj.to_frame() if isinstance(obj, pd.Series) else obj
    shown = df.head(MAX_TABLE_ROWS)
    if not isinstance(shown.index, pd.RangeIndex):
        shown = shown.reset_index()
    shown.columns = [str(c) for c in shown.columns]
    payload = json.loads(shown.to_json(orient="split", date_format="iso", default_handler=str))
    return {
        "title": title or "",
        "columns": payload["columns"],
        "rows": payload["data"],
        "shape": [int(df.shape[0]), len(payload["columns"])],  # columns as displayed (index included)
        "truncated": len(df) > MAX_TABLE_ROWS,
    }
