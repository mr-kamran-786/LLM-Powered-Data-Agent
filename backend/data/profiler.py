"""Data profiler: the agent's "eyes" (requirement F3) and cleaning advisor (F10).

The LLM never sees the whole dataset, only the compact text profile built here.
The same profile, as a dict, feeds the data preview panel in the UI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MAX_CELL_CHARS = 60  # long text cells are cut so data cannot smuggle long instructions into prompts


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[: MAX_CELL_CHARS - 1] + "…"


def _fmt_num(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    try:
        return f"{float(x):.4g}" if abs(float(x)) < 1e4 else f"{float(x):,.2f}"
    except (TypeError, ValueError):
        return str(x)


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "categorical" if s.nunique(dropna=True) <= max(20, len(s) * 0.05) else "text"


def profile_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Structured per-column statistics."""
    out = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        kind = column_kind(s)
        nunique = int(s.nunique(dropna=True))
        info: dict[str, Any] = {
            "name": str(col),
            "dtype": str(s.dtype),
            "kind": kind,
            "null_pct": round(float(s.isna().mean() * 100), 2) if n else 0.0,
            "unique": nunique,
            "is_candidate_key": n > 0 and nunique == n and s.notna().all(),
        }
        if kind == "numeric":
            desc = s.describe()
            info.update(
                min=_py(desc.get("min")), max=_py(desc.get("max")), mean=_py(desc.get("mean")),
                std=_py(desc.get("std")), median=_py(s.median()),
            )
            counts, edges = np.histogram(s.dropna(), bins=min(20, max(nunique, 1))) if s.notna().any() else ([], [])
            info["histogram"] = {"counts": [int(c) for c in counts], "edges": [float(e) for e in edges]}
        elif kind == "datetime":
            info.update(min=str(s.min()), max=str(s.max()))
        else:
            top = s.astype("string").value_counts().head(5)
            info["top"] = [{"value": _short(k), "count": int(v)} for k, v in top.items()]
        out.append(info)
    return out


def _py(x: Any) -> Any:
    """numpy scalar -> plain Python, NaN -> None (JSON-safe)."""
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def profile(df: pd.DataFrame, name: str, n_samples: int = 5) -> str:
    """Compact text profile for the LLM prompt (format from the implementation plan)."""
    lines = [f"Table `{name}`: {len(df):,} rows x {df.shape[1]} columns"]
    for c in profile_columns(df):
        info = f"- {c['name']} ({c['dtype']}), nulls={c['null_pct']:.0f}%, unique={c['unique']}"
        if c["kind"] == "numeric":
            info += f", min={_fmt_num(c['min'])}, max={_fmt_num(c['max'])}, mean={_fmt_num(c['mean'])}"
        elif c["kind"] == "datetime":
            info += f", range={c['min']} .. {c['max']}"
        else:
            info += f", top={[t['value'] for t in c['top'][:3]]}"
        if c["is_candidate_key"]:
            info += ", KEY"
        lines.append(info)
    sample = df.head(n_samples).copy()
    for col in sample.columns:
        if not pd.api.types.is_numeric_dtype(sample[col]):
            sample[col] = sample[col].map(lambda v: _short(v) if v is not None else v)
    lines.append("Sample rows:\n" + sample.to_markdown(index=False))
    return "\n".join(lines)


def profile_dict(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """Profile for the UI, including cleaning suggestions."""
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_profiles": profile_columns(df),
        "cleaning_suggestions": cleaning_suggestions(df),
    }


def cleaning_suggestions(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rule-based data-cleaning suggestions (requirement F10). Each comes with pandas code."""
    tips: list[dict[str, Any]] = []
    dups = int(df.duplicated().sum())
    if dups:
        tips.append({
            "type": "duplicates", "column": None, "severity": "medium",
            "message": f"{dups:,} fully duplicated rows ({dups / len(df):.1%}).",
            "code": "df = df.drop_duplicates()",
        })
    for col in df.columns:
        s = df[col]
        null_pct = s.isna().mean()
        if null_pct >= 0.5:
            tips.append({
                "type": "missing", "column": col, "severity": "high",
                "message": f"`{col}` is {null_pct:.0%} empty; consider dropping it.",
                "code": f"df = df.drop(columns=[{col!r}])",
            })
        elif null_pct > 0:
            fill = "median()" if pd.api.types.is_numeric_dtype(s) else "mode()[0]"
            tips.append({
                "type": "missing", "column": col, "severity": "low",
                "message": f"`{col}` has {null_pct:.1%} missing values.",
                "code": f"df[{col!r}] = df[{col!r}].fillna(df[{col!r}].{fill})",
            })
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.notna().sum() > 10:
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                n_out = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
                if n_out:
                    tips.append({
                        "type": "outliers", "column": col, "severity": "medium",
                        "message": f"`{col}` has {n_out:,} extreme outliers (beyond 3 x IQR).",
                        "code": (f"q1, q3 = df[{col!r}].quantile([0.25, 0.75]); iqr = q3 - q1\n"
                                 f"df = df[df[{col!r}].between(q1 - 3 * iqr, q3 + 3 * iqr) | df[{col!r}].isna()]"),
                    })
            if (s < 0).any() and any(k in str(col).lower() for k in ("price", "qty", "quantity", "amount", "age", "count", "units")):
                tips.append({
                    "type": "invalid", "column": col, "severity": "medium",
                    "message": f"`{col}` contains negative values, which look invalid for this field.",
                    "code": f"df = df[df[{col!r}] >= 0]",
                })
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            non_null = s.dropna().astype(str)
            if len(non_null):
                numeric_share = pd.to_numeric(non_null.str.replace(r"[,$€£%\s]", "", regex=True), errors="coerce").notna().mean()
                if numeric_share > 0.9:
                    tips.append({
                        "type": "type_fix", "column": col, "severity": "medium",
                        "message": f"`{col}` is stored as text but {numeric_share:.0%} of values are numbers.",
                        "code": f"df[{col!r}] = pd.to_numeric(df[{col!r}].astype(str).str.replace(r'[,$€£%\\s]', '', regex=True), errors='coerce')",
                    })
                stripped = non_null.str.strip().str.lower()
                if stripped.nunique() < non_null.nunique():
                    tips.append({
                        "type": "inconsistent_text", "column": col, "severity": "low",
                        "message": f"`{col}` has values that differ only in case or spacing "
                                   f"({non_null.nunique()} raw vs {stripped.nunique()} normalised).",
                        "code": f"df[{col!r}] = df[{col!r}].str.strip().str.title()",
                    })
    return tips
