"""Verifier: rule-based sanity checks on every executed step (plan, section 7).

The optional LLM critique lives in the orchestrator; these checks are deterministic,
free and run on every step. They return warnings, which go back to the Coder.
"""

from __future__ import annotations

import math
import re
from typing import Any

_PCT_COL = re.compile(r"(pct|percent|percentage|share|_rate$|^rate)", re.I)
_NAN_TOTAL = re.compile(r"(total|sum|mean|average|avg)[^\n]{0,40}\bnan\b", re.I)
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?%?")


def check_step(step: dict[str, Any], source_rows: int | None = None) -> list[str]:
    """Return human-readable warnings for a successful step (empty list = looks fine)."""
    if not step.get("ok"):
        return []
    notes: list[str] = []
    tables = step.get("tables") or []
    if step.get("type") in ("python", "sql") and not (tables or step.get("stdout") or step.get("value") or step.get("charts")):
        if not step.get("new_vars"):
            notes.append("The step produced no visible output. print() the result or end the cell with it.")
    for t in tables:
        rows = t.get("shape", [len(t.get("rows", []))])[0]
        name = f"table {t['title']!r}" if t.get("title") else "the result table"
        if rows == 0:
            notes.append(f"{name} is EMPTY. Check filter values (exact spelling, case, dtype, date range).")
            continue
        cols = t.get("columns", [])
        data = t.get("rows", [])
        for j, col in enumerate(cols):
            values = [r[j] for r in data if j < len(r)]
            if values and all(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
                notes.append(f"Column {col!r} in {name} is entirely NaN/None.")
            nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)]
            if nums and _PCT_COL.search(str(col)):
                if max(nums) > 100.0001:
                    notes.append(f"Column {col!r} looks like a percentage but has values over 100 (max {max(nums):.4g}).")
                if min(nums) < -100.0001 and "change" not in str(col).lower() and "growth" not in str(col).lower():
                    notes.append(f"Column {col!r} looks like a percentage but has values below -100.")
        if source_rows and rows > source_rows * 1.5 and rows > 1000:
            notes.append(f"{name} has {rows:,} rows, more than the source data ({source_rows:,}). "
                         "A join on non-unique keys may have duplicated rows.")
    if _NAN_TOTAL.search(step.get("stdout") or ""):
        notes.append("A printed total/mean is NaN. Handle missing values (dropna / fillna) or check the column dtype.")
    return notes


# ---------------------------------------------------------------- grounding of the final answer


def _numbers(text: str) -> list[tuple[float, int, bool]]:
    """Numbers in text as (value, decimals, is_percent)."""
    out = []
    for raw in _NUM.findall(text or ""):
        is_pct = raw.endswith("%")
        clean = raw.rstrip("%").replace(",", "")
        try:
            value = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean and "e" not in clean.lower() else 0
        out.append((value, decimals, is_pct))
    return out


def evidence_text(steps: list[dict[str, Any]]) -> str:
    parts = []
    for s in steps:
        if not s.get("ok"):
            continue
        parts.append(s.get("stdout") or "")
        parts.append(s.get("value") or "")
        for t in s.get("tables", []):
            parts.append(" ".join(str(v) for row in t.get("rows", []) for v in row))
            parts.append(str(t.get("shape", "")))
    return "\n".join(parts)


def ungrounded_numbers(answer_text: str, steps: list[dict[str, Any]]) -> list[str]:
    """Numbers in the answer that do not appear (within rounding) in any executed output."""
    evidence = [v for v, _, _ in _numbers(evidence_text(steps))]
    if not evidence:
        return []
    missing = []
    for value, decimals, is_pct in _numbers(answer_text):
        if value == 0 or (float(value).is_integer() and abs(value) <= 31) or 1900 <= value <= 2100:
            continue  # small counts, days and years are not worth flagging
        candidates = [value] + ([value / 100] if is_pct else []) + ([value / 100, value * 100] if not is_pct else [])
        tol = 0.5 * 10 ** (-decimals) if decimals else 0.5
        found = False
        for e in evidence:
            for c in candidates:
                scale = 10 ** decimals
                if abs(e - c) <= max(tol, abs(c) * 0.005) or round(e * scale) == round(c * scale):
                    found = True
                    break
                # thousands abbreviations: 1.2M / 3.4k written from 1234567
                if decimals and abs(c) < 1000 and any(abs(e / m - c) <= tol for m in (1e3, 1e6, 1e9)):
                    found = True
                    break
            if found:
                break
        if not found:
            missing.append(f"{value:g}{'%' if is_pct else ''}")
    return missing
