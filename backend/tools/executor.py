"""Executes tool calls and turns their results into (a) a structured step for the UI
and (b) a compact text observation for the LLM."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pandas as pd

from backend.config import get_settings
from backend.data.sql_guard import UnsafeSQLError, ensure_read_only
from backend.memory.session import Session
from backend.sandbox.guard import check_code
from backend.sandbox.payload import table_payload

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")


def _empty_step(kind: str, purpose: str, code: str) -> dict[str, Any]:
    return {"type": kind, "purpose": purpose, "code": code, "ok": False, "stdout": "", "error": None,
            "error_type": None, "value": None, "tables": [], "charts": [], "new_vars": [], "duration_s": None}


async def run_python(session: Session, code: str, purpose: str = "") -> dict[str, Any]:
    step = _empty_step("python", purpose, code)
    verdict = check_code(code)
    if not verdict.ok:
        step.update(error=verdict.reason, error_type="BlockedBySandboxPolicy")
        return step
    result = await session.sandbox.aexecute(code)
    step.update({k: result.get(k, step[k]) for k in step if k in result})
    step["ok"] = bool(result.get("ok"))
    if result.get("restarted"):
        step["kernel_restarted"] = True
    return step


async def run_sql(session: Session, query: str, target: str, result_name: str, purpose: str = "") -> dict[str, Any]:
    step = _empty_step("sql", purpose, query)
    step["target"] = target
    name = result_name if result_name and _IDENT.match(result_name) else "sql_result"
    step["result_name"] = name
    if target in ("files", "", None):
        try:
            ensure_read_only(query, dialect="duckdb")
        except UnsafeSQLError as exc:
            step.update(error=str(exc), error_type="UnsafeSQL")
            return step
        code = f"{name} = sql({query!r})\n{name}"
        result = await session.sandbox.aexecute(code)
        step.update({k: result.get(k, step[k]) for k in ("stdout", "error", "error_type", "tables", "duration_s")})
        step["ok"] = bool(result.get("ok"))
        step["new_vars"] = [name] if step["ok"] else []
        return step

    conn = session.connections.get(target)
    if conn is None:
        step.update(error=f"Unknown target '{target}'. Use 'files' or one of: {list(session.connections)}",
                    error_type="UnknownTarget")
        return step
    try:
        df = await asyncio.to_thread(conn.query, query)
    except UnsafeSQLError as exc:
        step.update(error=str(exc), error_type="UnsafeSQL")
        return step
    except Exception as exc:  # driver errors are returned to the LLM to fix the query
        step.update(error=f"{type(exc).__name__}: {exc}", error_type=type(exc).__name__)
        return step
    # Hand the result to the Python sandbox so later steps can use it.
    path = session.dir / "data" / f"_q_{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(df.to_parquet, path, index=False)
    await asyncio.to_thread(session.sandbox.request, {"cmd": "load", "name": name, "path": str(path.resolve())}, 60)
    step.update(ok=True, tables=[table_payload(df, name)], new_vars=[name])
    return step


def describe_table(session: Session, name: str) -> dict[str, Any]:
    step = _empty_step("describe", f"Profile of {name}", name)
    ds = session.datasets.get(name)
    if ds is None:
        step.update(error=f"Unknown table '{name}'. Available: {list(session.datasets)}", error_type="UnknownTable")
        return step
    step.update(ok=True, stdout=ds.profile_text)
    return step


# ---------------------------------------------------------------- observation for the LLM


def _table_markdown(table: dict[str, Any], max_rows: int = 15) -> str:
    df = pd.DataFrame(table["rows"][:max_rows], columns=table["columns"])
    shape = table.get("shape", [len(table["rows"]), len(table["columns"])])
    head = f"Table{(' ' + repr(table['title'])) if table.get('title') else ''} shape={shape[0]}x{shape[1]}"
    more = f"\n... ({shape[0] - max_rows} more rows not shown)" if shape[0] > max_rows else ""
    return f"{head}\n{df.to_markdown(index=False, floatfmt='.6g')}{more}"


def observation(step: dict[str, Any], verifier_notes: list[str] | None = None) -> str:
    """Compact, truncated text the LLM sees after a tool call."""
    limit = get_settings().tool_output_chars
    parts: list[str] = []
    if step.get("ok"):
        parts.append("Status: success")
    else:
        parts.append(f"Status: ERROR ({step.get('error_type')})")
    if step.get("kernel_restarted"):
        parts.append("Note: the kernel was restarted; only the original datasets are loaded.")
    if step.get("stdout"):
        parts.append("stdout:\n" + step["stdout"][: limit // 2])
    if step.get("value"):
        parts.append("Last expression value: " + step["value"][:1500])
    for t in step.get("tables", [])[:3]:
        parts.append(_table_markdown(t))
    for c in step.get("charts", []):
        parts.append(f"Chart displayed to the user: {c.get('title') or '(untitled)'} [{c.get('kind')}]")
    if step.get("new_vars"):
        parts.append("Variables created/updated: " + ", ".join(step["new_vars"]))
    if step.get("error"):
        parts.append("Error:\n" + step["error"][-2000:])
    if verifier_notes:
        parts.append("Verifier warnings:\n" + "\n".join(f"- {n}" for n in verifier_notes))
    text = "\n\n".join(parts)
    if len(text) > limit:
        text = text[:limit] + f"\n... [truncated {len(text) - limit} characters]"
    return text
