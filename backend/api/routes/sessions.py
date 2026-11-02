"""Sessions, dataset uploads, profiles, cleaning suggestions and database connections."""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.config import PROJECT_ROOT, get_settings
from backend.data.connectors import SUPPORTED_KINDS
from backend.data.ingestion import SUPPORTED_EXTENSIONS, IngestionError
from backend.memory.session import Session, registry
from backend.memory.store import store
from backend.sandbox.payload import table_payload

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def get_session(session_id: str) -> Session:
    try:
        return registry.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session '{session_id}' not found") from None


def session_payload(sess: Session) -> dict[str, Any]:
    info = store.get_session(sess.id) or {}
    return {
        "id": sess.id,
        "title": info.get("title"),
        "created_at": info.get("created_at"),
        "usage": {"tokens_in": info.get("tokens_in", 0), "tokens_out": info.get("tokens_out", 0),
                  "cost_usd": round(info.get("cost_usd", 0.0), 6),
                  "budget_tokens": get_settings().session_token_budget},
        "datasets": [d.public_info() for d in sess.datasets.values()],
        "connections": [c.public_info() for c in sess.connections.values()],
        "turns": store.list_turns(sess.id),
    }


class CreateSession(BaseModel):
    title: str = "New analysis"


@router.post("")
def create_session(body: CreateSession | None = None) -> dict[str, Any]:
    sess = registry.create((body.title if body else None) or "New analysis")
    return session_payload(sess)


@router.get("")
def list_sessions() -> list[dict[str, Any]]:
    return store.list_sessions()


@router.get("/{session_id}")
def read_session(session_id: str) -> dict[str, Any]:
    return session_payload(get_session(session_id))


@router.delete("/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    get_session(session_id)
    registry.delete(session_id)
    shutil.rmtree(get_settings().upload_dir / session_id, ignore_errors=True)
    return {"status": "deleted"}


# ---------------------------------------------------------------- datasets (F1, F3, F10)


@router.post("/{session_id}/datasets")
async def upload_dataset(session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    sess = get_session(session_id)
    filename = Path(file.filename or "data.csv").name
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    raw_dir = sess.dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / re.sub(r"[^\w.\- ]", "_", filename)
    limit = get_settings().max_upload_mb * 1024 * 1024
    size = 0
    with open(target, "wb") as out:  # stream to disk in chunks; never hold 500 MB in memory
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, f"File is larger than {get_settings().max_upload_mb} MB")
            out.write(chunk)
    try:
        ds = await asyncio.to_thread(sess.add_file, target, filename)
    except IngestionError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"dataset": ds.public_info(), "profile": ds.profile}


@router.get("/{session_id}/datasets/{name}/profile")
def dataset_profile(session_id: str, name: str) -> dict[str, Any]:
    sess = get_session(session_id)
    ds = sess.datasets.get(name)
    if ds is None:
        raise HTTPException(404, f"Dataset '{name}' not found")
    return {"dataset": ds.public_info(), "profile": ds.profile, "profile_text": ds.profile_text}


@router.get("/{session_id}/datasets/{name}/preview")
async def dataset_preview(session_id: str, name: str, rows: int = 100) -> dict[str, Any]:
    sess = get_session(session_id)
    if name not in sess.datasets:
        raise HTTPException(404, f"Dataset '{name}' not found")
    df: pd.DataFrame = await asyncio.to_thread(sess.frame, name, min(max(rows, 1), 500))
    payload = table_payload(df, name)
    payload["rows"] = payload["rows"][:rows]
    return payload


# ---------------------------------------------------------------- database connections (F2)


class ConnectRequest(BaseModel):
    kind: str = Field(description="sqlite | duckdb | postgres")
    target: str = Field(description="File path (sqlite/duckdb) or postgresql:// DSN")
    name: str | None = None


@router.post("/{session_id}/connections")
async def connect_database(session_id: str, body: ConnectRequest) -> dict[str, Any]:
    sess = get_session(session_id)
    if body.kind not in SUPPORTED_KINDS:
        raise HTTPException(400, f"kind must be one of {SUPPORTED_KINDS}")
    target = body.target.strip()
    if body.kind in ("sqlite", "duckdb"):
        path = Path(target)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            raise HTTPException(404, f"Database file not found: {target}")
        target = str(path)
    name = re.sub(r"[^a-z0-9_]", "_", (body.name or Path(target).stem or body.kind).lower())[:40] or "db"
    try:
        conn = await asyncio.to_thread(sess.add_connection, name, body.kind, target)
    except Exception as exc:
        raise HTTPException(400, f"Could not connect: {type(exc).__name__}: {exc}") from exc
    return {"connection": conn.public_info(), "session": session_payload(sess)}



SAMPLES_DIR = PROJECT_ROOT / "eval" / "datasets"


@router.post("/{session_id}/samples")
async def load_samples(session_id: str) -> dict[str, Any]:
    """Load the demo datasets (sales, taxi trips, employees) and connect the shop database."""
    sess = get_session(session_id)
    for fname in ("sales.csv", "taxi_trips.parquet", "employees.xlsx"):
        path = SAMPLES_DIR / fname
        if path.exists() and Path(fname).stem not in sess.datasets:
            await asyncio.to_thread(sess.add_file, path, fname)
    db = SAMPLES_DIR / "shop.sqlite"
    if db.exists() and "shop" not in sess.connections:
        await asyncio.to_thread(sess.add_connection, "shop", "sqlite", str(db))
    return session_payload(sess)
