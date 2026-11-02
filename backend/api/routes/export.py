"""Export a session as a report (requirement F12)."""

from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.api.routes.sessions import get_session
from backend.memory.store import store
from backend.reports import export

router = APIRouter(prefix="/api/sessions", tags=["export"])

FORMATS = {
    "md": ("text/markdown; charset=utf-8", export.to_markdown),
    "html": ("text/html; charset=utf-8", export.to_html),
    "ipynb": ("application/x-ipynb+json", export.to_notebook),
    "pdf": ("application/pdf", export.to_pdf),
}


@router.get("/{session_id}/export")
async def export_session(session_id: str, format: str = "md") -> Response:
    sess = get_session(session_id)
    if format not in FORMATS:
        raise HTTPException(400, f"format must be one of {list(FORMATS)}")
    media, fn = FORMATS[format]
    body = await asyncio.to_thread(fn, sess)
    title = (store.get_session(session_id) or {}).get("title") or "analysis"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-")[:50].lower() or "analysis"
    return Response(content=body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{slug}.{format}"'})
