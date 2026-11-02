"""Chat endpoint: streams the agent's progress as Server-Sent Events (SSE)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agent.orchestrator import ask
from backend.api.routes.sessions import get_session
from backend.config import get_settings

router = APIRouter(prefix="/api/sessions", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    model: str | None = None        # F13: swap the LLM per question
    critic: bool | None = None      # F14: multi-agent mode (LLM critic) on/off


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"


@router.post("/{session_id}/chat")
async def chat(session_id: str, body: ChatRequest) -> StreamingResponse:
    sess = get_session(session_id)
    if body.model and body.model not in get_settings().available_models:
        raise HTTPException(400, f"Unknown model '{body.model}'")

    async def stream() -> AsyncIterator[str]:
        yield ": connected\n\n"
        async for event in ask(sess, body.message, model=body.model, critic=body.critic):
            yield _sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
