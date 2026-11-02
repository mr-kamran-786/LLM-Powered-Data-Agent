"""Shared fixtures. Tests never call a real LLM: a scripted fake replaces `chat`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backend.config import get_settings
from backend.llm.client import LLMResponse, ToolCall


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Every test gets its own upload folder and session database."""
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "db_path", tmp_path / "uploads" / "sessions.sqlite3")
    monkeypatch.setattr(settings, "trace_dir", tmp_path / "uploads" / "traces")
    from backend.memory import store as store_mod

    monkeypatch.setattr(store_mod.store, "_conn", None)
    yield
    from backend.sandbox.manager import pool

    pool.shutdown()


@pytest.fixture
def sales_csv(tmp_path) -> Path:
    df = pd.DataFrame({
        "order_date": ["2024-01-05", "2024-02-10", "2025-01-03", "2025-03-09", "2025-03-09"],
        "region": ["North", "South", "North", "South", "South"],
        "revenue": [100.0, 50.0, 120.0, 90.0, 90.0],
        "units": [1, 2, 3, 1, 1],
    })
    path = tmp_path / "sales.csv"
    df.to_csv(path, index=False)
    return path


def reply(text: str | None = None, calls: list[tuple[str, dict[str, Any]]] | None = None) -> LLMResponse:
    tool_calls = [ToolCall(id=f"call_{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls or [])]
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [{"id": t.id, "type": "function",
                                  "function": {"name": t.name, "arguments": json.dumps(t.arguments)}} for t in tool_calls]
    return LLMResponse(text=text, tool_calls=tool_calls, message=message, input_tokens=100, output_tokens=20,
                       cost_usd=0.0001, latency_s=0.01, model="fake/model")


class ScriptedLLM:
    """Returns pre-written replies in order and records every prompt it received."""

    def __init__(self, replies: list[LLMResponse]) -> None:
        self.replies = list(replies)
        self.prompts: list[list[dict]] = []

    async def __call__(self, messages, tools=None, **kwargs) -> LLMResponse:
        self.prompts.append(list(messages))  # snapshot: the agent keeps appending to its list
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)
