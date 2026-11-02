"""The single gateway to the LLM. Nothing else in the app calls a provider directly."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import litellm

from backend.config import get_settings

logger = logging.getLogger(__name__)

litellm.drop_params = True  # ignore params a provider doesn't support instead of crashing
litellm.suppress_debug_info = True  # no "Give Feedback" banners in the server log
logging.getLogger("LiteLLM").setLevel(logging.WARNING)  # silence LiteLLM's per-call info logs
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    # Optional tracing of every LLM call (token cost, latency) in Langfuse; LiteLLM ships the callback.
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


# ---------- Normalized response types ----------

@dataclass
class ToolCall:
    id: str                      # links the call to its result
    name: str                    # which tool the model wants
    arguments: dict[str, Any]    # already parsed from JSON


@dataclass
class LLMResponse:
    text: str | None             # the model's written reply (None if it only called tools)
    tool_calls: list[ToolCall]
    message: dict[str, Any]      # the full assistant message, to append to history as-is
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    model: str = ""              # the model that actually answered (may be a fallback)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ---------- Message helpers ----------

def system_message(text: str) -> dict:
    return {"role": "system", "content": text}


def user_message(text: str) -> dict:
    return {"role": "user", "content": text}


def tool_result_message(call: ToolCall, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}


# ---------- The main call ----------

async def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> LLMResponse:
    """Send a conversation to the LLM and return a normalized response."""
    settings = get_settings()
    model = model or settings.llm_model
    candidates = [model] + [m for m in settings.fallback_models if m != model and not model.startswith("ollama")]
    last_exc: Exception | None = None
    for candidate in candidates:
        for attempt in range(2):
            try:
                await _limiter(candidate).acquire()
                return await _chat_once(messages, tools, candidate, temperature, max_tokens, json_mode)
            except _FALLBACK_ERRORS as exc:
                last_exc = exc
                wait = _per_minute_retry_delay(exc)
                if attempt == 0 and wait is not None and wait <= 65:
                    # Per-minute quota or a short overload: waiting is cheaper than switching models.
                    logger.warning("LLM %s busy (%s); retrying in %.0fs", candidate, type(exc).__name__, wait)
                    await asyncio.sleep(wait + 1)
                    continue
                logger.warning("LLM %s unavailable (%s); trying the next model", candidate, type(exc).__name__)
                break
    assert last_exc is not None
    raise last_exc


def _per_minute_retry_delay(exc: Exception) -> float | None:
    """Seconds to wait before retrying the same model, or None to move on to the next model."""
    text = str(exc)
    if isinstance(exc, (litellm.exceptions.ServiceUnavailableError, litellm.exceptions.InternalServerError)):
        return 8.0  # "model overloaded" spikes usually pass within seconds
    if not isinstance(exc, litellm.exceptions.RateLimitError) or "PerDay" in text:
        return None
    m = re.search(r"retry in ([\d.]+)s", text)
    return float(m.group(1)) if m else 20.0


class _RateLimiter:
    """Sliding-window limiter: at most `rpm` requests per 60 s for one model (free tiers allow ~15)."""

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self.calls: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self.rpm <= 0:
            return
        async with self.lock:
            while True:
                now = time.monotonic()
                while self.calls and now - self.calls[0] > 60:
                    self.calls.popleft()
                if len(self.calls) < self.rpm:
                    self.calls.append(now)
                    return
                await asyncio.sleep(60 - (now - self.calls[0]) + 0.05)


_limiters: dict[str, _RateLimiter] = {}


def _limiter(model: str) -> _RateLimiter:
    if model not in _limiters:
        _limiters[model] = _RateLimiter(0 if model.startswith("ollama") else get_settings().llm_requests_per_minute)
    return _limiters[model]


_FALLBACK_ERRORS = (
    litellm.exceptions.ServiceUnavailableError, litellm.exceptions.RateLimitError,
    litellm.exceptions.NotFoundError, litellm.exceptions.InternalServerError, litellm.exceptions.Timeout,
)


async def _chat_once(
    messages: list[dict], tools: list[dict] | None, model: str, temperature: float | None,
    max_tokens: int | None, json_mode: bool,
) -> LLMResponse:
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens or settings.llm_max_tokens,
        "num_retries": 1,  # one quick retry; after that the fallback chain takes over
    }
    temp = temperature if temperature is not None else settings.llm_temperature
    if temp is not None:  # only send it when explicitly set; otherwise use the model default
        kwargs["temperature"] = temp
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if model.startswith("gemini/"):
        kwargs["api_key"] = settings.gemini_api_key.get_secret_value()
    elif model.startswith("ollama"):
        kwargs["api_base"] = settings.ollama_base_url

    start = time.perf_counter()
    response = await litellm.acompletion(**kwargs)
    latency = time.perf_counter() - start

    msg = response.choices[0].message
    tool_calls = [_parse_tool_call(tc) for tc in (msg.tool_calls or [])]

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = 0.0  # unknown pricing for this model; not fatal

    usage = response.usage
    if usage is None:  # some local providers omit usage
        usage = litellm.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    logger.info(
        "LLM %s | %.1fs | in=%d out=%d | $%.5f | tools=%s",
        model, latency, usage.prompt_tokens, usage.completion_tokens, cost,
        [t.name for t in tool_calls] or "-",
    )

    return LLMResponse(
        text=msg.content,
        tool_calls=tool_calls,
        message=msg.model_dump(),
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=cost,
        latency_s=latency,
        model=model,
    )


def _parse_tool_call(tc: Any) -> ToolCall:
    """Turn the provider's tool call into our ToolCall, parsing JSON arguments safely."""
    raw = tc.function.arguments or "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        args = {"_invalid_json": raw}  # the agent loop can report this back to the model
    return ToolCall(id=tc.id, name=tc.function.name, arguments=args)
