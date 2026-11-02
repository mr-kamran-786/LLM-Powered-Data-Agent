# ADR 0001 — Gemini via LiteLLM as the LLM layer

**Status:** accepted

**Context.** The group has a Gemini API key and no GPU server. Requirement F13 asks for swappable
providers, including a local model.

**Decision.** All LLM calls go through `backend/llm/client.py`, a thin wrapper around LiteLLM.
The default model is `gemini/gemini-3.1-flash-lite` (cheap, fast tool calling). A fallback chain
(`gemini-3.5-flash-lite`, `gemini-3-flash-preview`, `gemini-3.5-flash`) is tried automatically on
HTTP 503/429/404, which we observed in practice when the default model was overloaded.

**Consequences.** Switching to Claude, GPT or an Ollama model is a configuration change. Costs and
latency are logged per call. Tool-calling quality differs between models, so every model change
must be accompanied by a benchmark run.
