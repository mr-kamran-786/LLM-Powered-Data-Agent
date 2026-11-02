# Architecture

## Overview

The system is a loop, not a single prompt. A question flows through five roles; only the
Executor touches data, and only the Responder writes prose for the user.

```
                ┌──────────────────────────── Orchestrator (backend/agent) ───────────────────────────┐
 question ──►   │ Planner ──► Coder ──tool call──► Executor ──► Verifier ──┬── warning/error ──► Coder │
                │   │  (JSON plan or                (sandbox,   (rules)   │   (max 3 consecutive)     │
                │   │   clarifying question)         SQL guard)           └── final_answer            │
                │   ▼                                                           │                     │
                │ ask user                                     grounding check ─┤                     │
                │                                              LLM critic ──────┤ revise (once)       │
                │                                              Responder ◄──────┘                     │
                └──────────────────────────────────────────────────────────────────────────────────────┘
```

| Layer | Module | Responsibility |
|---|---|---|
| Web UI | `frontend/web` | Chat, uploads, data profile drawer, streamed steps, charts, exports |
| API | `backend/api` | REST + Server-Sent Events; sessions, uploads, connections, export |
| Orchestrator | `backend/agent/orchestrator.py` | State machine of the roles; streaming events; budget |
| Prompts | `prompts/*.md` | Versioned templates; versions recorded with every benchmark run |
| Tools | `backend/tools` | `run_python`, `run_sql`, `describe_table`, `ask_user`, `final_answer` |
| Sandbox | `backend/sandbox` | Stateful worker process (or container) per session |
| Data | `backend/data` | Ingestion, profiling, cleaning advice, read-only connectors, SQL guard |
| LLM | `backend/llm/client.py` | LiteLLM wrapper, cost/latency logging, fallback chain |
| Memory | `backend/memory` | SQLite store of sessions/turns; runtime session; schema retrieval |
| Reports | `backend/reports/export.py` | Markdown, HTML, PDF, Jupyter export |

## Request flow

1. `POST /api/sessions/{id}/chat` opens an SSE stream; the session lock serialises turns.
2. **Context**: dataset profiles (top-k tables by relevance for large schemas), the last
   four turns verbatim plus a rolling summary of older ones, and the variables still alive
   in the sandbox.
3. **Planner** (JSON mode) returns intent, 1–5 steps, whether a chart is expected, or a
   clarifying question (F9).
4. **Coder** works in a tool-calling loop. Each tool result is turned into a compact,
   truncated observation (stdout, first rows of tables, chart titles, errors).
5. **Executor** runs Python in the sandbox or SQL through DuckDB/the database driver.
6. **Verifier** checks every step: empty results, all-NaN columns, percentages over 100,
   NaN totals, row explosions after joins. Warnings and tracebacks go back to the Coder.
   Three consecutive failures stop the loop with an honest explanation (F5).
7. **Grounding check**: numbers in `final_answer` must appear (within rounding) in executed
   output; otherwise the answer is rejected once.
8. **Critic** (optional, F14): a second LLM pass judges the code and output against the
   question and may request one revision.
9. **Responder** writes the final Markdown answer from the evidence only.
10. The turn (plan, every step with code/output/tables/charts, usage) is stored in SQLite
    and appended to the JSONL trace log.

## Sandbox

| Layer | Process backend (default) | Docker backend |
|---|---|---|
| Isolation | separate OS process | container per session |
| Network | audit hook blocks `socket.connect`, DNS, `urllib` | `--network none` |
| Processes | audit hook blocks `subprocess`, `os.system`, `exec*`, `fork` | plus `--pids-limit` |
| Files | reads only session/data/library folders; writes only to the work folder | read-only data mount, `--read-only` root |
| Secrets | environment stripped (no API keys) | same |
| Time | 60 s per cell, then kill + restart + reload datasets | same |
| Memory | `RLIMIT_AS` on Linux | `--memory 4g --cpus 2` |
| Static check | AST check of imports, calls and paths before execution | same |
| DuckDB | `enable_external_access=false`, configuration locked | same |

The worker speaks JSON lines over stdin/stdout, which is why the same script runs
unchanged in both backends.

## Data safety

* Connections are opened read-only by the driver (`mode=ro`, `read_only=True`,
  `default_transaction_read_only=on`).
* Every query is parsed by sqlglot; only a single `SELECT`/`WITH`/set operation is allowed,
  and file-reading table functions are rejected.
* Only the profile and five sample rows (cells truncated to 60 characters) are sent to the
  LLM, never the dataset; data is treated as data in every prompt.

## Persistence

`uploads/sessions.sqlite3` holds `sessions`, `datasets`, `connections` and `turns`.
Uploaded files are normalised to Parquet in `uploads/<session>/data/`. After a server restart
a session is restored lazily: datasets are reloaded into a new sandbox, database
connections are re-opened.
