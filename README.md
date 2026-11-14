# Data Agent — LLM-Powered Data Analysis

University West · Project Group 23 — Adnan Adnan, Atiqa Bashir, Kamran Rauf, Shruti Sharma

Upload a dataset or connect a database, ask a question in plain language, and the agent
**plans** the analysis, **writes and runs** pandas/SQL code in a sandbox, **checks** its own
result, and answers with numbers, charts and the code that produced them.

```
 

## Quick start (Windows, local)

```bash
python -m venv zonix && zonix\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # then put your GEMINI_API_KEY in .env
python -m eval.make_datasets    # demo datasets + benchmark
uvicorn backend.main:app --port 8000
```

Open <http://localhost:8000>, click **Load demo datasets**, and ask e.g.
*"Which region grew fastest in revenue from 2024 to 2025?"* then *"Now split that by product category"*.

With Docker: `docker compose up --build` (same URL).

## Features

| ID | Requirement | Where |
|---|---|---|
| F1 | Upload CSV, Excel, Parquet, JSON up to 500 MB | `backend/data/ingestion.py`, streamed upload in `api/routes/sessions.py` |
| F2 | Read-only SQLite, DuckDB, PostgreSQL | `backend/data/connectors.py`, `sql_guard.py` (sqlglot) |
| F3 | Automatic profiling | `backend/data/profiler.py`, data drawer in the UI |
| F4 | Answer questions by generating and executing code | `backend/agent/orchestrator.py`, `backend/tools/` |
| F5 | Self-correction, max N retries | orchestrator loop (`MAX_RETRIES`) |
| F6 | Bar, line, scatter, histogram, heatmap, box charts | Plotly in the sandbox, rendered in the UI |
| F7 | Code and intermediate tables for every answer | step cards in the UI, stored per turn in SQLite |
| F8 | Multi-turn memory | persistent sandbox variables + history + rolling summary |
| F9 | Clarifying questions | Planner `needs_clarification`, `ask_user` tool |
| F10 | Data-cleaning suggestions | `profiler.cleaning_suggestions`, "Cleaning" tab |
| F11 | Statistics and ML | scipy, scikit-learn, statsmodels in the sandbox |
| F12 | Report export: Markdown, HTML, PDF, notebook | `backend/reports/export.py` |
| F13 | Swap LLM providers / local model | LiteLLM, model picker, automatic fallbacks, Ollama |
| F14 | Multi-agent mode (planner, coder, critic) | Critic toggle |

Safety: generated code runs in a separate process with no API keys in its environment, a
runtime audit hook that blocks network, subprocesses and file access outside the session
folder, a static code check, per-cell timeouts and (in Docker mode) `--network none` containers.
Every numeric claim in an answer is checked against executed output before it is shown.

## Project layout

```
backend/   api/ agent/ tools/ sandbox/ data/ llm/ memory/ reports/
prompts/   versioned prompt templates (planner, coder, critic, responder)
frontend/  web/ (main UI, no build step) · streamlit_app.py (phase-1 prototype)
eval/      make_datasets.py · run_eval.py · datasets/ · results/
tests/     unit, sandbox-security, agent (mocked LLM) and API tests
docs/      architecture, user guide, API, ADRs, final report
```

## Tests and benchmark

```bash
pytest -q                                      # 72 tests, no API key needed
python -m eval.run_eval --limit 30             # 30-question subset
python -m eval.run_eval                        # full benchmark (102 questions)
python -m eval.run_eval --model gemini/gemini-3.5-flash --tag compare
```

Results are written to `eval/results/` (JSON + Markdown, one file per run).

## Documentation

- [Architecture](docs/architecture.md)
- [User guide](docs/user_guide.md)
- [API reference](docs/api.md)
- [Architecture decision records](docs/adr/)
- Final report: `docs/report/`
