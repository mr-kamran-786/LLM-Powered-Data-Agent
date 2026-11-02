# ADR 0004 — Dependency-free web UI served by FastAPI

**Status:** accepted

**Context.** The plan suggests Streamlit for the prototype and React + Vite for the final UI.

**Decision.** Phase 1 used Streamlit (`frontend/streamlit_app.py`, still runnable). The final UI is a
single-page app in plain HTML/CSS/JavaScript served by FastAPI itself, with Plotly.js loaded
from the Python package. It consumes the same SSE API a React client would.

**Consequences.** One command starts the whole system, no Node toolchain, works offline. The API
contract (docs/api.md) is unchanged, so a React client can replace it without backend changes.
