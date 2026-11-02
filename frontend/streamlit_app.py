"""Phase 1 prototype UI (Streamlit). The main UI is the web app served by FastAPI.

Run: streamlit run frontend/streamlit_app.py
It uses the same agent, sandbox and memory as the API, in-process.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agent.orchestrator import ask  # noqa: E402
from backend.memory.session import registry  # noqa: E402

st.set_page_config(page_title="Data Agent (prototype)", layout="wide")
st.title("Data Agent — Streamlit prototype")

if "session_id" not in st.session_state:
    st.session_state.session_id = registry.create("Streamlit prototype").id
    st.session_state.history = []
session = registry.get(st.session_state.session_id)

with st.sidebar:
    upload = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "parquet", "json"])
    if upload and upload.name not in [d.original_filename for d in session.datasets.values()]:
        path = Path(tempfile.mkdtemp()) / upload.name
        path.write_bytes(upload.getvalue())
        ds = session.add_file(path, upload.name)
        st.success(f"Loaded `{ds.name}`: {ds.rows:,} rows")
    for d in session.datasets.values():
        st.caption(f"`{d.name}` · {d.rows:,} rows × {d.columns} cols")


def render(events: list[dict]) -> None:
    for ev in events:
        if ev["type"] == "plan":
            st.markdown("**Plan:** " + " → ".join(p["goal"] for p in ev["plan"]))
        elif ev["type"] == "step":
            s = ev["step"]
            with st.expander(("✅ " if s["ok"] else "❌ ") + (s.get("purpose") or s["type"])):
                st.code(s["code"], language="sql" if s["type"] == "sql" else "python")
                if s.get("stdout"):
                    st.text(s["stdout"])
                for t in s.get("tables", []):
                    st.dataframe(pd.DataFrame(t["rows"], columns=t["columns"]))
                if s.get("error"):
                    st.error(s["error"])
            for c in s.get("charts", []) if s["ok"] else []:
                if c["kind"] == "plotly":
                    st.plotly_chart(pio.from_json(__import__("json").dumps(c["spec"])), use_container_width=True)
        elif ev["type"] in ("answer", "clarification"):
            st.markdown(ev.get("answer_md") or ev.get("question"))
        elif ev["type"] == "error":
            st.error(ev["message"])


for q, events in st.session_state.history:
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        render(events)

if question := st.chat_input("Ask a question about your data"):
    st.chat_message("user").write(question)
    with st.chat_message("assistant"), st.spinner("Analysing…"):
        async def collect() -> list[dict]:
            return [e async for e in ask(session, question)]
        events = asyncio.run(collect())
        render(events)
    st.session_state.history.append((question, events))
