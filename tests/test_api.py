"""API tests with FastAPI's TestClient (LLM mocked)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.agent import orchestrator
from backend.main import app
from tests.conftest import ScriptedLLM, reply


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_config(client):
    assert client.get("/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["default_model"] in cfg["models"]
    assert "planner" in cfg["prompt_versions"]


def test_upload_profile_preview_and_chat(client, sales_csv, monkeypatch):
    sid = client.post("/api/sessions", json={"title": "t"}).json()["id"]
    with open(sales_csv, "rb") as fh:
        r = client.post(f"/api/sessions/{sid}/datasets", files={"file": ("sales.csv", fh, "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["dataset"]["rows"] == 5
    prof = client.get(f"/api/sessions/{sid}/datasets/sales/profile").json()
    assert prof["profile"]["duplicate_rows"] == 1
    prev = client.get(f"/api/sessions/{sid}/datasets/sales/preview?rows=2").json()
    assert len(prev["rows"]) == 2

    llm = ScriptedLLM([
        reply(json.dumps({"needs_clarification": False, "plan": [{"step": 1, "goal": "count"}]})),
        reply(calls=[("run_python", {"purpose": "count", "code": "print(len(sales))"})]),
        reply(calls=[("final_answer", {"answer": "5", "key_findings": []})]),
        reply("**5** rows"),
    ])
    monkeypatch.setattr(orchestrator, "chat", llm)
    with client.stream("POST", f"/api/sessions/{sid}/chat", json={"message": "How many rows?", "critic": False}) as resp:
        body = "".join(resp.iter_text())
    events = [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")]
    types = [e["type"] for e in events]
    assert types[0] == "status" and "plan" in types and "answer" in types and types[-1] == "done"

    for fmt in ("md", "html", "ipynb", "pdf"):
        r = client.get(f"/api/sessions/{sid}/export?format={fmt}")
        assert r.status_code == 200 and len(r.content) > 200, fmt
    assert "print(len(sales))" in client.get(f"/api/sessions/{sid}/export?format=md").text


def test_rejects_bad_upload(client, tmp_path):
    sid = client.post("/api/sessions").json()["id"]
    bad = tmp_path / "evil.exe"
    bad.write_bytes(b"MZ")
    with open(bad, "rb") as fh:
        r = client.post(f"/api/sessions/{sid}/datasets", files={"file": ("evil.exe", fh)})
    assert r.status_code == 400


def test_unknown_session_404(client):
    assert client.get("/api/sessions/doesnotexist").status_code == 404
