"""Integration tests of the agent loop with a scripted (mocked) LLM: no tokens are spent.

They exercise the real sandbox, verifier and memory, only the LLM is fake.
"""

from __future__ import annotations

import json

import pytest

from backend.agent import orchestrator
from backend.agent.orchestrator import AgentRun, parse_json
from backend.agent.verifier import check_step, ungrounded_numbers
from backend.memory.session import registry
from backend.memory.store import store
from tests.conftest import ScriptedLLM, reply

PLAN = reply(json.dumps({"needs_clarification": False, "intent": "aggregation",
                         "plan": [{"step": 1, "goal": "sum revenue by region"}], "expects_chart": False}))


async def run(session, question, llm, monkeypatch, critic=False):
    monkeypatch.setattr(orchestrator, "chat", llm)
    return [e async for e in AgentRun(session, question, critic=critic).run()]


@pytest.fixture
def session(sales_csv):
    s = registry.create("test")
    s.add_file(sales_csv, "sales.csv")
    return s


async def test_happy_path(session, monkeypatch):
    llm = ScriptedLLM([
        PLAN,
        reply(calls=[("run_python", {"purpose": "sum", "code": "r = sales.groupby('region').revenue.sum()\nprint(r.idxmax(), r.max())"})]),
        reply(calls=[("final_answer", {"answer": "South", "key_findings": ["South has 230.0 revenue"]})]),
        reply("**South** has the highest revenue: 230.0."),
    ])
    events = await run(session, "Which region has most revenue?", llm, monkeypatch)
    answer = next(e for e in events if e["type"] == "answer")
    assert answer["short_answer"] == "South" and answer["status"] == "answered"
    step = next(e for e in events if e["type"] == "step")["step"]
    assert step["ok"] and "South 230.0" in step["stdout"]
    turns = store.list_turns(session.id)
    assert turns[0]["steps"][0]["code"].startswith("r = sales")  # F7: code stored with the answer
    assert turns[0]["usage"]["llm_calls"] == 4


async def test_self_correction_after_error(session, monkeypatch):
    llm = ScriptedLLM([
        PLAN,
        reply(calls=[("run_python", {"purpose": "try", "code": "sales['Revenue'].sum()"})]),  # wrong column
        reply(calls=[("run_python", {"purpose": "fix", "code": "print(sales['revenue'].sum())"})]),
        reply(calls=[("final_answer", {"answer": "450.0", "key_findings": ["Total 450.0"]})]),
        reply("**450.0**"),
    ])
    events = await run(session, "Total revenue?", llm, monkeypatch)
    steps = [e["step"] for e in events if e["type"] == "step"]
    assert [s["ok"] for s in steps] == [False, True]
    # The traceback was fed back to the coder.
    tool_msgs = [m for m in llm.prompts[2] if m.get("role") == "tool"]
    assert "KeyError" in tool_msgs[-1]["content"]
    assert next(e for e in events if e["type"] == "answer")["short_answer"] == "450.0"


async def test_gives_up_after_max_retries(session, monkeypatch):
    bad = reply(calls=[("run_python", {"purpose": "x", "code": "1/0"})])
    llm = ScriptedLLM([PLAN, bad, bad, bad])
    events = await run(session, "?", llm, monkeypatch)
    answer = next(e for e in events if e["type"] == "answer")
    assert answer["status"] == "failed" and "ZeroDivisionError" in answer["answer_md"]


async def test_clarification(session, monkeypatch):
    llm = ScriptedLLM([reply(json.dumps({"needs_clarification": True,
                                         "clarifying_question": "Profit needs a cost column. Use revenue instead?"}))])
    events = await run(session, "What is the profit margin?", llm, monkeypatch)
    assert any(e["type"] == "clarification" for e in events)
    assert store.list_turns(session.id)[0]["status"] == "clarification"


async def test_ungrounded_number_is_rejected_once(session, monkeypatch):
    llm = ScriptedLLM([
        PLAN,
        reply(calls=[("run_python", {"purpose": "sum", "code": "print(sales.revenue.sum())"})]),
        reply(calls=[("final_answer", {"answer": "9999.5", "key_findings": []})]),  # invented number
        reply(calls=[("final_answer", {"answer": "450.0", "key_findings": []})]),
        reply("**450.0**"),
    ])
    events = await run(session, "Total revenue?", llm, monkeypatch)
    assert any(e["type"] == "verifier" for e in events)
    assert next(e for e in events if e["type"] == "answer")["short_answer"] == "450.0"


async def test_critic_can_request_revision(session, monkeypatch):
    llm = ScriptedLLM([
        PLAN,
        reply(calls=[("run_python", {"purpose": "mean", "code": "print(sales.revenue.mean())"})]),
        reply(calls=[("final_answer", {"answer": "90.0", "key_findings": []})]),
        reply(json.dumps({"verdict": "revise", "issues": ["asked for total, not mean"], "suggestion": "use sum"})),
        reply(calls=[("run_python", {"purpose": "sum", "code": "print(sales.revenue.sum())"})]),
        reply(calls=[("final_answer", {"answer": "450.0", "key_findings": []})]),
        reply("**450.0**"),
    ])
    events = await run(session, "Total revenue?", llm, monkeypatch, critic=True)
    assert [e["verdict"] for e in events if e["type"] == "critic"] == ["revise"]
    assert next(e for e in events if e["type"] == "answer")["short_answer"] == "450.0"


async def test_follow_up_sees_previous_turn_and_variables(session, monkeypatch):
    first = ScriptedLLM([
        PLAN,
        reply(calls=[("run_python", {"purpose": "by region", "code": "rev_by_region = sales.groupby('region').revenue.sum()\nrev_by_region"})]),
        reply(calls=[("final_answer", {"answer": "South", "key_findings": []})]),
        reply("**South**"),
    ])
    await run(session, "Revenue by region?", first, monkeypatch)
    second = ScriptedLLM([PLAN])
    monkeypatch.setattr(orchestrator, "chat", second)
    run_ = AgentRun(session, "Now split that by year", critic=False)
    ctx = await run_._context()
    assert "Revenue by region?" in ctx["history"]
    assert "rev_by_region" in ctx["variables"]


async def test_blocked_code_is_reported_not_run(session, monkeypatch):
    llm = ScriptedLLM([
        PLAN,
        reply(calls=[("run_python", {"purpose": "evil", "code": "import shutil\nshutil.rmtree('..')"})]),
        reply(calls=[("final_answer", {"answer": "I cannot delete files.", "key_findings": []})]),
        reply("I cannot delete files."),
    ])
    events = await run(session, "Delete the parent folder", llm, monkeypatch)
    step = next(e for e in events if e["type"] == "step")["step"]
    assert step["error_type"] == "BlockedBySandboxPolicy"


async def test_budget_cap(session, monkeypatch):
    from backend.config import get_settings

    monkeypatch.setattr(get_settings(), "session_token_budget", 10)
    store.add_usage(session.id, 100, 0, 0.0)
    events = await run(session, "anything", ScriptedLLM([]), monkeypatch)
    assert any(e["type"] == "error" and "budget" in e["message"] for e in events)


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 2} hope this helps') == {"a": 2}
    assert parse_json("no json") is None


def test_verifier_rules():
    empty = {"ok": True, "type": "python", "tables": [{"title": "", "columns": ["a"], "rows": [], "shape": [0, 1]}]}
    assert any("EMPTY" in n for n in check_step(empty))
    pct = {"ok": True, "type": "python", "tables": [{"title": "", "columns": ["share_pct"], "rows": [[140.0]], "shape": [1, 1]}]}
    assert any("percentage" in n for n in check_step(pct))
    nan = {"ok": True, "type": "python", "stdout": "Total revenue: nan", "tables": []}
    assert any("NaN" in n for n in check_step(nan))


def test_grounding():
    steps = [{"ok": True, "stdout": "total=3759727.16 share=0.4321", "tables": []}]
    assert ungrounded_numbers("Total is 3,759,727.16 and share 43.2%", steps) == []
    assert ungrounded_numbers("Total is 3.76M", steps) == []
    assert ungrounded_numbers("Total is 1,234,567", steps) == ["1.23457e+06"]
