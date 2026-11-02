"""Agent orchestrator: Planner -> Coder <-> Executor -> Verifier -> (Critic) -> Responder.

The loop is an explicit state machine written in plain Python (see docs/adr/0002). Each
role is a method; `run()` is an async generator that yields events, so the API can stream
progress to the UI while the agent works.

    plan ──► code ──► execute ──► verify ──┬─► (error/warning) ──► code   (retry, max N)
                                           └─► final_answer ──► ground ──► critic ──► respond
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncIterator

from backend.agent import prompts
from backend.agent.verifier import check_step, ungrounded_numbers
from backend.config import get_settings
from backend.llm.client import LLMResponse, chat, system_message, tool_result_message, user_message
from backend.memory.session import Session
from backend.memory.store import store
from backend.tools import executor
from backend.tools.definitions import AGENT_TOOLS

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    pass


def parse_json(text: str | None) -> dict[str, Any] | None:
    """Parse a JSON object from an LLM reply, tolerating ```json fences and surrounding prose."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None


class AgentRun:
    """One user question answered end to end."""

    def __init__(self, session: Session, question: str, *, model: str | None = None,
                 critic: bool | None = None) -> None:
        self.settings = get_settings()
        self.session = session
        self.question = question.strip()
        self.model = model or self.settings.llm_model
        self.critic_enabled = self.settings.critic_enabled if critic is None else critic
        self.steps: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.plan: dict[str, Any] | None = None
        self.started = time.perf_counter()

    # ------------------------------------------------------------ LLM with accounting

    async def _llm(self, role: str, messages: list[dict], *, tools: list[dict] | None = None,
                   json_mode: bool = False) -> LLMResponse:
        info = store.get_session(self.session.id) or {}
        used = int(info.get("tokens_in", 0)) + int(info.get("tokens_out", 0)) + self._tokens()
        if used >= self.settings.session_token_budget:
            raise BudgetExceeded(
                f"This session used {used:,} tokens, its budget is {self.settings.session_token_budget:,}. "
                "Start a new session or raise SESSION_TOKEN_BUDGET.")
        r = await chat(messages, tools=tools, model=self.model, json_mode=json_mode)
        self.llm_calls.append({"role": role, "model": r.model or self.model, "input_tokens": r.input_tokens,
                               "output_tokens": r.output_tokens, "cost_usd": r.cost_usd,
                               "latency_s": round(r.latency_s, 3), "tools": [t.name for t in r.tool_calls]})
        return r

    def _tokens(self) -> int:
        return sum(c["input_tokens"] + c["output_tokens"] for c in self.llm_calls)

    def usage(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "llm_calls": len(self.llm_calls),
            "input_tokens": sum(c["input_tokens"] for c in self.llm_calls),
            "output_tokens": sum(c["output_tokens"] for c in self.llm_calls),
            "cost_usd": round(sum(c["cost_usd"] for c in self.llm_calls), 6),
            "latency_s": round(time.perf_counter() - self.started, 2),
            "steps": len([s for s in self.steps if s["type"] in ("python", "sql")]),
            "failed_steps": len([s for s in self.steps if s["type"] in ("python", "sql") and not s["ok"]]),
            "calls": self.llm_calls,
            "prompt_versions": prompts.versions(),
        }

    # ------------------------------------------------------------ context

    async def _context(self) -> dict[str, str]:
        turns = store.list_turns(self.session.id)
        keep = self.settings.history_turns_verbatim
        info = store.get_session(self.session.id) or {}
        lines = []
        if info.get("summary") and len(turns) > keep:
            lines.append(f"Summary of earlier turns: {info['summary']}")
        for t in turns[-keep:]:
            lines.append(f"User: {t['question']}")
            if t["status"] == "clarification":
                lines.append(f"Agent asked for clarification: {t['answer_md']}")
                continue
            lines.append(f"Agent answer: {t.get('short_answer') or ''} | {(t.get('answer_md') or '')[:500]}")
            code = [s for s in t.get("steps", []) if s.get("ok") and s.get("type") in ("python", "sql")]
            if code:
                lines.append("Code that produced it:\n" + code[-1]["code"][:700])
        try:
            variables = await self.session.sandbox.avariables()
        except Exception:  # the sandbox may still be starting; not fatal for planning
            variables = []
        var_lines = [f"- {v['name']}: {v['description']}" for v in variables if not v.get("is_dataset")]
        conns = [f"- {c.name} ({c.kind}): tables {', '.join(c.tables[:30])}" for c in self.session.connections.values()]
        return {
            "schema": self.session.schema_text(self.question),
            "history": "\n".join(lines) or "(this is the first question)",
            "variables": "\n".join(var_lines[:40]) or "(none yet)",
            "connections": "\n".join(conns) or "(none; only uploaded files)",
        }

    # ------------------------------------------------------------ roles

    async def _planner(self, ctx: dict[str, str]) -> dict[str, Any]:
        prompt = prompts.render("planner", **ctx)
        r = await self._llm("planner", [system_message(prompt), user_message(self.question)], json_mode=True)
        plan = parse_json(r.text) or {}
        steps = plan.get("plan") if isinstance(plan.get("plan"), list) else []
        return {
            "needs_clarification": bool(plan.get("needs_clarification")) and bool(plan.get("clarifying_question")),
            "clarifying_question": str(plan.get("clarifying_question") or ""),
            "intent": str(plan.get("intent") or "other"),
            "plan": [{"step": i + 1, "goal": str(s.get("goal", s) if isinstance(s, dict) else s)} for i, s in enumerate(steps)]
                    or [{"step": 1, "goal": "Answer the question directly from the data."}],
            "expects_chart": bool(plan.get("expects_chart")),
        }

    async def _critic(self, ctx: dict[str, str], final: dict[str, Any]) -> dict[str, Any]:
        prompt = prompts.render("critic", question=self.question, schema=ctx["schema"],
                                steps=self._evidence(with_code=True), answer=json.dumps(final, ensure_ascii=False))
        r = await self._llm("critic", [user_message(prompt)], json_mode=True)
        verdict = parse_json(r.text) or {"verdict": "accept", "issues": []}
        verdict["verdict"] = "revise" if str(verdict.get("verdict", "")).lower() == "revise" else "accept"
        return verdict

    async def _responder(self, final: dict[str, Any]) -> str:
        prompt = prompts.render("responder", question=self.question, steps=self._evidence(with_code=False),
                                answer=json.dumps(final, ensure_ascii=False))
        r = await self._llm("responder", [user_message(prompt)])
        return (r.text or "").strip() or f"**{final.get('answer', '')}**"

    def _evidence(self, *, with_code: bool, limit: int = 9000) -> str:
        parts = []
        for i, s in enumerate(self.steps, 1):
            if s["type"] not in ("python", "sql") or not s["ok"]:
                continue
            head = f"## Step {i}: {s.get('purpose') or s['type']}"
            body = (f"```{s['type']}\n{s['code'][:1500]}\n```\n" if with_code else "") + executor.observation(s)
            parts.append(f"{head}\n{body}")
        text = "\n\n".join(parts) or "(no successful steps)"
        return text[-limit:]

    # ------------------------------------------------------------ tool dispatch

    async def _execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if "_invalid_json" in args:
            return {**executor._empty_step(name, "", str(args["_invalid_json"])[:500]),
                    "error": "Tool arguments were not valid JSON. Send a JSON object.", "error_type": "BadArguments"}
        if name == "run_python":
            return await executor.run_python(self.session, str(args.get("code", "")), str(args.get("purpose", "")))
        if name == "run_sql":
            return await executor.run_sql(self.session, str(args.get("query", "")), str(args.get("target", "files")),
                                          str(args.get("result_name", "sql_result")), str(args.get("purpose", "")))
        if name == "describe_table":
            return executor.describe_table(self.session, str(args.get("name", "")))
        return {**executor._empty_step(name, "", ""), "error": f"Unknown tool '{name}'.", "error_type": "UnknownTool"}

    # ------------------------------------------------------------ main loop

    async def run(self) -> AsyncIterator[dict[str, Any]]:
        turn: dict[str, Any] = {"question": self.question, "model": self.model, "status": "answered"}
        try:
            async for event in self._run(turn):
                yield event
        except BudgetExceeded as exc:
            turn.update(status="failed", answer_md=str(exc))
            yield {"type": "error", "message": str(exc)}
        except Exception as exc:  # LLM outage, auth error, ...: report instead of crashing the stream
            logger.exception("Agent run failed")
            msg = f"{type(exc).__name__}: {str(exc)[:500]}"
            turn.update(status="failed", answer_md=f"The agent stopped because of an error: {msg}")
            yield {"type": "error", "message": msg}
        turn.update(plan=self.plan, steps=self.steps, usage=self.usage())
        turn_id = store.save_turn(self.session.id, turn)
        u = turn["usage"]
        store.add_usage(self.session.id, u["input_tokens"], u["output_tokens"], u["cost_usd"])
        self._trace(turn_id, turn)
        yield {"type": "usage", "usage": {k: v for k, v in u.items() if k != "calls"}}
        try:
            await self._maybe_summarise()
        except Exception:  # the summary is an optimisation; never fail the turn for it
            logger.warning("Rolling summary update failed", exc_info=True)
        yield {"type": "done", "turn_id": turn_id, "status": turn["status"]}

    async def _run(self, turn: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        if len(store.list_turns(self.session.id)) == 0:
            store.rename_session(self.session.id, self.question[:80])

        yield {"type": "status", "stage": "planning", "message": "Reading the data profile and planning"}
        ctx = await self._context()
        self.plan = await self._planner(ctx)
        yield {"type": "plan", **self.plan}

        if self.plan["needs_clarification"]:
            turn.update(status="clarification", answer_md=self.plan["clarifying_question"], short_answer="")
            yield {"type": "clarification", "question": self.plan["clarifying_question"]}
            return

        plan_text = "\n".join(f"{p['step']}. {p['goal']}" for p in self.plan["plan"])
        if self.plan["expects_chart"]:
            plan_text += "\n(The user expects a chart: produce one with show_chart.)"
        few_shot = prompts.render("few_shot")
        messages = [
            system_message(prompts.render("coder_system", plan=plan_text, few_shot=few_shot, **ctx)),
            user_message(self.question),
        ]
        source_rows = max((d.rows for d in self.session.datasets.values()), default=None)
        failures = critic_rounds = grounding_rounds = nudges = 0
        final: dict[str, Any] | None = None
        stop_reason = ""

        for _ in range(self.settings.max_agent_steps):
            yield {"type": "status", "stage": "coding", "message": "Writing code"}
            r = await self._llm("coder", messages, tools=AGENT_TOOLS)
            if not r.tool_calls:
                if nudges == 0:
                    nudges += 1
                    if r.text:
                        messages.append({"role": "assistant", "content": r.text})
                    messages.append(user_message("Use run_python/run_sql to compute the answer from the data, "
                                                 "then call final_answer."))
                    continue
                final = {"answer": (r.text or "").strip()[:300], "key_findings": [], "caveats": ""}
                break
            messages.append(r.message)
            final_call = None
            for call in r.tool_calls:
                if call.name == "final_answer":
                    final_call = call
                    continue
                if call.name == "ask_user":
                    q = str(call.arguments.get("question", "Could you clarify your question?"))
                    turn.update(status="clarification", answer_md=q, short_answer="")
                    yield {"type": "clarification", "question": q}
                    return
                yield {"type": "step_start", "index": len(self.steps), "tool": call.name,
                       "purpose": call.arguments.get("purpose", ""),
                       "code": call.arguments.get("code") or call.arguments.get("query") or call.arguments.get("name", "")}
                step = await self._execute(call.name, call.arguments)
                notes = check_step(step, source_rows)
                step["verifier"] = notes
                step["index"] = len(self.steps)
                self.steps.append(step)
                yield {"type": "step", "step": step}
                messages.append(tool_result_message(call, executor.observation(step, notes)))
                failures = 0 if step["ok"] else failures + 1

            if failures >= self.settings.max_retries:
                stop_reason = f"{failures} consecutive steps failed"
                break

            if final_call is not None:
                candidate = {"answer": str(final_call.arguments.get("answer", "")),
                             "key_findings": [str(x) for x in final_call.arguments.get("key_findings") or []],
                             "caveats": str(final_call.arguments.get("caveats") or "")}
                claimed = candidate["answer"] + "\n" + "\n".join(candidate["key_findings"])
                missing = ungrounded_numbers(claimed, self.steps)
                if missing and grounding_rounds == 0:
                    grounding_rounds += 1
                    yield {"type": "verifier", "message": f"Numbers not found in executed output: {', '.join(missing[:6])}"}
                    messages.append(tool_result_message(final_call, (
                        "REJECTED by the verifier: these numbers do not appear in any executed output: "
                        f"{', '.join(missing[:6])}. Print them with code first, or correct the answer, "
                        "then call final_answer again.")))
                    continue
                if self.critic_enabled and critic_rounds == 0 and any(s["ok"] for s in self.steps):
                    yield {"type": "status", "stage": "verifying", "message": "Critic is checking the result"}
                    verdict = await self._critic(ctx, candidate)
                    yield {"type": "critic", **verdict}
                    if verdict["verdict"] == "revise":
                        critic_rounds += 1
                        messages.append(tool_result_message(final_call, (
                            "REVISE requested by the critic. Issues: " + "; ".join(map(str, verdict.get("issues", [])))
                            + f"\nSuggestion: {verdict.get('suggestion', '')}\nFix the analysis if the critic is "
                            "right (if the critic is wrong, call final_answer again unchanged).")))
                        continue
                messages.append(tool_result_message(final_call, "Accepted."))
                final = candidate
                break
        else:
            stop_reason = f"the step limit ({self.settings.max_agent_steps}) was reached"

        if final is None:
            last_error = next((s["error"] for s in reversed(self.steps) if s.get("error")), "")
            md = (f"**I could not complete this analysis** because {stop_reason or 'no answer was produced'}.\n\n"
                  + (f"Last error:\n```\n{last_error.strip()[-800:]}\n```\n\n" if last_error else "")
                  + "Try rephrasing the question, naming the exact columns, or splitting it into smaller questions.")
            turn.update(status="failed", answer_md=md, short_answer="")
            yield {"type": "answer", "answer_md": md, "short_answer": "", "key_findings": [], "status": "failed"}
            return

        yield {"type": "status", "stage": "responding", "message": "Writing the answer"}
        answer_md = await self._responder(final)
        turn.update(answer_md=answer_md, short_answer=final["answer"])
        yield {"type": "answer", "answer_md": answer_md, "short_answer": final["answer"],
               "key_findings": final["key_findings"], "caveats": final.get("caveats", ""), "status": "answered"}

    # ------------------------------------------------------------ memory and tracing

    async def _maybe_summarise(self) -> None:
        """Fold turns that leave the verbatim window into the rolling summary (F8)."""
        turns = store.list_turns(self.session.id)
        keep = self.settings.history_turns_verbatim
        if len(turns) <= keep:
            return
        leaving = turns[-keep - 1]
        info = store.get_session(self.session.id) or {}
        text = (f"User: {leaving['question']}\nAnswer: {leaving.get('short_answer')} | "
                f"{(leaving.get('answer_md') or '')[:600]}\nVariables: "
                + ", ".join(v for s in leaving.get("steps", []) for v in s.get("new_vars", [])))
        prompt = prompts.render("summarizer", summary=info.get("summary") or "(empty)", turns=text)
        r = await chat([user_message(prompt)], model=self.model, max_tokens=400)
        if r.text:
            store.set_summary(self.session.id, r.text.strip()[:1500])
        store.add_usage(self.session.id, r.input_tokens, r.output_tokens, r.cost_usd)

    def _trace(self, turn_id: str, turn: dict[str, Any]) -> None:
        """One JSON line per turn: the local trace log (Langfuse can be enabled on top)."""
        try:
            d = self.settings.trace_dir
            d.mkdir(parents=True, exist_ok=True)
            record = {"turn_id": turn_id, "session_id": self.session.id, "time": time.time(),
                      "question": self.question, "status": turn["status"], "plan": self.plan,
                      "usage": turn["usage"],
                      "steps": [{k: s.get(k) for k in ("type", "purpose", "code", "ok", "error_type", "duration_s",
                                                       "verifier")} for s in self.steps]}
            with open(d / f"{self.session.id}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.warning("Could not write trace", exc_info=True)


async def ask(session: Session, question: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
    """Run one turn while holding the session lock (one question at a time per session)."""
    async with session.lock:
        async for event in AgentRun(session, question, **kwargs).run():
            yield event
