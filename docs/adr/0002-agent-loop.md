# ADR 0002 — Explicit state machine in plain Python instead of LangGraph

**Status:** accepted

**Context.** The plan recommends LangGraph, with a plain Python loop as an alternative. The group
has four members instead of the 8–12 the plan assumes.

**Decision.** The orchestrator is a small explicit state machine (`AgentRun`): Planner → Coder ⇄
Executor → Verifier → Critic → Responder, implemented as async methods with the retry as a loop
condition. Each role maps one-to-one onto a LangGraph node and the retry onto a conditional
edge, so migrating later is mechanical.

**Consequences.** No extra framework to learn or debug; the whole control flow fits in one
readable file; streaming is a plain async generator. We give up LangGraph's built-in
checkpointing and visual graph, which the project does not need.
