# ADR 0003 — Process sandbox with audit hooks, Docker as an option

**Status:** accepted

**Context.** Generated code must run isolated, without network and with limits. Docker is not
available on every group member's laptop (Windows without Docker Desktop).

**Decision.** One worker process per session, speaking JSON lines over stdin/stdout. Defence in
depth: (1) static AST check; (2) a CPython audit hook installed after the libraries are imported,
blocking network, subprocesses and file access outside the session folders; (3) stripped
environment without secrets; (4) per-cell timeout with kill and restart; (5) DuckDB external
access disabled. The same worker runs in a container (`SANDBOX_BACKEND=docker`) with
`--network none`, memory/CPU/pid limits, a non-root user and a read-only data mount.

**Consequences.** Works everywhere and starts in ~3.5 s. Audit hooks are not a security boundary
against a determined attacker with native code; for untrusted multi-user deployment the Docker
backend must be used.
