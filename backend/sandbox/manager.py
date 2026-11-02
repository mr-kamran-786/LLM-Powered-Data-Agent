"""Sandbox manager: one isolated, stateful worker process per session.

Two backends share the same worker script and JSON-lines protocol:

* ``process`` (default, works everywhere): a separate Python process with a stripped
  environment (no API keys), a runtime audit hook (no network, no subprocesses, no file
  access outside the session folder), a per-cell timeout and, on Linux, a memory limit.
* ``docker``: the same worker inside a container started with ``--network none``,
  CPU/memory limits, a non-root user and a read-only data mount (plan, section 6).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from backend.config import get_settings

logger = logging.getLogger(__name__)

WORKER_SCRIPT = Path(__file__).with_name("worker.py")
STARTUP_TIMEOUT_S = 90


class SandboxError(RuntimeError):
    pass


class Sandbox:
    """A stateful kernel. Variables survive between calls until the kernel restarts."""

    def __init__(self, session_dir: Path, backend: str | None = None) -> None:
        self.session_dir = session_dir
        self.work_dir = session_dir / "work"
        self.data_dir = session_dir / "data"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend or os.environ.get("SANDBOX_BACKEND", "process")
        self.datasets: dict[str, Path] = {}  # loaded again after every restart
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._next_id = 0
        self._lock = threading.Lock()
        self.restarts = 0

    # ------------------------------------------------------------ lifecycle

    def _command(self) -> list[str]:
        settings = get_settings()
        if self.backend == "docker":
            image = os.environ.get("SANDBOX_IMAGE", "data-agent-sandbox:latest")
            args = [f"{n}=/data/{p.name}" for n, p in self.datasets.items()]
            return [
                "docker", "run", "-i", "--rm", "--network", "none",
                "--memory", f"{settings.sandbox_memory_mb}m", "--cpus", "2", "--pids-limit", "128",
                "--read-only", "--tmpfs", "/tmp", "--user", "1000:1000",
                "-v", f"{self.data_dir.resolve()}:/data:ro", "-v", f"{self.work_dir.resolve()}:/work",
                image, "python", "-I", "/sandbox/worker.py", "/work", "/data", str(settings.sandbox_memory_mb), *args,
            ]
        args = [f"{n}={p.resolve()}" for n, p in self.datasets.items()]
        return [sys.executable, "-I", "-u", str(WORKER_SCRIPT), str(self.work_dir.resolve()),
                str(self.data_dir.resolve()), str(settings.sandbox_memory_mb), *args]

    def _env(self) -> dict[str, str]:
        """Minimal environment: nothing secret (no API keys) reaches generated code."""
        keep = ("SYSTEMROOT", "WINDIR", "PATHEXT", "COMSPEC", "LANG", "LC_ALL", "TZ")
        env = {k: v for k, v in os.environ.items() if k.upper() in keep}
        tmp = str((self.work_dir / "tmp").resolve())
        Path(tmp).mkdir(exist_ok=True)
        env.update({
            "PATH": os.path.dirname(sys.executable) if self.backend == "process" else os.environ.get("PATH", ""),
            "HOME": str(self.work_dir.resolve()), "USERPROFILE": str(self.work_dir.resolve()),
            "TEMP": tmp, "TMP": tmp, "TMPDIR": tmp, "MPLCONFIGDIR": str((self.work_dir / ".mpl").resolve()),
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "LOKY_MAX_CPU_COUNT": "1", "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0",
        })
        return env

    def start(self) -> None:
        if self.alive:
            return
        self._lines = queue.Queue()
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            self._command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=str(self.work_dir), env=self._env(), text=True, encoding="utf-8", bufsize=1,
            creationflags=flags,
        )
        threading.Thread(target=self._pump, args=(self._proc,), daemon=True).start()
        ready = self._read_json(lambda m: m.get("ready"), STARTUP_TIMEOUT_S)
        if ready is None:
            self.stop()
            raise SandboxError("Sandbox worker did not start.")
        logger.info("Sandbox started (%s, pid %s) for %s", self.backend, ready.get("pid"), self.session_dir.name)

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self._lines.put(line)
        self._lines.put(None)  # EOF: the worker died

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self._proc = None

    def restart(self) -> None:
        self.stop()
        self.restarts += 1
        self.start()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------ protocol

    def _read_json(self, match, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                return None
            if line is None:
                return {"ok": False, "error": "Sandbox process crashed (possibly out of memory).",
                        "error_type": "SandboxCrash", "_dead": True}
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # stray output written straight to the real stdout; ignore it
            if isinstance(msg, dict) and match(msg):
                return msg

    def request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """Send one command and wait for its reply. Thread-safe; blocks the caller."""
        timeout = timeout or get_settings().code_timeout_seconds
        with self._lock:
            self.start()
            self._next_id += 1
            msg_id = self._next_id
            assert self._proc and self._proc.stdin
            try:
                self._proc.stdin.write(json.dumps({**payload, "id": msg_id}) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self.restart()
                return {"ok": False, "error": "Sandbox restarted after a crash; earlier variables are lost.",
                        "error_type": "SandboxCrash", "restarted": True}
            reply = self._read_json(lambda m: m.get("id") == msg_id or m.get("_dead"), timeout)
            if reply is None:
                self.restart()
                return {"ok": False, "error_type": "TimeoutError", "restarted": True,
                        "error": f"Execution exceeded the {timeout:.0f}s limit and was stopped. The kernel was "
                                 "restarted: the datasets are reloaded but variables from earlier steps are lost. "
                                 "Use a faster approach (vectorised pandas, aggregation, sampling)."}
            if reply.get("_dead"):
                self.restart()
                reply["restarted"] = True
            return reply

    # ------------------------------------------------------------ high-level API

    def execute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        reply = self.request({"cmd": "exec", "code": code}, timeout)
        for key, default in (("stdout", ""), ("error", None), ("error_type", None), ("value", None),
                             ("tables", []), ("charts", []), ("new_vars", []), ("duration_s", None)):
            reply.setdefault(key, default)
        return reply

    def load_dataset(self, name: str, parquet_path: Path) -> dict[str, Any]:
        self.datasets[name] = parquet_path
        if self.backend == "docker":
            return self.request({"cmd": "load", "name": name, "path": f"/data/{parquet_path.name}"}, 120)
        return self.request({"cmd": "load", "name": name, "path": str(parquet_path.resolve())}, 120)

    def variables(self) -> list[dict[str, Any]]:
        reply = self.request({"cmd": "vars"}, 15)
        return reply.get("variables", [])

    # async wrappers: the worker call blocks, so run it in a thread
    async def aexecute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.execute, code, timeout)

    async def avariables(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.variables)


class SandboxPool:
    """Keeps one sandbox per session id."""

    def __init__(self) -> None:
        self._boxes: dict[str, Sandbox] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str, session_dir: Path) -> Sandbox:
        with self._lock:
            box = self._boxes.get(session_id)
            if box is None:
                box = Sandbox(session_dir)
                self._boxes[session_id] = box
            return box

    def drop(self, session_id: str) -> None:
        with self._lock:
            box = self._boxes.pop(session_id, None)
        if box:
            box.stop()

    def shutdown(self) -> None:
        for sid in list(self._boxes):
            self.drop(sid)


pool = SandboxPool()
