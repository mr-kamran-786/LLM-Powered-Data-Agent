"""Sandbox worker: a stateful Python kernel that runs generated code.

Started by `sandbox.manager` as a separate process (locally) or inside a Docker
container with no network. It talks JSON lines over stdin/stdout:

    -> {"id": 1, "cmd": "exec", "code": "..."}
    <- {"id": 1, "ok": true, "stdout": "...", "tables": [...], "charts": [...], ...}

This file is deliberately self-contained (stdlib + data libraries only) so it can be
copied into a container image unchanged.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

WORK_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
DATA_DIR = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else WORK_DIR
MEMORY_MB = int(sys.argv[3]) if len(sys.argv) > 3 else 0

MAX_TABLE_ROWS = 50
MAX_STDOUT = 20_000
MAX_CHART_BYTES = 4_000_000

# ---------------------------------------------------------------- resource limits
if MEMORY_MB and sys.platform != "win32":
    import resource

    limit = MEMORY_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

os.chdir(WORK_DIR)
os.environ["MPLCONFIGDIR"] = str(WORK_DIR / ".mpl")

# ---------------------------------------------------------------- heavy imports (before the audit hook)
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import duckdb  # noqa: E402
from scipy import stats  # noqa: E402

pio.templates.default = "plotly_white"
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
plt.figure()  # warm the font cache while file access is still unrestricted
plt.close("all")

_PROTO_OUT = sys.stdout  # protocol channel; user prints go to a buffer instead
_PROTO_IN = sys.stdin

# ---------------------------------------------------------------- runtime audit hook


def _allowed_roots() -> tuple[list[str], list[str]]:
    import tempfile

    read_roots = {WORK_DIR, DATA_DIR, Path(tempfile.gettempdir())}
    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, *sys.path):
        if p:
            read_roots.add(Path(p).resolve())
    write_roots = {WORK_DIR, Path(tempfile.gettempdir())}
    norm = lambda paths: [os.path.normcase(str(p)) for p in paths]  # noqa: E731
    return norm(read_roots), norm(write_roots)


_READ_ROOTS, _WRITE_ROOTS = _allowed_roots()
_BLOCKED_EVENTS = (
    "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.startfile",
    "os.kill", "subprocess.Popen", "socket.connect", "socket.bind", "socket.getaddrinfo",
    "socket.gethostbyname", "socket.sendto", "socket.sendmsg", "urllib.Request", "webbrowser.open",
    "winreg.", "ctypes.call_function", "sys.setprofile", "sys.settrace",
)
_PATH_EVENTS = {"os.remove", "os.rmdir", "os.rename", "shutil.rmtree", "os.truncate", "os.chmod",
                "os.mkdir", "shutil.copyfile", "shutil.move"}


def _inside(path: object, roots: list[str]) -> bool:
    if isinstance(path, int):  # file descriptor
        return True
    if isinstance(path, bytes):
        path = path.decode(errors="replace")
    try:
        full = os.path.normcase(os.path.abspath(str(path)))
    except Exception:
        return False
    return any(full == r or full.startswith(r.rstrip("\\/") + os.sep) for r in roots)


def _audit(event: str, args: tuple) -> None:
    if event.startswith(_BLOCKED_EVENTS):
        raise PermissionError(f"Sandbox: '{event}' is blocked (no network, no processes).")
    if event == "open" and args:
        path, mode = args[0], (args[1] if len(args) > 1 else "r")
        writing = isinstance(mode, str) and any(c in mode for c in "wax+")
        if isinstance(args[2] if len(args) > 2 else 0, int) and not isinstance(mode, str):
            writing = bool((args[2] or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if path is None:
            return
        if writing and not _inside(path, _WRITE_ROOTS):
            raise PermissionError(f"Sandbox: writing to {path!r} is blocked; save files with a relative name.")
        if not writing and not _inside(path, _READ_ROOTS):
            raise PermissionError(f"Sandbox: reading {path!r} is blocked; use the preloaded DataFrames.")
    elif event in _PATH_EVENTS and args:
        for p in args[:2]:
            if isinstance(p, (str, bytes, os.PathLike)) and not _inside(p, _WRITE_ROOTS):
                raise PermissionError(f"Sandbox: '{event}' on {p!r} is blocked.")
    elif event in ("os.listdir", "os.scandir", "glob.glob") and args:
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)) and not _inside(p, _READ_ROOTS):
            raise PermissionError(f"Sandbox: listing {p!r} is blocked.")


# ---------------------------------------------------------------- namespace helpers

_artifacts: dict[str, list] = {"tables": [], "charts": []}
_datasets: dict[str, str] = {}  # table name -> parquet path


def _table_payload(obj, title: str | None = None) -> dict:
    df = obj.to_frame() if isinstance(obj, pd.Series) else obj
    shown = df.head(MAX_TABLE_ROWS)
    if not isinstance(shown.index, pd.RangeIndex):
        shown = shown.reset_index()
    shown.columns = [str(c) if not isinstance(c, tuple) else " / ".join(map(str, c)) for c in shown.columns]
    payload = json.loads(shown.to_json(orient="split", date_format="iso", default_handler=str))
    return {
        "title": title or "",
        "columns": payload["columns"],
        "rows": payload["data"],
        "shape": [int(df.shape[0]), len(payload["columns"])],  # columns as displayed (index included)
        "truncated": len(df) > MAX_TABLE_ROWS,
    }


def show_table(df, title: str | None = None):
    """Show a DataFrame/Series to the user as a result table (first 50 rows)."""
    _artifacts["tables"].append(_table_payload(df, title))
    return None


def show_chart(fig, title: str | None = None):
    """Show a Plotly figure (or matplotlib figure) to the user as a chart."""
    if isinstance(fig, go.Figure):
        if title and not fig.layout.title.text:
            fig.update_layout(title=title)
        spec = fig.to_json()
        if len(spec) > MAX_CHART_BYTES:
            raise ValueError("Chart is too large to display; aggregate or sample the data first (e.g. df.sample(5000)).")
        _artifacts["charts"].append({"kind": "plotly", "title": title or (fig.layout.title.text or ""), "spec": json.loads(spec)})
    elif hasattr(fig, "savefig"):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        _artifacts["charts"].append({"kind": "image", "title": title or "", "png": base64.b64encode(buf.getvalue()).decode()})
        plt.close(fig)
    else:
        raise TypeError("show_chart expects a Plotly figure or a matplotlib figure.")
    return None


go.Figure.show = lambda self, *a, **k: show_chart(self)  # fig.show() works as users expect

_duck = duckdb.connect(":memory:")
_duck.execute("SET enable_external_access = false")
_duck.execute("SET lock_configuration = true")


def sql(query: str) -> pd.DataFrame:
    """Run a read-only DuckDB SQL query over the loaded tables and DataFrame variables."""
    for name, value in list(NS.items()):
        if isinstance(value, pd.DataFrame) and name.isidentifier() and not name.startswith("_"):
            _duck.register(name, value)
    return _duck.execute(query).df()


NS: dict = {
    "__name__": "__sandbox__",
    "pd": pd, "np": np, "px": px, "go": go, "plt": plt, "stats": stats,
    "show_table": show_table, "show_chart": show_chart, "sql": sql,
}
_BASE_NAMES = set(NS)

# ---------------------------------------------------------------- execution


def _is_matplotlib(value) -> bool:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return type(value).__module__.startswith("matplotlib")


def _run(code: str) -> dict:
    _artifacts["tables"].clear()
    _artifacts["charts"].clear()
    out, err = io.StringIO(), io.StringIO()
    before = {k: id(v) for k, v in NS.items()}
    result: dict = {"ok": True, "stdout": "", "error": None, "error_type": None, "value": None}
    start = time.perf_counter()
    try:
        tree = ast.parse(code, mode="exec")
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)
        sys.stdout, sys.stderr = out, err
        try:
            exec(compile(tree, "<agent-code>", "exec"), NS)
            value = eval(compile(last_expr, "<agent-code>", "eval"), NS) if last_expr else None
        finally:
            sys.stdout, sys.stderr = _PROTO_OUT, sys.__stderr__
        if value is not None:
            if isinstance(value, (pd.DataFrame, pd.Series)):
                show_table(value)
            elif isinstance(value, go.Figure):
                show_chart(value)
            elif not _is_matplotlib(value):
                if isinstance(value, np.generic):  # numpy 2 reprs read "np.float64(13.0)"
                    value = value.item()
                result["value"] = repr(value)[:4000]
        for num in plt.get_fignums():  # matplotlib figures left open become PNG charts
            show_chart(plt.figure(num))
        plt.close("all")
    except BaseException as exc:  # noqa: BLE001 - report everything back to the agent
        sys.stdout, sys.stderr = _PROTO_OUT, sys.__stderr__
        plt.close("all")
        result["ok"] = False
        result["error_type"] = type(exc).__name__
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        # Keep only frames from the generated code, not from the worker itself.
        tb = [line for line in tb if 'File "' not in line or "<agent-code>" in line or "site-packages" in line]
        result["error"] = "".join(tb)[-3000:]
    result["stdout"] = (out.getvalue() + err.getvalue())[:MAX_STDOUT]
    result["tables"] = list(_artifacts["tables"])
    result["charts"] = list(_artifacts["charts"])
    result["new_vars"] = sorted(
        k for k, v in NS.items()
        if not k.startswith("_") and k not in _BASE_NAMES and before.get(k) != id(v)
        and not callable(v) and not isinstance(v, type(sys))
    )
    result["duration_s"] = round(time.perf_counter() - start, 3)
    return result


def _variables() -> list[dict]:
    out = []
    for k, v in NS.items():
        if k.startswith("_") or k in _BASE_NAMES or callable(v) or isinstance(v, type(sys)):
            continue
        if isinstance(v, pd.DataFrame):
            desc = f"DataFrame {v.shape[0]}x{v.shape[1]} columns={list(map(str, v.columns[:15]))}"
        elif isinstance(v, pd.Series):
            desc = f"Series len={len(v)} name={v.name!r}"
        elif isinstance(v, (int, float, str, bool, np.number)):
            desc = f"{type(v).__name__} = {repr(v)[:80]}"
        else:
            desc = type(v).__name__
        out.append({"name": k, "description": desc, "is_dataset": k in _datasets})
    return out


def _load(name: str, path: str) -> dict:
    NS[name] = pd.read_parquet(path)
    _datasets[name] = path
    return {"ok": True, "name": name, "shape": list(NS[name].shape)}


def _handle(msg: dict) -> dict:
    cmd = msg.get("cmd")
    if cmd == "exec":
        return _run(msg["code"])
    if cmd == "load":
        return _load(msg["name"], msg["path"])
    if cmd == "vars":
        return {"ok": True, "variables": _variables()}
    if cmd == "ping":
        return {"ok": True, "pid": os.getpid()}
    return {"ok": False, "error": f"unknown command {cmd!r}"}


def main() -> None:
    # Load datasets that already exist (e.g. after a restart), then lock down.
    for line in (msg for msg in sys.argv[4:]):
        name, _, path = line.partition("=")
        _load(name, path)
    sys.addaudithook(_audit)
    _PROTO_OUT.write(json.dumps({"ready": True, "pid": os.getpid()}) + "\n")
    _PROTO_OUT.flush()
    for raw in _PROTO_IN:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
            reply = _handle(msg)
        except Exception as exc:  # noqa: BLE001
            msg, reply = {}, {"ok": False, "error": f"worker error: {exc}"}
        reply["id"] = msg.get("id")
        _PROTO_OUT.write(json.dumps(reply, default=str) + "\n")
        _PROTO_OUT.flush()


if __name__ == "__main__":
    main()
