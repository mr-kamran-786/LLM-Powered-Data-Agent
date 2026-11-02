"""Static safety check for generated Python code.

This is the *second* layer of defence, not the only one: the sandbox worker also
installs a runtime audit hook (no network, no subprocesses, no file access outside
its folders), and in Docker mode the container has no network and a read-only data
mount. The static check exists to give the LLM a fast, readable error message.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

BLOCKED_MODULES = {
    "subprocess", "socket", "ssl", "requests", "urllib", "urllib3", "http", "httpx", "aiohttp",
    "ftplib", "smtplib", "telnetlib", "poplib", "imaplib", "paramiko", "ctypes", "cffi",
    "multiprocessing", "concurrent", "asyncio", "threading", "_thread", "signal", "pty", "shutil",
    "importlib", "pickle", "marshal", "shelve", "webbrowser", "winreg", "msvcrt", "builtins",
    "duckdb", "sqlite3", "psycopg", "psycopg2", "sqlalchemy", "code", "codeop", "gc", "inspect",
}
BLOCKED_CALLS = {
    "eval", "exec", "compile", "__import__", "globals", "locals", "breakpoint", "input", "help",
    "setattr", "delattr", "vars", "open",
}
# os.* / sys.* members that start processes or change the file system (checked only on `os.` / `sys.`,
# so pandas methods such as df.replace or df.rename stay allowed).
BLOCKED_OS_ATTRS = {
    "system", "popen", "spawnl", "spawnv", "execv", "execve", "execl", "fork", "kill", "startfile",
    "remove", "unlink", "rmdir", "removedirs", "rename", "replace", "chmod", "chown", "putenv",
    "environ", "getenv", "listdir", "scandir", "walk", "modules", "settrace", "setprofile", "exit",
}
# Escape hatches through object internals, blocked on any object.
BLOCKED_ATTRS = {
    "__subclasses__", "__globals__", "__builtins__", "__code__", "__closure__", "__bases__", "__mro__",
    "__loader__", "__spec__", "__import__", "f_globals", "f_locals", "f_back", "gi_frame", "tb_frame",
}
_SUSPICIOUS_PATH = re.compile(
    r"(^[a-zA-Z]:[\\/])"                                                     # C:\...
    r"|(^/(etc|proc|sys|home|root|var|usr|bin|dev|mnt|opt|srv|app|data|Users)\b)"  # /etc/...
    r"|(\.\.[\\/])|(^~[\\/])|(^\\\\)|(\.env\b)"                              # ../, ~/, UNC, .env
)


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""


def check_code(code: str) -> GuardResult:
    """Return GuardResult(ok=False, reason=...) for code that must not run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return GuardResult(False, f"SyntaxError: {exc.msg} (line {exc.lineno})")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_MODULES:
                    return GuardResult(False, f"Import of '{alias.name}' is not allowed in the sandbox.")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BLOCKED_MODULES:
                return GuardResult(False, f"Import from '{node.module}' is not allowed in the sandbox.")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_CALLS:
                return GuardResult(False, f"Calling {func.id}() is not allowed in the sandbox.")
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRS:
                return GuardResult(False, f"Access to '.{node.attr}' is not allowed in the sandbox.")
            if (isinstance(node.value, ast.Name) and node.value.id in {"os", "sys", "pathlib", "Path"}
                    and node.attr in BLOCKED_OS_ATTRS):
                return GuardResult(False, f"{node.value.id}.{node.attr} is not allowed in the sandbox.")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) < 300 and _SUSPICIOUS_PATH.search(node.value.strip()):
                return GuardResult(
                    False,
                    f"The path {node.value!r} points outside the sandbox. Use the preloaded DataFrames "
                    "and save files only with relative names.",
                )
    return GuardResult(True)
