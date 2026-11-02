"""Loads versioned prompt templates from prompts/*.md (never inline strings, per the plan)."""

from __future__ import annotations

import re
from functools import lru_cache

from backend.config import get_settings

_VERSION = re.compile(r"<!--\s*prompt:\s*(\S+)\s*\|\s*version:\s*([\w.]+)\s*-->")


@lru_cache
def _load(name: str) -> str:
    return (get_settings().prompts_dir / f"{name}.md").read_text(encoding="utf-8")


def render(name: str, **values: str) -> str:
    """Fill {{placeholders}}; the version comment is stripped from what the model sees."""
    text = _VERSION.sub("", _load(name)).strip()
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value if value else "(none)")
    return text


def versions() -> dict[str, str]:
    """{'planner': '1.2', ...}: recorded with every benchmark run so results can be compared."""
    out = {}
    for path in sorted(get_settings().prompts_dir.glob("*.md")):
        m = _VERSION.search(path.read_text(encoding="utf-8"))
        if m:
            out[m.group(1)] = m.group(2)
    return out
