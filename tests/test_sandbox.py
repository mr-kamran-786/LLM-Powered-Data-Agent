"""Sandbox tests: execution, captured artifacts, statefulness, limits and security.

These start a real worker process (no LLM involved).
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.config import PROJECT_ROOT
from backend.sandbox.guard import check_code
from backend.sandbox.manager import Sandbox


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    d = tmp_path_factory.mktemp("session")
    (d / "data").mkdir()
    pd.DataFrame({"region": ["N", "S", "N"], "sales": [1.5, 2.0, 3.0]}).to_parquet(d / "data" / "sales.parquet")
    sb = Sandbox(d)
    sb.load_dataset("sales", d / "data" / "sales.parquet")
    yield sb
    sb.stop()


def test_stdout_tables_and_state(box):
    r = box.execute("total = sales.sales.sum()\nprint(total)\nsales.groupby('region').sales.sum()")
    assert r["ok"] and r["stdout"].strip() == "6.5"
    assert r["tables"][0]["shape"] == [2, 2]
    assert "total" in r["new_vars"]
    assert box.execute("total * 2")["value"] == "13.0"  # variables persist between calls


def test_charts_plotly_and_matplotlib(box):
    r = box.execute("fig = px.bar(sales, x='region', y='sales', title='t')\nshow_chart(fig, 'Sales')")
    assert r["ok"] and r["charts"][0]["kind"] == "plotly"
    r = box.execute("plt.plot([1, 2, 3])")
    assert r["ok"] and r["charts"][0]["kind"] == "image"


def test_sql_helper(box):
    r = box.execute("sql('SELECT region, SUM(sales) AS s FROM sales GROUP BY 1 ORDER BY 1')")
    assert r["ok"] and r["tables"][0]["rows"][0] == ["N", 4.5]


def test_errors_return_traceback(box):
    r = box.execute("sales['missing_column']")
    assert not r["ok"] and r["error_type"] == "KeyError" and "missing_column" in r["error"]


@pytest.mark.parametrize("code, fragment", [
    ("import socket\nsocket.create_connection(('example.com', 80))", "blocked"),
    ("import urllib.request\nurllib.request.urlopen('http://example.com')", "blocked"),
    ("import subprocess\nsubprocess.run(['echo', 'hi'])", "blocked"),
    ("import os\nos.system('echo hi')", "blocked"),
    (f"import pathlib\npathlib.Path({str(PROJECT_ROOT / '.env')!r}).read_text()", "blocked"),
    ("pd.read_csv('C:/Windows/win.ini' if __import__('sys').platform == 'win32' else '/etc/passwd')", "blocked"),
    ("sql(\"SELECT * FROM read_text('/etc/passwd')\")", ""),
])
def test_runtime_layer_blocks_escape(box, code, fragment):
    """The audit hook stops these even though the static check is bypassed (raw execute)."""
    r = box.execute(code)
    assert not r["ok"]
    assert fragment in (r["error"] or "")


def test_timeout_restarts_kernel_and_reloads_data(box):
    r = box.execute("while True: pass", timeout=3)
    assert not r["ok"] and r["error_type"] == "TimeoutError" and r["restarted"]
    assert box.execute("len(sales)")["value"] == "3"


def test_secrets_not_in_environment(box):
    r = box.execute("import os\nprint(sorted(k for k in os.environ if 'KEY' in k.upper() or 'SECRET' in k.upper()))")
    assert r["stdout"].strip() == "[]"


@pytest.mark.parametrize("code", [
    "import subprocess", "from socket import socket", "os.system('rm -rf /')", "eval('1+1')",
    "open('x.txt', 'w')", "().__class__.__bases__[0].__subclasses__()", "import shutil; shutil.rmtree('.')",
    "pd.read_csv('C:/Users/me/secret.csv')", "pd.read_csv('../../.env')", "import requests",
])
def test_static_guard_rejects(code):
    assert not check_code(code).ok


@pytest.mark.parametrize("code", [
    "df = sales.replace({'N': 'North'}).rename(columns={'sales': 'revenue'})",
    "x = [1, 2]; x.remove(1)",
    "s = '2025/01/02'.split('/')",
    "import sklearn.linear_model as lm",
    "from statsmodels.tsa.holtwinters import ExponentialSmoothing",
])
def test_static_guard_allows_normal_analysis_code(code):
    assert check_code(code).ok, check_code(code).reason
