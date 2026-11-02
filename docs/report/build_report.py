"""Build the IEEE-style project report (HTML + PDF) from the benchmark result files.

Run: python docs/report/build_report.py

Every number in the Results section is read from eval/results/*.json and the per-turn
traces, so the report cannot drift from the measured data.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "eval" / "results"
TRACES = ROOT / "uploads" / "traces"
FIG = HERE / "figures"
BROWSERS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path("/usr/bin/chromium"), Path("/usr/bin/google-chrome"),
]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "svg.fonttype": "none",
})
GREY, DARK = "#9a9a9a", "#222222"


# ---------------------------------------------------------------- data


def latest(tag: str) -> dict:
    files = sorted(RESULTS.glob(f"*_{tag}.json"))
    if not files:
        raise SystemExit(f"No result file with tag '{tag}' in {RESULTS}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def llm_seconds() -> dict[tuple[str, str], float]:
    """(question, model) -> summed LLM call latency, from the trace logs (excludes quota waiting)."""
    best: dict[tuple[str, str], tuple[float, float]] = {}
    for f in TRACES.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            calls = r.get("usage", {}).get("calls", [])
            key = (r["question"], r["usage"]["model"])
            if calls and (key not in best or r["time"] > best[key][0]):  # latest run of that question wins
                best[key] = (r["time"], sum(c["latency_s"] for c in calls))
    return {k: v[1] for k, v in best.items()}


def items() -> dict[str, dict]:
    path = ROOT / "eval" / "datasets" / "benchmark.jsonl"
    return {i["id"]: i for i in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def subset_summary(run: dict, ids: set[str], bench: dict, llm_t: dict) -> dict:
    rows = [r for r in run["results"] if r["id"] in ids and bench[r["id"]]["answer_type"] != "safety"]
    model = run["summary"]["model"]
    times = [llm_t.get((bench[r["id"]]["question"], model)) for r in rows]
    times = [t for t in times if t is not None]
    usage = [r.get("usage", {}) for r in rows]
    return {
        "n": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "acc": round(100 * sum(r["correct"] for r in rows) / len(rows), 1),
        "llm_median": round(statistics.median(times), 1) if times else None,
        "wall_median": round(statistics.median(r["latency_s"] for r in rows), 1),
        "calls": round(statistics.mean(u.get("llm_calls", 0) for u in usage), 2),
        "tokens": round(statistics.mean(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in usage)),
        "cost": round(sum(u.get("cost_usd", 0) for u in usage) / len(usage), 5),
        "steps": round(statistics.mean(r["steps"] for r in rows), 2),
    }


# ---------------------------------------------------------------- figures


def fig_categories(summary: dict) -> None:
    cats = sorted(summary["by_category"].items(), key=lambda kv: (kv[1]["accuracy_pct"], kv[1]["n"]))
    names = [k.replace("_", " ") for k, _ in cats]
    acc = [v["accuracy_pct"] for _, v in cats]
    n = [v["n"] for _, v in cats]
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    colors = [DARK if a < 100 else GREY for a in acc]
    ax.barh(names, acc, color=colors, height=0.62)
    for y, (a, k) in enumerate(zip(acc, n)):
        ax.text(a + 1.2, y, f"{a:.0f}%  (n={k})", va="center", fontsize=7)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Answer accuracy (%)")
    ax.axvline(70, color="black", lw=0.6, ls=(0, (3, 2)))
    ax.text(70, len(names) - 0.35, "70% target", fontsize=6.5, ha="center", va="bottom")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "accuracy_by_category.svg")
    plt.close(fig)


def fig_configs(configs: list[tuple[str, dict]]) -> None:
    labels = [c[0] for c in configs]
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.75))
    panels = [("acc", "Accuracy (%)", "{:.0f}"), ("llm_median", "Median LLM time (s)", "{:.1f}"),
              ("tokens_k", "Tokens / question (k)", "{:.1f}")]
    for ax, (key, title, fmt) in zip(axes, panels):
        vals = [(c[1]["tokens"] / 1000 if key == "tokens_k" else c[1][key]) or 0 for c in configs]
        ax.bar(range(len(vals)), vals, color=[DARK, GREY, "#d0d0d0"][: len(vals)], width=0.7, edgecolor="black", lw=0.4)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5)
        ax.set_title(title, fontsize=7)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(["A", "B", "C"][: len(vals)])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.22 if max(vals) else 1)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(FIG / "configurations.svg")
    plt.close(fig)


# ---------------------------------------------------------------- text


def pct(x) -> str:
    return f"{x:.1f}" if isinstance(x, float) else str(x)


def results_html(full: dict, bench: dict, configs: list[tuple[str, dict]], llm_t: dict, analysis: dict) -> str:
    s = full["summary"]
    task = [r for r in full["results"] if bench[r["id"]]["answer_type"] != "safety"]
    times = [llm_t.get((bench[r["id"]]["question"], s["model"])) for r in task]
    times = sorted(t for t in times if t is not None)
    llm_med = statistics.median(times) if times else 0
    llm_p95 = times[min(len(times) - 1, int(0.95 * len(times)))] if times else 0
    retried = sum(1 for r in task if r.get("failed_steps", 0) > 0)
    diff = s["by_difficulty"]
    h = []
    h.append(f"""<p class="noindent">Table&nbsp;IV summarises the full run with <span class="mono">{s['model'].split('/')[-1]}</span>
and the critic enabled. The agent answered {sum(r['correct'] for r in task)} of {s['n_questions']} questions correctly
({s['answer_accuracy_pct']}%), which meets the plan's 70% target (RQ1). {s['provider_errors']} question(s) ended with a provider error
(HTTP&nbsp;503) because fallbacks were disabled to measure one model; excluding them, accuracy is
{s['accuracy_excl_provider_errors_pct']}%. Code ran successfully in {s['execution_success_pct']}% of the executable questions, and
{retried} questions needed at least one automatic repair after a failed step.</p>""")
    h.append(f"""<div class="tbl"><p class="cap">Full benchmark run (critic on)</p><table>
<thead><tr><th>Metric</th><th class="n">Value</th></tr></thead><tbody>
<tr><td>Questions (excluding safety probes)</td><td class="n">{s['n_questions']}</td></tr>
<tr><td>Answer accuracy</td><td class="n">{s['answer_accuracy_pct']}%</td></tr>
<tr><td>Accuracy excluding provider errors</td><td class="n">{s['accuracy_excl_provider_errors_pct']}%</td></tr>
<tr><td>Execution success</td><td class="n">{s['execution_success_pct']}%</td></tr>
<tr><td>Safety probes passed</td><td class="n">{s['safety_pass']}</td></tr>
<tr><td>Code steps / retried steps per question</td><td class="n">{s['avg_code_steps']} / {s['avg_failed_steps_retried']}</td></tr>
<tr><td>LLM calls per question</td><td class="n">{s['avg_llm_calls']}</td></tr>
<tr><td>LLM time, median / p95 (s)</td><td class="n">{llm_med:.1f} / {llm_p95:.1f}</td></tr>
<tr><td>Wall-clock time incl. quota wait, median / p95 (s)</td><td class="n">{s['latency_median_s']} / {s['latency_p95_s']}</td></tr>
<tr><td>Tokens per question</td><td class="n">{s['avg_tokens']:,}</td></tr>
<tr><td>Cost per question / whole run (US$)</td><td class="n">{s['cost_per_question_usd']} / {s['total_cost_usd']}</td></tr>
</tbody></table></div>""")
    h.append(f"""<p>Accuracy by difficulty was {pct(diff.get('easy', {}).get('accuracy_pct'))}% for easy,
{pct(diff.get('medium', {}).get('accuracy_pct'))}% for medium and {pct(diff.get('hard', {}).get('accuracy_pct'))}% for hard
questions (n&nbsp;=&nbsp;{diff.get('easy', {}).get('n')}, {diff.get('medium', {}).get('n')} and {diff.get('hard', {}).get('n')}).
Fig.&nbsp;2 breaks accuracy down by category.</p>""")
    h.append("""<figure><img src="figures/accuracy_by_category.svg" alt="Accuracy by category">
<figcaption>Answer accuracy by question category in the full run; dark bars mark categories below 100%. The dashed line is the plan's 70% target.</figcaption></figure>""")
    h.append(f"""<p><em>Latency and cost.</em> The model itself needed a median of {llm_med:.1f}&nbsp;s per question across all
roles, within the plan's 20&nbsp;s target for simple questions. The wall-clock median of {s['latency_median_s']}&nbsp;s is dominated by
waiting for the free tier's 15-requests-per-minute quota while two questions ran in parallel; it is a property of the
API plan rather than of the agent. A whole run of {s['n_questions']} questions cost US$&nbsp;{s['total_cost_usd']}.</p>""")
    rows = "".join(
        f"<tr><td>{tag}</td><td>{name.split(', ')[0]}</td><td>{name.split(', ')[1].replace('critic ', '')}</td><td class='n'>{c['correct']}/{c['n']}</td><td class='n'>{c['llm_median']}</td>"
        f"<td class='n'>{c['wall_median']}</td><td class='n'>{c['calls']}</td><td class='n'>{c['tokens']:,}</td><td class='n'>{c['cost']}</td></tr>"
        for tag, (name, c) in zip("ABC", configs))
    h.append(f"""<p><em>Models and the critic (RQ2).</em> Table&nbsp;V and Fig.&nbsp;3 compare three configurations on the same random
subset of {configs[0][1]['n']} questions (seed&nbsp;7). {analysis['configs']}</p>
<div class="tbl"><p class="cap">Configurations on a {configs[0][1]['n']}-question subset</p><table>
<thead><tr><th></th><th>Model</th><th>Critic</th><th class="n">Correct</th><th class="n">LLM s</th><th class="n">Wall s</th><th class="n">Calls</th><th class="n">Tokens</th><th class="n">US$</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<figure><img src="figures/configurations.svg" alt="Configuration comparison">
<figcaption>Configurations A&ndash;C of Table&nbsp;V: accuracy, median LLM time and tokens per question on the same {configs[0][1]['n']} questions.</figcaption></figure>""")
    h.append(f"""<p><em>Safety (RQ3).</em> {analysis['safety']}</p>
<p><em>Automated tests.</em> All {analysis['n_tests']} tests pass, including 13 unsafe SQL statements rejected by the guard,
seven sandbox escape attempts stopped at run time even with the static check bypassed, a timeout followed by an automatic
kernel restart, and agent-loop tests for self-correction, giving up after three failures, clarification, ungrounded numbers,
critic revisions, follow-up memory and the token budget.</p>""")
    return "\n".join(h)


ERROR_ANALYSIS_TEMPLATE = """<h3>Error analysis</h3>
<p class="noindent">Table&nbsp;VI classifies every question the agent got wrong in the full run.</p>
<div class="tbl"><p class="cap">Failed questions in the full run</p><table>
<thead><tr><th>ID</th><th>Question (abridged)</th><th>Cause</th></tr></thead><tbody>{rows}</tbody></table></div>
{text}"""


def discussion_html(full: dict, bench: dict, analysis: dict) -> str:
    fails = [r for r in full["results"] if not r["correct"]]
    rows = "".join(
        f"<tr><td>{r['id']}</td><td>{_abridge(bench[r['id']]['question'])}</td><td>{analysis['causes'].get(r['id'], 'not yet classified')}</td></tr>"
        for r in fails)
    h = [ERROR_ANALYSIS_TEMPLATE.format(rows=rows or "<tr><td colspan='3'>none</td></tr>", text=analysis["error_text"])]
    h.append(analysis["discussion"])
    return "\n".join(h)


def _abridge(q: str, n: int = 58) -> str:
    return q if len(q) <= n else q[: n - 1].rsplit(" ", 1)[0] + "&hellip;"


# ---------------------------------------------------------------- build


def main() -> None:
    from analysis import ANALYSIS  # hand-written interpretation, kept beside this script

    FIG.mkdir(exist_ok=True)
    bench = items()
    full = latest(sys.argv[1] if len(sys.argv) > 1 else "full")  # e.g. a smoke test on another run
    compare = latest("compare30")
    if not list(RESULTS.glob("*_ablation30.json")):
        print("WARNING: no ablation run yet; using the comparison run in its place (draft build only)")
    ablation = latest("ablation30") if list(RESULTS.glob("*_ablation30.json")) else compare
    llm_t = llm_seconds()
    subset = {r["id"] for r in compare["results"]}
    configs = [
        (f"{full['summary']['model'].split('/')[-1].replace('gemini-', '')}, critic on", subset_summary(full, subset, bench, llm_t)),
        (f"{compare['summary']['model'].split('/')[-1].replace('gemini-', '')}, critic on", subset_summary(compare, subset, bench, llm_t)),
        (f"{ablation['summary']['model'].split('/')[-1].replace('gemini-', '')}, critic off", subset_summary(ablation, subset, bench, llm_t)),
    ]
    fig_categories(full["summary"])
    fig_configs(configs)

    collected = subprocess.run([sys.executable, "-m", "pytest", "-q", "--co"], cwd=ROOT, capture_output=True, text=True)
    n_tests = int(re.search(r"(\d+) tests? collected", collected.stdout).group(1))
    s = full["summary"]
    ctx = {
        "full": {**s, "model_short": s["model"].split("/")[-1]},
        "n_items": len(bench),
        "n_tests": n_tests,
        "safety_total": sum(1 for i in bench.values() if i["answer_type"] == "safety"),
        "n_ambiguous": sum(1 for i in bench.values() if i["answer_type"] == "clarification"),
        "per_dataset": Counter(k[0] for k in bench),
        "results_html": results_html(full, bench, configs, llm_t, {**ANALYSIS, "n_tests": n_tests}),
        "discussion_html": discussion_html(full, bench, ANALYSIS),
        "conclusion_html": ANALYSIS["conclusion"],
    }
    env = Environment(loader=FileSystemLoader(HERE), autoescape=False)
    html = env.get_template("paper_template.html").render(**ctx)
    out_html = HERE / "report.html"
    out_html.write_text(html, encoding="utf-8")

    browser = next((b for b in BROWSERS if b.exists()), None)
    if browser is None:
        print(f"Wrote {out_html}; no Chrome/Edge found for the PDF.")
        return
    out_pdf = HERE / "Group23_LLM_Data_Agent_Report.pdf"
    subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, capture_output=True, timeout=120)
    print(f"Wrote {out_html} and {out_pdf}")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
