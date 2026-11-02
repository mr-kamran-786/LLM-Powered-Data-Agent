"""Tool schemas the LLM can call (plan, section 4).

Plain JSON schemas in the OpenAI function-calling format, so they work with every
provider LiteLLM supports (Gemini, Claude, GPT, Ollama models with tool support).
"""

from __future__ import annotations


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


RUN_PYTHON = _tool(
    "run_python",
    "Execute Python code in the session's stateful sandbox. Datasets are preloaded as pandas "
    "DataFrames named after their table. Available: pd, np, px (plotly.express), go, plt, stats "
    "(scipy.stats), sql(query)->DataFrame, show_table(df, title), show_chart(fig, title). scikit-learn "
    "and statsmodels can be imported. Variables persist between calls. print() what you need to see; "
    "the value of the last line is displayed automatically.",
    {
        "purpose": {"type": "string", "description": "One short sentence: what this step does."},
        "code": {"type": "string", "description": "Python code to execute."},
    },
    ["purpose", "code"],
)

RUN_SQL = _tool(
    "run_sql",
    "Execute ONE read-only SELECT query. target='files' queries the uploaded datasets with DuckDB SQL; "
    "otherwise target is the name of a connected database and its own dialect applies. The result is "
    "stored as a DataFrame variable `result_name` in the Python sandbox so you can continue in pandas.",
    {
        "purpose": {"type": "string", "description": "One short sentence: what this query computes."},
        "query": {"type": "string", "description": "A single SELECT statement."},
        "target": {"type": "string", "description": "'files' or the database connection name."},
        "result_name": {"type": "string", "description": "Python variable name for the result, e.g. 'q1'."},
    },
    ["purpose", "query", "target", "result_name"],
)

DESCRIBE_TABLE = _tool(
    "describe_table",
    "Return the full profile of one table: columns, types, missing values, distributions and sample rows.",
    {"name": {"type": "string", "description": "Table name."}},
    ["name"],
)

ASK_USER = _tool(
    "ask_user",
    "Pause and ask the user a clarifying question. Use only when the request is genuinely ambiguous "
    "and a reasonable default would likely be wrong.",
    {"question": {"type": "string", "description": "The question for the user, with 2-3 concrete options."}},
    ["question"],
)

FINAL_ANSWER = _tool(
    "final_answer",
    "Finish the analysis. Call this only after the code that produced every number has run successfully.",
    {
        "answer": {
            "type": "string",
            "description": "The direct answer in as few words as possible: a number, a name, a short list "
                           "(comma-separated) or 'see chart'. Values must be copied from executed output.",
        },
        "key_findings": {
            "type": "array", "items": {"type": "string"},
            "description": "2-5 findings, each citing exact values from the executed output.",
        },
        "caveats": {"type": "string", "description": "Assumptions or data-quality caveats; empty if none."},
    },
    ["answer", "key_findings"],
)

AGENT_TOOLS = [RUN_PYTHON, RUN_SQL, DESCRIBE_TABLE, ASK_USER, FINAL_ANSWER]
