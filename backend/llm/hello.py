"""Smoke test for the LLM wrapper. Run: python -m backend.llm.hello"""

import asyncio
import json
import logging

from backend.llm.client import chat, system_message, tool_result_message, user_message

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# A fake tool: the model can ask for a table's row count.
ROW_COUNT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_row_count",
        "description": "Return the number of rows in a table.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string", "description": "Name of the table"},
            },
            "required": ["table_name"],
        },
    },
}
FAKE_TABLES = {"sales": 12480, "customers": 873}


async def main() -> None:
    print("\n=== Test 1: plain text ===")
    r = await chat([user_message("Say hello in one short sentence.")])
    print("Reply:", r.text)

    print("\n=== Test 2: tool-calling round trip ===")
    messages = [
        system_message("You are a data assistant. Use tools to look up facts; never guess numbers."),
        user_message("How many rows are in the sales table?"),
    ]
    r = await chat(messages, tools=[ROW_COUNT_TOOL])

    if not r.has_tool_calls:
        print("Model answered without calling the tool:", r.text)
        return

    messages.append(r.message)  # the model's tool request goes into history
    for call in r.tool_calls:
        print(f"Model asked for: {call.name}({call.arguments})")
        count = FAKE_TABLES.get(call.arguments.get("table_name"), "table not found")
        messages.append(tool_result_message(call, json.dumps({"row_count": count})))

    final = await chat(messages, tools=[ROW_COUNT_TOOL])
    print("Final answer:", final.text)


if __name__ == "__main__":
    asyncio.run(main())
