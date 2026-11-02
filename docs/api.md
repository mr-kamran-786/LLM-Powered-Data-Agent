# API reference

Interactive OpenAPI docs: <http://localhost:8000/docs>.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and active model |
| GET | `/api/config` | Models, defaults, prompt versions |
| POST | `/api/sessions` | Create a session `{title}` |
| GET | `/api/sessions` | List sessions |
| GET | `/api/sessions/{id}` | Session with datasets, connections, turns, usage |
| DELETE | `/api/sessions/{id}` | Delete a session and its files |
| POST | `/api/sessions/{id}/datasets` | Upload a file (multipart `file`) → dataset + profile |
| GET | `/api/sessions/{id}/datasets/{name}/profile` | Profile + cleaning suggestions |
| GET | `/api/sessions/{id}/datasets/{name}/preview?rows=100` | First rows |
| POST | `/api/sessions/{id}/connections` | `{kind: sqlite|duckdb|postgres, target, name?}` |
| POST | `/api/sessions/{id}/samples` | Load demo datasets |
| POST | `/api/sessions/{id}/chat` | `{message, model?, critic?}` → `text/event-stream` |
| GET | `/api/sessions/{id}/export?format=md|html|pdf|ipynb` | Download a report |

## Chat event stream

Each SSE message has `event: <type>` and a JSON `data:` payload.

| type | payload |
|---|---|
| `status` | `stage` (planning, coding, verifying, responding), `message` |
| `plan` | `intent`, `plan[] {step, goal}`, `expects_chart`, `needs_clarification` |
| `clarification` | `question` |
| `step_start` | `index`, `tool`, `purpose`, `code` |
| `step` | `step {type, purpose, code, ok, stdout, value, tables[], charts[], error, error_type, verifier[], duration_s}` |
| `verifier` | `message` (answer rejected: ungrounded numbers) |
| `critic` | `verdict` (accept/revise), `issues[]`, `suggestion` |
| `answer` | `answer_md`, `short_answer`, `key_findings[]`, `caveats`, `status` |
| `error` | `message` |
| `usage` | `llm_calls`, `input_tokens`, `output_tokens`, `cost_usd`, `latency_s`, `steps`, `failed_steps` |
| `done` | `turn_id`, `status` |

Tables are `{title, columns[], rows[][], shape[2], truncated}`; charts are
`{kind: "plotly", spec}` (Plotly JSON) or `{kind: "image", png}` (base64).
