# User guide

## 1. Start the application

```bash
uvicorn backend.main:app --port 8000
```

Open <http://localhost:8000>. Your analyses are kept between restarts.

## 2. Load data

* **Upload files**: drag CSV, TSV, Excel, Parquet or JSON files onto *Upload files* (up to
  500 MB each). Each file becomes a table named after the file (`Sales 2025.csv` → `sales_2025`).
* **Connect a database**: open *Connect a database*, choose SQLite, DuckDB or PostgreSQL, and
  enter a file path or a connection string such as `postgresql://reader:pass@host:5432/db`.
  The connection is always read-only.
* **Demo data**: *Load demo datasets* adds a retail sales file, taxi trips, an HR file and a
  small shop database.

Click a table in the sidebar to see its **profile** (types, missing values, distributions),
**cleaning suggestions** (duplicates, outliers, wrong types) and a **preview** of the rows.
*Ask agent to fix* turns a suggestion into a question.

## 3. Ask questions

Type a question and press Enter. While the agent works you see:

1. **Plan** – the steps it intends to take.
2. **Steps** – each piece of code it ran, with its output, tables and any errors. A red step
   was corrected automatically by a later one.
3. **Answer** – the result in plain language, with charts. Numbers come only from code that ran.

Tips:

* Name the measure and the period: *"total revenue by region in 2025"* is better than *"sales"*.
* Follow-up questions keep the context: *"now split that by category"*, *"only for Online"*.
* Ask for charts explicitly: *"plot…"*, *"show a bar chart of…"*.
* Statistics and ML work too: *"is the difference significant?"*, *"fit a regression"*,
  *"cluster customers into 3 groups"*, *"forecast the next 3 months"*.
* If the question is ambiguous the agent asks you first; answer in the chat.

## 4. Options

* **Model** – choose another LLM for the next question. If a model is overloaded the agent
  switches to a fallback automatically.
* **Critic** – a second model pass that checks each answer before you see it (slower, more
  reliable).
* The token counter shows usage and cost for the session; a budget cap stops runaway costs.

## 5. Export

*Export* downloads the whole analysis as an **HTML report** (interactive charts), **PDF**,
**Markdown** or a **Jupyter notebook** that re-runs the exact code.

## 6. Privacy

Only a profile and a few sample rows are sent to the LLM provider, never the full dataset.
For sensitive data use a local model (Ollama) via the model picker.
