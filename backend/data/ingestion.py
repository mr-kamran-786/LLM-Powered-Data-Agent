"""Data ingestion: turn an uploaded file into a clean, typed DataFrame (requirement F1).

Every dataset is normalised and stored as Parquet in the session folder. The sandbox
loads those Parquet files, so pandas and DuckDB inside the sandbox see identical data.
"""

from __future__ import annotations

import csv
import io
import json
import re
import warnings
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}
_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
_DATE_NAME_HINT = re.compile(r"(date|time|day|month|year|_at$|_on$|timestamp|period)", re.I)


class IngestionError(ValueError):
    """Raised when a file cannot be read as a table."""


def table_name_from_filename(filename: str, taken: set[str] | None = None) -> str:
    """'Sales Data 2025.csv' -> 'sales_data_2025'; a valid SQL and Python identifier."""
    stem = Path(filename).stem.lower()
    name = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "table"
    if name[0].isdigit():
        name = f"t_{name}"
    base, i = name, 2
    while taken and name in taken:
        name = f"{base}_{i}"
        i += 1
    return name


def detect_encoding(sample: bytes) -> str:
    for enc in _ENCODINGS:
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"  # never fails


def detect_delimiter(text_sample: str) -> str:
    try:
        return csv.Sniffer().sniff(text_sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_any(path: Path) -> pd.DataFrame:
    """Read CSV/TSV, Excel, Parquet or JSON into a DataFrame."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type '{ext}'. Use CSV, Excel, Parquet or JSON.")
    try:
        if ext in {".csv", ".tsv", ".txt"}:
            with open(path, "rb") as fh:
                sample = fh.read(64_000)
            encoding = detect_encoding(sample)
            text = sample.decode(encoding, errors="replace")
            delimiter = "\t" if ext == ".tsv" else detect_delimiter(text)
            return pd.read_csv(path, encoding=encoding, sep=delimiter, low_memory=False)
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(path)  # first sheet
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext == ".jsonl":
            return pd.read_json(path, lines=True)
        # .json: either records, columns, or a nested document
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            # {"data": [...]} style wrappers are common in API dumps
            lists = [v for v in payload.values() if isinstance(v, list)]
            payload = lists[0] if len(lists) == 1 else payload
        return pd.json_normalize(payload) if isinstance(payload, list) else pd.DataFrame(payload)
    except IngestionError:
        raise
    except Exception as exc:  # pandas raises many different types
        raise IngestionError(f"Could not read {path.name}: {exc}") from exc


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and make column names unique; keep them human-readable."""
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        name = str(c).strip() or "column"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    df.columns = cols
    return df


def parse_dates(df: pd.DataFrame, min_success: float = 0.9) -> pd.DataFrame:
    """Convert text columns that look like dates into datetime64."""
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        sample = non_null.astype(str).head(500)
        # Only try columns whose name or content suggests a date; avoids turning IDs into dates.
        looks_like_date = _DATE_NAME_HINT.search(str(col)) or sample.str.match(
            r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
        ).mean() > 0.8
        if not looks_like_date:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= min_success:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_datetime(s, errors="coerce", format="mixed")
    return df


def load_dataframe(path: Path) -> pd.DataFrame:
    df = read_any(path)
    if df.empty and len(df.columns) == 0:
        raise IngestionError(f"{path.name} contains no data.")
    df = clean_column_names(df)
    # Integers are deliberately not downcast: int8 * int8 overflows silently in generated code.
    return parse_dates(df)


def save_as_parquet(df: pd.DataFrame, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Mixed-type object columns break Parquet; store them as strings.
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_object_dtype(safe[col]):
            types = {type(v) for v in safe[col].dropna().head(1000)}
            if len(types) > 1 or (types and not types <= {str}):
                safe[col] = safe[col].astype("string")
    safe.to_parquet(target, index=False)
    return target


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
