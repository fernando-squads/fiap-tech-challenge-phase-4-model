from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from constants import SCHEMA_DESCRIPTIONS, SCHEMA_DTYPES, UNIVERSAL_SCHEMA_COLUMNS
from io_utils import write_json


def ensure_universal_schema(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in UNIVERSAL_SCHEMA_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result = result[UNIVERSAL_SCHEMA_COLUMNS]

    for column, dtype in SCHEMA_DTYPES.items():
        if dtype == "float":
            result[column] = pd.to_numeric(result[column], errors="coerce")
        elif dtype == "integer_nullable":
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
        elif dtype == "string":
            result[column] = result[column].astype("string")

    return result


def schema_payload(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "Unified",
        "version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "columns": [
            {
                "name": column,
                "dtype": SCHEMA_DTYPES[column],
                "description": SCHEMA_DESCRIPTIONS[column],
            }
            for column in UNIVERSAL_SCHEMA_COLUMNS
        ],
        "label_conventions": {
            "depression_label": {"0": "negative", "1": "positive"},
            "anxiety_label": {"0": "negative", "1": "positive"},
        },
        "split_convention": "Splits sao gerados por participant_id para evitar data leakage.",
    }
    if extra:
        payload.update(extra)
    return payload


def write_schema_json(path: Path, extra: dict[str, Any] | None = None) -> None:
    write_json(schema_payload(extra), path)

