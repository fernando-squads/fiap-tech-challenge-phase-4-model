from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


POSITIVE_VALUES = {
    "1",
    "true",
    "yes",
    "y",
    "positive",
    "pos",
    "depressed",
    "depression",
    "anxious",
    "anxiety",
    "case",
    "control_positive",
}

NEGATIVE_VALUES = {
    "0",
    "false",
    "no",
    "n",
    "negative",
    "neg",
    "not_depressed",
    "non_depressed",
    "healthy",
    "normal",
    "control",
}


def missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> str | None:
    if missing(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    text = re.sub(r"\s+", " ", text)
    return text


def parse_float(value: Any) -> float | None:
    if missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def parse_int_label(value: Any) -> int | None:
    if missing(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        if float(value) in {0.0, 1.0}:
            return int(value)
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if text in POSITIVE_VALUES:
        return 1
    if text in NEGATIVE_VALUES:
        return 0
    return None


def normalize_gender(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if normalized in {"f", "female", "woman", "mulher", "feminino"}:
        return "female"
    if normalized in {"m", "male", "man", "homem", "masculino"}:
        return "male"
    if normalized in {"non_binary", "nonbinary", "nb", "nao_binario"}:
        return "non_binary"
    if normalized in {"unknown", "unk", "na", "n_a", "not_informed"}:
        return "unknown"
    return normalized


def label_from_score(score: Any, threshold: float) -> int | None:
    parsed = parse_float(score)
    if parsed is None:
        return None
    return int(parsed >= threshold)


def standardize_binary_label(
    explicit_label: Any,
    score: Any,
    threshold: float,
) -> int | None:
    parsed_label = parse_int_label(explicit_label)
    if parsed_label is not None:
        return parsed_label
    return label_from_score(score, threshold)
