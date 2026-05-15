from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def slugify(value: object, fallback: str = "item") -> str:
    text = normalize_column_name(value)
    return text or fallback


def sha1_short(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def as_project_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_project_path(path_value: object, project_root: Path) -> Path | None:
    if path_value is None or pd.isna(path_value):
        return None
    text = str(path_value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return project_root / path


def list_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    if not root.exists():
        return []
    files: dict[str, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                files[path.resolve().as_posix()] = path
    return sorted(files.values(), key=lambda item: item.as_posix())


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                if all(isinstance(value, list) for value in data.values()):
                    return pd.DataFrame(data)
                return pd.json_normalize(data)
    raise ValueError(f"Formato tabular nao suportado: {path}")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Nao foi possivel salvar Parquet. Instale pyarrow ou fastparquet."
        ) from exc


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo Parquet nao encontrado: {path}")
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Nao foi possivel ler Parquet. Instale pyarrow ou fastparquet."
        ) from exc


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
