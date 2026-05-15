from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from io_utils import resolve_project_path


EMBEDDING_COLUMNS: dict[str, str] = {
    "audio": "audio_embedding_path",
    "text": "text_embedding_path",
}

SAFE_METADATA_COLUMNS: list[str] = [
    "age",
    "duration_seconds",
]


@dataclass(frozen=True)
class MetadataStats:
    columns: list[str]
    means: dict[str, float]
    stds: dict[str, float]


def parse_modalities(value: str) -> list[str]:
    if value == "auto":
        return ["audio", "text"]
    modalities = [item.strip() for item in value.split(",") if item.strip()]
    valid = set(EMBEDDING_COLUMNS) | {"metadata"}
    unknown = [item for item in modalities if item not in valid]
    if unknown:
        raise ValueError(f"Modalidades desconhecidas: {unknown}. Use audio,text,metadata.")
    return modalities


def resolve_existing_path(path_value: object, project_root: Path) -> Path | None:
    path = resolve_project_path(path_value, project_root)
    if path is None or not path.exists():
        return None
    return path


def infer_embedding_dims(
    dataframes: list[pd.DataFrame],
    project_root: Path,
    modalities: list[str],
) -> dict[str, int]:
    dims: dict[str, int] = {}
    for modality in modalities:
        if modality not in EMBEDDING_COLUMNS:
            continue
        column = EMBEDDING_COLUMNS[modality]
        for df in dataframes:
            if column not in df.columns:
                continue
            for path_value in df[column].dropna().tolist():
                path = resolve_existing_path(path_value, project_root)
                if path is None:
                    continue
                dims[modality] = int(np.load(path, mmap_mode="r").reshape(-1).shape[0])
                break
            if modality in dims:
                break
        if modality not in dims:
            raise ValueError(f"Nenhum embedding valido encontrado para modalidade {modality}.")
    return dims


def fit_metadata_stats(df: pd.DataFrame, columns: list[str] | None = None) -> MetadataStats:
    selected_columns = [column for column in (columns or SAFE_METADATA_COLUMNS) if column in df.columns]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in selected_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        mean = float(values.mean()) if values.notna().any() else 0.0
        std = float(values.std(ddof=0)) if values.notna().sum() > 1 else 1.0
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        means[column] = mean
        stds[column] = std
    return MetadataStats(columns=selected_columns, means=means, stds=stds)


def _metadata_vector(row: pd.Series, stats: MetadataStats) -> np.ndarray:
    values: list[float] = []
    for column in stats.columns:
        raw_value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.isna(raw_value):
            raw_value = stats.means[column]
        values.append((float(raw_value) - stats.means[column]) / stats.stds[column])
    return np.asarray(values, dtype=np.float32)


def filter_rows_for_training(
    df: pd.DataFrame,
    project_root: Path,
    modalities: list[str],
    target: str,
    require_all_modalities: bool = False,
) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Coluna target nao encontrada: {target}")

    rows: list[dict[str, Any]] = []
    for _, row in df[df[target].notna()].iterrows():
        present: dict[str, bool] = {}
        for modality, column in EMBEDDING_COLUMNS.items():
            if modality not in modalities:
                continue
            present[modality] = resolve_existing_path(row.get(column), project_root) is not None
        requested_embedding_modalities = [item for item in modalities if item in EMBEDDING_COLUMNS]
        has_embedding = any(present.values()) if requested_embedding_modalities else True
        has_all = all(present.values()) if requested_embedding_modalities else True
        if require_all_modalities and not has_all:
            continue
        if not require_all_modalities and not has_embedding:
            continue
        rows.append(row.to_dict())
    return pd.DataFrame(rows)


class MultimodalEmbeddingDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        df: pd.DataFrame,
        project_root: Path,
        modalities: list[str],
        embedding_dims: dict[str, int],
        target: str,
        metadata_stats: MetadataStats | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.project_root = project_root
        self.modalities = modalities
        self.embedding_dims = embedding_dims
        self.target = target
        self.metadata_stats = metadata_stats

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        item: dict[str, Any] = {
            "sample_id": str(row.get("sample_id", index)),
            "label": torch.tensor(float(row[self.target]), dtype=torch.float32),
        }

        for modality, column in EMBEDDING_COLUMNS.items():
            if modality not in self.modalities:
                continue
            path = resolve_existing_path(row.get(column), self.project_root)
            if path is None:
                vector = np.zeros(self.embedding_dims[modality], dtype=np.float32)
                present = False
            else:
                vector = np.load(path).astype(np.float32).reshape(-1)
                present = True
            item[modality] = torch.from_numpy(vector)
            item[f"{modality}_present"] = torch.tensor(present, dtype=torch.bool)

        if "metadata" in self.modalities:
            if self.metadata_stats is None or not self.metadata_stats.columns:
                vector = np.zeros(1, dtype=np.float32)
                present = False
            else:
                vector = _metadata_vector(row, self.metadata_stats)
                present = bool(np.isfinite(vector).any())
            item["metadata"] = torch.from_numpy(vector)
            item["metadata_present"] = torch.tensor(present, dtype=torch.bool)

        return item

