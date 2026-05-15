from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"


@dataclass(frozen=True)
class DownloadConfig:
    source_type: str = "direct_url"
    url: str | None = None
    hf_dataset_name: str | None = None
    hf_data_dir: str | None = None
    load_dataset_example: str | None = None
    filename: str | None = None
    sha256: str | None = None
    md5: str | None = None
    extract: bool = True
    auth_token_env: str | None = None
    expected_files: list[str] = field(default_factory=list)
    manual_instructions: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    source_name: str
    raw_dir: Path
    language: str | None = None
    audio_globs: list[str] = field(default_factory=list)
    metadata_globs: list[str] = field(default_factory=list)
    transcript_globs: list[str] = field(default_factory=list)
    column_aliases: dict[str, list[str]] = field(default_factory=dict)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    label_rules: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    codec: str = "pcm_s16le"
    extensions: list[str] = field(default_factory=lambda: [".wav", ".mp3", ".flac"])


@dataclass(frozen=True)
class SplitConfig:
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    stratify_by: str = "depression_label"


@dataclass(frozen=True)
class EmbeddingConfig:
    audio_model_name: str = "microsoft/wavlm-large"
    text_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    max_chunk_seconds: int = 30
    text_batch_size: int = 16


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    raw_root: Path
    processed_root: Path
    unified_root: Path
    audio: AudioConfig
    splits: SplitConfig
    embeddings: EmbeddingConfig
    label_thresholds: dict[str, dict[str, float]]
    datasets: dict[str, DatasetConfig]
    random_seed: int = 42


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    project_root = path.parent.parent if path.parent.name == "config" else PROJECT_ROOT

    paths = raw.get("paths", {})
    audio_raw = raw.get("audio", {})
    splits_raw = raw.get("splits", {})
    embeddings_raw = raw.get("embeddings", {})
    project_raw = raw.get("project", {})

    datasets: dict[str, DatasetConfig] = {}
    for key, dataset_raw in (raw.get("datasets", {}) or {}).items():
        raw_dir = _resolve_path(project_root, dataset_raw.get("raw_dir", f"raw/{key}"))
        download_raw = dataset_raw.get("download", {}) or {}
        datasets[key] = DatasetConfig(
            key=key,
            source_name=dataset_raw.get("source_name", key),
            raw_dir=raw_dir,
            language=dataset_raw.get("language"),
            audio_globs=list(dataset_raw.get("audio_globs") or []),
            metadata_globs=list(dataset_raw.get("metadata_globs") or []),
            transcript_globs=list(dataset_raw.get("transcript_globs") or []),
            column_aliases={
                name: list(values)
                for name, values in (dataset_raw.get("column_aliases", {}) or {}).items()
            },
            download=DownloadConfig(
                source_type=str(download_raw.get("source_type") or "direct_url"),
                url=download_raw.get("url"),
                hf_dataset_name=download_raw.get("hf_dataset_name"),
                hf_data_dir=download_raw.get("hf_data_dir"),
                load_dataset_example=download_raw.get("load_dataset_example"),
                filename=download_raw.get("filename"),
                sha256=download_raw.get("sha256"),
                md5=download_raw.get("md5"),
                extract=bool(download_raw.get("extract", True)),
                auth_token_env=download_raw.get("auth_token_env"),
                expected_files=list(download_raw.get("expected_files") or []),
                manual_instructions=download_raw.get("manual_instructions"),
            ),
            label_rules=dict(dataset_raw.get("label_rules") or {}),
        )

    return PipelineConfig(
        project_root=project_root,
        raw_root=_resolve_path(project_root, paths.get("raw_root", "raw")),
        processed_root=_resolve_path(project_root, paths.get("processed_root", "processed")),
        unified_root=_resolve_path(project_root, paths.get("unified_root", "unified")),
        audio=AudioConfig(
            sample_rate=int(audio_raw.get("sample_rate", 16000)),
            channels=int(audio_raw.get("channels", 1)),
            codec=str(audio_raw.get("codec", "pcm_s16le")),
            extensions=list(audio_raw.get("extensions", [".wav", ".mp3", ".flac"])),
        ),
        splits=SplitConfig(
            train_size=float(splits_raw.get("train_size", 0.70)),
            validation_size=float(splits_raw.get("validation_size", 0.15)),
            test_size=float(splits_raw.get("test_size", 0.15)),
            stratify_by=str(splits_raw.get("stratify_by", "depression_label")),
        ),
        embeddings=EmbeddingConfig(
            audio_model_name=str(
                (embeddings_raw.get("audio", {}) or {}).get(
                    "model_name", "microsoft/wavlm-large"
                )
            ),
            text_model_name=str(
                (embeddings_raw.get("text", {}) or {}).get(
                    "model_name", "sentence-transformers/all-mpnet-base-v2"
                )
            ),
            max_chunk_seconds=int(
                (embeddings_raw.get("audio", {}) or {}).get("max_chunk_seconds", 30)
            ),
            text_batch_size=int(
                (embeddings_raw.get("text", {}) or {}).get("batch_size", 16)
            ),
        ),
        label_thresholds=raw.get(
            "label_thresholds",
            {"depression": {"phq_score": 10}, "anxiety": {"gad_score": 10}},
        ),
        datasets=datasets,
        random_seed=int(project_raw.get("random_seed", 42)),
    )
