from __future__ import annotations

from pathlib import Path

from clean_generated_data import build_clean_targets, clean_targets
from config import AudioConfig, EmbeddingConfig, PipelineConfig, SplitConfig


def _pipeline_config(project_root: Path) -> PipelineConfig:
    return PipelineConfig(
        project_root=project_root,
        raw_root=project_root / "raw",
        processed_root=project_root / "processed",
        unified_root=project_root / "unified",
        audio=AudioConfig(),
        splits=SplitConfig(),
        embeddings=EmbeddingConfig(),
        label_thresholds={},
        datasets={},
    )


def test_clean_generated_data_preserves_gitkeep_and_downloaded_raw(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    downloaded_raw = config.raw_root / "womanhealthfiap" / "dataset.csv"
    downloaded_raw.parent.mkdir(parents=True)
    downloaded_raw.write_text("id\n1\n", encoding="utf-8")

    generated_paths = [
        config.raw_root / "voice_risk_training" / "metadata.csv",
        config.raw_root / "voice_risk_synthetic" / "metadata.csv",
        config.processed_root / "voice_risk" / "training_dataset.csv",
        config.processed_root / "audio" / "womanhealthfiap" / "sample.wav",
        config.processed_root / "metadata" / "metadata.parquet",
        config.processed_root / "labels" / "labels.parquet",
        config.processed_root / "embeddings" / "audio" / "sample.npy",
        config.processed_root / "models" / "depression_baseline.joblib",
        config.unified_root / "train.parquet",
        config.unified_root / "schema.json",
    ]
    for path in generated_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated", encoding="utf-8")

    for keep_dir in [
        config.processed_root / "audio",
        config.processed_root / "metadata",
        config.processed_root / "labels",
        config.processed_root / "embeddings",
        config.processed_root / "models",
    ]:
        keep_dir.mkdir(parents=True, exist_ok=True)
        (keep_dir / ".gitkeep").write_text("", encoding="utf-8")

    targets = build_clean_targets(config)
    dry_run = clean_targets(targets, config, dry_run=True)
    assert dry_run.planned > 0
    assert all(path.exists() for path in generated_paths)

    summary = clean_targets(targets, config, dry_run=False)
    assert summary.removed > 0
    assert downloaded_raw.exists()
    assert not (config.raw_root / "voice_risk_training").exists()
    assert not (config.raw_root / "voice_risk_synthetic").exists()
    assert not (config.processed_root / "voice_risk").exists()
    assert not (config.unified_root / "train.parquet").exists()
    assert not (config.unified_root / "schema.json").exists()

    for keep_dir in [
        config.processed_root / "audio",
        config.processed_root / "metadata",
        config.processed_root / "labels",
        config.processed_root / "embeddings",
        config.processed_root / "models",
    ]:
        assert (keep_dir / ".gitkeep").exists()
