from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from train_anomaly_detector import (
    build_anomaly_matrix,
    fit_metadata_fill_values,
    train_anomaly_detector,
)


def _write_embedding(path: Path, values: list[float]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.float32))
    return path.relative_to(path.parents[2]).as_posix()


def test_anomaly_matrix_uses_presence_flags_and_metadata(tmp_path: Path):
    audio_path = _write_embedding(tmp_path / "processed" / "embeddings" / "a.npy", [1, 2, 3])
    text_path = _write_embedding(tmp_path / "processed" / "embeddings" / "t.npy", [4, 5])
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "audio_embedding_path": audio_path,
                "text_embedding_path": text_path,
                "age": 31,
                "duration_seconds": 6.5,
            },
            {
                "sample_id": "s2",
                "audio_embedding_path": audio_path,
                "text_embedding_path": None,
                "age": None,
                "duration_seconds": 4.0,
            },
        ]
    )
    metadata_columns = ["age", "duration_seconds"]
    fill_values = fit_metadata_fill_values(df, metadata_columns)

    result = build_anomaly_matrix(
        df=df,
        project_root=tmp_path,
        modalities=["audio", "text", "metadata"],
        embedding_dims={"audio": 3, "text": 2},
        metadata_columns=metadata_columns,
        metadata_fill_values=fill_values,
        require_all_modalities=False,
    )

    assert result.x.shape == (2, 9)
    assert result.skipped_rows == 0
    assert result.x[0, 3] == 1.0
    assert result.x[0, 6] == 1.0
    assert result.x[1, 6] == 0.0
    assert result.x[1, 7] == fill_values["age"]


def test_train_anomaly_detector_writes_self_contained_artifact(tmp_path: Path):
    rows = []
    for index in range(8):
        audio_path = _write_embedding(
            tmp_path / "processed" / "embeddings" / f"audio_{index}.npy",
            [float(index), float(index + 1), float(index + 2)],
        )
        text_path = _write_embedding(
            tmp_path / "processed" / "embeddings" / f"text_{index}.npy",
            [float(index) / 10.0, float(index + 1) / 10.0],
        )
        rows.append(
            {
                "sample_id": f"s{index}",
                "participant_id": f"p{index}",
                "audio_embedding_path": audio_path,
                "text_embedding_path": text_path,
                "age": 25 + index,
                "duration_seconds": 5.0 + index,
                "depression_label": int(index >= 6),
            }
        )

    unified_root = tmp_path / "unified"
    unified_root.mkdir()
    pd.DataFrame(rows[:5]).to_parquet(unified_root / "train.parquet", index=False)
    pd.DataFrame(rows[5:7]).to_parquet(unified_root / "validation.parquet", index=False)
    pd.DataFrame(rows[7:]).to_parquet(unified_root / "test.parquet", index=False)

    config = SimpleNamespace(
        project_root=tmp_path,
        unified_root=unified_root,
        processed_root=tmp_path / "processed",
        random_seed=42,
        embeddings=SimpleNamespace(
            audio_model_name="microsoft/wavlm-large",
            text_model_name="sentence-transformers/all-mpnet-base-v2",
        ),
    )
    args = Namespace(
        modalities="audio,text,metadata",
        metadata_columns="age,duration_seconds",
        require_all_modalities=True,
        contamination=0.2,
        threshold=None,
        n_estimators=20,
        max_samples="auto",
        pca_components=0,
        normal_label_column=None,
        normal_label_value="0",
        target_eval_column="depression_label",
        n_jobs=1,
        seed=123,
    )

    paths = train_anomaly_detector(config, args)
    artifact = joblib.load(paths["artifact"])

    assert paths["metrics"].exists()
    assert paths["config"].exists()
    assert artifact["task"] == "unsupervised_anomaly_detection"
    assert artifact["model_type"] == "isolation_forest"
    assert artifact["feature_count"] == 9
    assert 0.0 <= artifact["threshold"] <= 1.0
