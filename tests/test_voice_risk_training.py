from __future__ import annotations

from pathlib import Path

import joblib

from train_voice_risk import train_voice_risk_models
from voice_risk import VoiceRiskTrainingConfig


def test_train_voice_risk_models_generates_api_compatible_artifacts(
    voice_risk_dataset: Path,
    tmp_path: Path,
):
    output_dir = tmp_path / "models" / "voice_risk"

    metrics = train_voice_risk_models(
        dataset_path=voice_risk_dataset,
        output_dir=output_dir,
        random_state=7,
        config=VoiceRiskTrainingConfig(audio_sample_rate=16_000),
    )

    binary_path = output_dir / "binary_risk_model.joblib"
    multilabel_path = output_dir / "multilabel_risk_model.joblib"
    assert binary_path.exists()
    assert multilabel_path.exists()
    assert (output_dir / "voice_risk_metrics.json").exists()
    assert metrics["binary"]["rows"] == 5
    assert metrics["multilabel"]["rows"] == 5

    binary_artifact = joblib.load(binary_path)
    assert binary_artifact["task"] == "binary"
    assert binary_artifact["threshold"] == 0.65
    assert binary_artifact["sklearn_version"]
    assert binary_artifact["pipeline"].predict_proba([{}]).shape == (1, 2)

    multilabel_artifact = joblib.load(multilabel_path)
    assert multilabel_artifact["task"] == "multilabel"
    assert multilabel_artifact["threshold"] == 0.55
    assert multilabel_artifact["sklearn_version"]
    assert multilabel_artifact["labels"] == [
        "anxiety",
        "postpartum_depression",
        "hormonal_fatigue",
        "domestic_violence",
    ]
    assert multilabel_artifact["pipeline"].predict_proba([{}]).shape == (1, 4)
