from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multimodal_transformer import MultimodalTransformerClassifier


def test_multimodal_transformer_checkpoint_loads_when_artifact_exists():
    model_dir = Path(__file__).resolve().parents[1] / "processed" / "models" / "multimodal_transformer"
    checkpoint_path = model_dir / "model.pt"
    if not checkpoint_path.exists():
        pytest.skip("Artefato multimodal ainda nao foi gerado.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MultimodalTransformerClassifier(
        input_dims=checkpoint["input_dims"],
        **checkpoint["model_hparams"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    assert checkpoint["target"] == "depression_label"
    assert set(checkpoint["input_dims"]) >= {"audio", "text"}
