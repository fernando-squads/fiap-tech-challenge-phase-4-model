from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def write_wav(path: Path, *, duration: float = 1.2, sr: int = 16_000, amplitude: float = 0.2) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sr)


@pytest.fixture()
def voice_risk_dataset(tmp_path: Path) -> Path:
    audio_path = tmp_path / "audio" / "sample.wav"
    write_wav(audio_path)
    dataset_path = tmp_path / "training_dataset.csv"
    dataset_path.write_text(
        "\n".join(
            [
                "audio_path,transcription,binary_risk,anxiety,postpartum_depression,hormonal_fatigue,domestic_violence",
                f"{audio_path},Consulta de rotina com boa rede de apoio.,0,0,0,0,0",
                f"{audio_path},Tenho ansiedade e medo constante.,1,1,0,0,0",
                f"{audio_path},Depois do parto estou triste e isolada.,1,0,1,0,0",
                f"{audio_path},Estou exausta com fadiga hormonal.,1,0,0,1,0",
                f"{audio_path},Ele me controla e houve violencia.,1,0,0,0,1",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_path
