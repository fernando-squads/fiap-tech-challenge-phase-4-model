#!/usr/bin/env python3
"""Generate minimal synthetic voice-risk audio and CSV data for development only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from config import load_config
from voice_risk_unified import materialize_voice_risk_raw_dataset
from voice_risk import OUTPUT_COLUMNS


SAMPLES = [
    {
        "file": "synthetic_001.wav",
        "transcription": "Estou bem, dormindo melhor e sem preocupacoes importantes.",
        "binary_risk": 0,
        "anxiety": 0,
        "postpartum_depression": 0,
        "hormonal_fatigue": 0,
        "domestic_violence": 0,
        "amp": 0.22,
        "pauses": [0.15],
    },
    {
        "file": "synthetic_002.wav",
        "transcription": "Tenho ansiedade, fico ansiosa e meu coracao acelerado aparece em crise.",
        "binary_risk": 1,
        "anxiety": 1,
        "postpartum_depression": 0,
        "hormonal_fatigue": 0,
        "domestic_violence": 0,
        "amp": 0.16,
        "pauses": [0.55, 0.8],
    },
    {
        "file": "synthetic_003.wav",
        "transcription": "Depois do parto me sinto triste, sozinha e sem esperanca com o bebe.",
        "binary_risk": 1,
        "anxiety": 0,
        "postpartum_depression": 1,
        "hormonal_fatigue": 0,
        "domestic_violence": 0,
        "amp": 0.09,
        "pauses": [0.9, 1.1],
    },
    {
        "file": "synthetic_004.wav",
        "transcription": "Estou exausta, com fadiga hormonal, sem energia e nao durmo.",
        "binary_risk": 1,
        "anxiety": 0,
        "postpartum_depression": 0,
        "hormonal_fatigue": 1,
        "domestic_violence": 0,
        "amp": 0.06,
        "pauses": [1.2],
    },
    {
        "file": "synthetic_005.wav",
        "transcription": "Tenho medo de voltar para casa, ele me controla e ja me empurrou.",
        "binary_risk": 1,
        "anxiety": 1,
        "postpartum_depression": 0,
        "hormonal_fatigue": 0,
        "domestic_violence": 1,
        "amp": 0.11,
        "pauses": [0.7, 1.4],
    },
    {
        "file": "synthetic_006.wav",
        "transcription": "Consulta de rotina, sem tristeza, sem medo e com boa rede de apoio.",
        "binary_risk": 0,
        "anxiety": 0,
        "postpartum_depression": 0,
        "hormonal_fatigue": 0,
        "domestic_violence": 0,
        "amp": 0.20,
        "pauses": [0.1],
    },
    {
        "file": "synthetic_007.wav",
        "transcription": "Estou nervosa, isolada, cansada e nao tenho ajuda desde o parto.",
        "binary_risk": 1,
        "anxiety": 1,
        "postpartum_depression": 1,
        "hormonal_fatigue": 1,
        "domestic_violence": 0,
        "amp": 0.08,
        "pauses": [0.6, 1.0],
    },
    {
        "file": "synthetic_008.wav",
        "transcription": "Ele ameaca, gritou, me vigia e nao me sinto segura.",
        "binary_risk": 1,
        "anxiety": 1,
        "postpartum_depression": 0,
        "hormonal_fatigue": 0,
        "domestic_violence": 1,
        "amp": 0.10,
        "pauses": [1.3, 0.6],
    },
]


def make_signal(sample_index: int, amplitude: float, pauses: list[float], sr: int) -> np.ndarray:
    base_frequency = 180 + sample_index * 9
    tone_segments = []
    for idx, pause_duration in enumerate(pauses + [0.2]):
        duration = 0.55 + idx * 0.1
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        tone = amplitude * np.sin(2 * np.pi * (base_frequency + idx * 18) * t)
        silence = np.zeros(int(sr * pause_duration))
        tone_segments.extend([tone, silence])
    signal = np.concatenate(tone_segments)
    noise = np.random.default_rng(sample_index).normal(0, amplitude * 0.015, signal.shape)
    return np.asarray(signal + noise, dtype=np.float32)


def generate(output_csv: Path, sample_rate: int) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = output_csv.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, sample in enumerate(SAMPLES, start=1):
        audio_path = audio_dir / sample["file"]
        signal = make_signal(index, float(sample["amp"]), list(sample["pauses"]), sample_rate)
        sf.write(audio_path, signal, sample_rate)

        rows.append(
            {
                "audio_path": str(audio_path),
                "transcription": sample["transcription"],
                "binary_risk": sample["binary_risk"],
                "anxiety": sample["anxiety"],
                "postpartum_depression": sample["postpartum_depression"],
                "hormonal_fatigue": sample["hormonal_fatigue"],
                "domestic_violence": sample["domestic_violence"],
            }
        )

    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset sintetico de voice-risk gerado em {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--skip-raw-materialization", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or config.processed_root / "voice_risk" / "synthetic_dataset.csv"
    raw_output_dir = args.raw_output_dir or config.raw_root / "voice_risk_synthetic"
    sample_rate = args.sample_rate or config.audio.sample_rate
    generate(output, sample_rate)
    if not args.skip_raw_materialization:
        metadata_path = materialize_voice_risk_raw_dataset(
            source_csv=output,
            raw_output_dir=raw_output_dir,
            dataset_key="voice_risk_synthetic",
        )
        print(f"Dataset raw Unified: {metadata_path}")


if __name__ == "__main__":
    main()
