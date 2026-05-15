from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from audio_utils import convert_to_wav_16k_mono, get_audio_duration_seconds
from config import load_config
from dataset_adapters import audio_files_for_dataset, build_sample_id, infer_participant_id
from io_utils import as_project_path, ensure_dir, read_parquet, write_parquet
from logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Converte audios brutos para WAV mono 16 kHz PCM.")
    parser.add_argument("--config", type=str, default=None, help="Caminho para config/datasets.yaml.")
    parser.add_argument("--dataset", type=str, default=None, help="Processa apenas um dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocessa WAVs existentes.")
    parser.add_argument("--fail-fast", action="store_true", help="Interrompe no primeiro erro.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    output_root = ensure_dir(config.processed_root / "audio")
    metadata_root = ensure_dir(config.processed_root / "metadata")

    rows: list[dict[str, object]] = []
    datasets = config.datasets
    if args.dataset:
        if args.dataset not in datasets:
            raise KeyError(f"Dataset desconhecido: {args.dataset}")
        datasets = {args.dataset: datasets[args.dataset]}

    for dataset_key, dataset_config in datasets.items():
        raw_audio_files = audio_files_for_dataset(dataset_config)
        LOGGER.info("%s: %d arquivos de audio encontrados.", dataset_config.source_name, len(raw_audio_files))

        for raw_path in tqdm(raw_audio_files, desc=f"audio:{dataset_key}"):
            sample_id = build_sample_id(dataset_key, raw_path, dataset_config.raw_dir)
            output_path = output_root / dataset_key / f"{sample_id}.wav"
            status = "ok"
            error_message = None
            duration_seconds = None
            try:
                convert_to_wav_16k_mono(raw_path, output_path, overwrite=args.overwrite)
                duration_seconds = get_audio_duration_seconds(output_path)
                if duration_seconds <= 0:
                    raise ValueError(f"Audio processado vazio: {output_path}")
            except Exception as exc:
                status = "error"
                error_message = str(exc)
                LOGGER.exception("Falha ao preparar audio %s", raw_path)
                if args.fail_fast:
                    raise

            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset_key": dataset_key,
                    "dataset_source": dataset_config.source_name,
                    "participant_id": infer_participant_id(raw_path),
                    "raw_audio_path": as_project_path(raw_path, config.project_root),
                    "audio_path": as_project_path(output_path, config.project_root)
                    if output_path.exists()
                    else None,
                    "duration_seconds": duration_seconds,
                    "status": status,
                    "error_message": error_message,
                }
            )

    manifest_path = metadata_root / "audio_manifest.parquet"
    manifest = pd.DataFrame(rows)
    if args.dataset and manifest_path.exists():
        previous_manifest = read_parquet(manifest_path)
        if "dataset_key" in previous_manifest.columns:
            previous_manifest = previous_manifest[
                previous_manifest["dataset_key"].astype(str) != args.dataset
            ]
            manifest = pd.concat([previous_manifest, manifest], ignore_index=True)
    write_parquet(manifest, manifest_path)
    LOGGER.info("Manifesto de audio salvo em %s (%d linhas).", manifest_path, len(manifest))


if __name__ == "__main__":
    main()
