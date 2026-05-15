from __future__ import annotations

import argparse
import logging

import pandas as pd

from config import load_config
from dataset_adapters import dataset_adapter_for, write_normalized_transcripts
from io_utils import read_parquet, write_parquet
from logging_utils import configure_logging
from schema import ensure_universal_schema

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normaliza metadados e labels no schema Unified.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)

    manifest_path = config.processed_root / "metadata" / "audio_manifest.parquet"
    if manifest_path.exists():
        audio_manifest = read_parquet(manifest_path)
        if "status" in audio_manifest.columns:
            audio_manifest = audio_manifest[audio_manifest["status"] == "ok"]
        if "duration_seconds" in audio_manifest.columns:
            durations = pd.to_numeric(audio_manifest["duration_seconds"], errors="coerce")
            before = len(audio_manifest)
            audio_manifest = audio_manifest[durations > 0]
            removed = before - len(audio_manifest)
            if removed:
                LOGGER.warning("Removidos %d audios com duracao zero ou invalida.", removed)
    else:
        LOGGER.warning("Manifesto de audio nao encontrado em %s; gerando apenas metadata textual.", manifest_path)
        audio_manifest = pd.DataFrame()

    datasets = config.datasets
    if args.dataset:
        if args.dataset not in datasets:
            raise KeyError(f"Dataset desconhecido: {args.dataset}")
        datasets = {args.dataset: datasets[args.dataset]}

    records: list[dict[str, object]] = []
    for dataset_key, dataset_config in datasets.items():
        adapter = dataset_adapter_for(config, dataset_config)
        dataset_records = adapter.build_unified_records(audio_manifest)
        LOGGER.info("%s: %d registros normalizados.", dataset_config.source_name, len(dataset_records))
        records.extend(dataset_records)

    metadata = ensure_universal_schema(pd.DataFrame(records))
    metadata = write_normalized_transcripts(metadata, config)
    output_path = config.processed_root / "metadata" / "metadata.parquet"
    write_parquet(metadata, output_path)
    labels_path = config.processed_root / "labels" / "labels.parquet"
    write_parquet(
        metadata[
            [
                "sample_id",
                "dataset_source",
                "participant_id",
                "phq_score",
                "gad_score",
                "depression_label",
                "anxiety_label",
                "emotion_label",
            ]
        ],
        labels_path,
    )
    LOGGER.info("Metadata unificada salva em %s (%d linhas).", output_path, len(metadata))
    LOGGER.info("Labels padronizados salvos em %s.", labels_path)


if __name__ == "__main__":
    main()
