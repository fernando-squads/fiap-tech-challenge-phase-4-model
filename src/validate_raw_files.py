from __future__ import annotations

import argparse
import logging
from pathlib import Path

from audio_utils import get_audio_duration_seconds
from config import load_config
from dataset_adapters import audio_files_for_dataset
from io_utils import as_project_path, list_files, write_json
from logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida presenca e legibilidade basica dos arquivos brutos.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--check-audio-readable", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)

    datasets = config.datasets
    if args.dataset:
        if args.dataset not in datasets:
            raise KeyError(f"Dataset desconhecido: {args.dataset}")
        datasets = {args.dataset: datasets[args.dataset]}

    report: dict[str, object] = {"datasets": {}, "total_audio_files": 0, "total_errors": 0}

    for dataset_key, dataset_config in datasets.items():
        audio_files = audio_files_for_dataset(dataset_config)
        metadata_files = list_files(dataset_config.raw_dir, dataset_config.metadata_globs)
        transcript_files = list_files(dataset_config.raw_dir, dataset_config.transcript_globs)
        errors: list[dict[str, str]] = []

        if not dataset_config.raw_dir.exists():
            errors.append({"path": str(dataset_config.raw_dir), "error": "raw_dir_not_found"})

        if args.check_audio_readable:
            for audio_path in audio_files:
                try:
                    get_audio_duration_seconds(audio_path)
                except Exception as exc:
                    errors.append(
                        {
                            "path": as_project_path(audio_path, config.project_root),
                            "error": str(exc),
                        }
                    )

        report["datasets"][dataset_key] = {
            "source_name": dataset_config.source_name,
            "raw_dir": as_project_path(dataset_config.raw_dir, config.project_root),
            "audio_files": len(audio_files),
            "metadata_files": len(metadata_files),
            "transcript_files": len(transcript_files),
            "sample_audio_files": [
                as_project_path(path, config.project_root) for path in audio_files[:5]
            ],
            "sample_metadata_files": [
                as_project_path(path, config.project_root) for path in metadata_files[:5]
            ],
            "sample_transcript_files": [
                as_project_path(path, config.project_root) for path in transcript_files[:5]
            ],
            "errors": errors,
        }
        report["total_audio_files"] = int(report["total_audio_files"]) + len(audio_files)
        report["total_errors"] = int(report["total_errors"]) + len(errors)
        LOGGER.info(
            "%s: audio=%d metadata=%d transcripts=%d errors=%d",
            dataset_config.source_name,
            len(audio_files),
            len(metadata_files),
            len(transcript_files),
            len(errors),
        )

    output_path = config.processed_root / "metadata" / "raw_validation_report.json"
    write_json(report, output_path)
    LOGGER.info("Relatorio de validacao salvo em %s.", output_path)


if __name__ == "__main__":
    main()

