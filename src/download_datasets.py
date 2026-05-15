from __future__ import annotations

import argparse
import logging

from config import load_config
from dataset_downloader import DatasetDownloader
from io_utils import write_json
from logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baixa datasets ausentes configurados no datasets.yaml.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset a baixar. Pode ser usado mais de uma vez.")
    parser.add_argument("--force", action="store_true", help="Baixa novamente mesmo se arquivos ja existirem.")
    parser.add_argument("--fail-fast", action="store_true", help="Interrompe no primeiro erro.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)

    downloader = DatasetDownloader(config)
    results = downloader.ensure_available(
        dataset_keys=args.dataset,
        force=args.force,
        fail_fast=args.fail_fast,
    )

    output_path = config.processed_root / "metadata" / "download_report.json"
    write_json(
        {
            "results": [
                {
                    "dataset_key": result.dataset_key,
                    "dataset_source": result.dataset_source,
                    "status": result.status,
                    "raw_dir": result.raw_dir,
                    "archive_path": result.archive_path,
                    "message": result.message,
                }
                for result in results
            ]
        },
        output_path,
    )

    summary = {result.status: 0 for result in results}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    LOGGER.info("Relatorio de download salvo em %s. Resumo: %s", output_path, summary)


if __name__ == "__main__":
    main()

