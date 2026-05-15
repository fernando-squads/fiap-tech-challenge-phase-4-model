from __future__ import annotations

import argparse
import logging

import pandas as pd

from config import load_config
from io_utils import read_parquet, write_parquet
from logging_utils import configure_logging
from schema import ensure_universal_schema, write_schema_json

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta train/validation/test Parquet no schema Unified.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _merge_optional_manifest(df: pd.DataFrame, manifest_path, columns: list[str]) -> pd.DataFrame:
    if not manifest_path.exists():
        LOGGER.warning("Manifesto opcional nao encontrado: %s", manifest_path)
        return df
    manifest = read_parquet(manifest_path)
    selected = ["sample_id"] + [column for column in columns if column in manifest.columns]
    if len(selected) == 1:
        return df
    df = df.drop(columns=[column for column in columns if column in df.columns], errors="ignore")
    return df.merge(manifest[selected], on="sample_id", how="left")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)

    metadata = read_parquet(config.processed_root / "metadata" / "metadata.parquet")
    splits = read_parquet(config.processed_root / "metadata" / "splits.parquet")
    unified = metadata.merge(splits[["sample_id", "split"]], on="sample_id", how="left")
    unified = _merge_optional_manifest(
        unified,
        config.processed_root / "metadata" / "audio_embeddings.parquet",
        ["audio_embedding_path"],
    )
    unified = _merge_optional_manifest(
        unified,
        config.processed_root / "metadata" / "text_embeddings.parquet",
        ["text_embedding_path"],
    )
    split_values = unified["split"].copy()
    unified = ensure_universal_schema(unified)

    config.unified_root.mkdir(parents=True, exist_ok=True)
    for split_name, file_name in {
        "train": "train.parquet",
        "validation": "validation.parquet",
        "test": "test.parquet",
    }.items():
        split_df = unified[split_values == split_name]
        output_path = config.unified_root / file_name
        write_parquet(split_df, output_path)
        LOGGER.info("%s: %d linhas salvas em %s.", split_name, len(split_df), output_path)

    write_schema_json(
        config.unified_root / "schema.json",
        extra={
            "audio_embedding_model": config.embeddings.audio_model_name,
            "text_embedding_model": config.embeddings.text_model_name,
            "label_thresholds": config.label_thresholds,
        },
    )
    LOGGER.info("schema.json salvo em %s.", config.unified_root / "schema.json")


if __name__ == "__main__":
    main()
