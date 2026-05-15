from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import load_config
from io_utils import as_project_path, ensure_dir, read_parquet, write_parquet
from labels import clean_text
from logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrai embeddings de texto com all-mpnet-base-v2.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)

    metadata = read_parquet(config.processed_root / "metadata" / "metadata.parquet")
    metadata = metadata[metadata["transcript"].notna()].copy()
    metadata["transcript"] = metadata["transcript"].map(clean_text)
    metadata = metadata[metadata["transcript"].notna()]
    if args.limit:
        metadata = metadata.head(args.limit)

    model_name = args.model_name or config.embeddings.text_model_name
    LOGGER.info("Carregando %s.", model_name)
    from sentence_transformers import SentenceTransformer

    model_kwargs = {}
    if args.device:
        model_kwargs["device"] = args.device
    model = SentenceTransformer(model_name, **model_kwargs)

    output_root = ensure_dir(config.processed_root / "embeddings" / "text")
    rows: list[dict[str, object]] = []
    batch_size = config.embeddings.text_batch_size

    pending: list[tuple[str, str, object]] = []
    for _, row in metadata.iterrows():
        sample_id = str(row["sample_id"])
        output_path = output_root / f"{sample_id}.npy"
        if output_path.exists() and not args.overwrite:
            vector = np.load(output_path, mmap_mode="r")
            rows.append(
                {
                    "sample_id": sample_id,
                    "text_embedding_path": as_project_path(output_path, config.project_root),
                    "text_embedding_dim": int(vector.shape[-1]),
                    "text_embedding_model": model_name,
                }
            )
        else:
            pending.append((sample_id, str(row["transcript"]), output_path))

    for start in tqdm(range(0, len(pending), batch_size), desc="text_embeddings"):
        batch = pending[start : start + batch_size]
        texts = [item[1] for item in batch]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype("float32")
        for (sample_id, _, output_path), vector in zip(batch, vectors, strict=True):
            np.save(output_path, vector)
            rows.append(
                {
                    "sample_id": sample_id,
                    "text_embedding_path": as_project_path(output_path, config.project_root),
                    "text_embedding_dim": int(vector.shape[-1]),
                    "text_embedding_model": model_name,
                }
            )

    manifest = pd.DataFrame(rows)
    output_path = config.processed_root / "metadata" / "text_embeddings.parquet"
    write_parquet(manifest, output_path)
    LOGGER.info("Manifesto de embeddings de texto salvo em %s (%d linhas).", output_path, len(manifest))


if __name__ == "__main__":
    main()

