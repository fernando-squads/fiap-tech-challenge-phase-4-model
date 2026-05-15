from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from audio_utils import load_audio_16k_mono
from config import load_config
from io_utils import as_project_path, ensure_dir, read_parquet, resolve_project_path, write_parquet
from logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrai embeddings de audio com microsoft/wavlm-large.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", help="Interrompe no primeiro audio invalido.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


@torch.inference_mode()
def embed_audio(
    audio_path: Path,
    feature_extractor: object,
    model: torch.nn.Module,
    device: torch.device,
    max_chunk_seconds: int,
) -> np.ndarray:
    waveform = load_audio_16k_mono(audio_path)
    sample_rate = 16000
    chunk_size = max_chunk_seconds * sample_rate
    chunks = torch.split(waveform, chunk_size) if waveform.numel() > chunk_size else (waveform,)
    vectors: list[torch.Tensor] = []
    for chunk in chunks:
        if chunk.numel() == 0:
            continue
        inputs = feature_extractor(
            chunk.cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        vector = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu()
        vectors.append(vector)
    if not vectors:
        raise ValueError(f"Audio vazio: {audio_path}")
    return torch.stack(vectors, dim=0).mean(dim=0).numpy().astype("float32")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)

    metadata_path = config.processed_root / "metadata" / "metadata.parquet"
    metadata = read_parquet(metadata_path)
    metadata = metadata[metadata["audio_path"].notna()]
    if args.limit:
        metadata = metadata.head(args.limit)

    model_name = args.model_name or config.embeddings.audio_model_name
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Carregando %s em %s.", model_name, device)

    from transformers import AutoFeatureExtractor, WavLMModel

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name).to(device)
    model.eval()

    output_root = ensure_dir(config.processed_root / "embeddings" / "audio")
    rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="audio_embeddings"):
        sample_id = str(row["sample_id"])
        source_path = resolve_project_path(row["audio_path"], config.project_root)
        if source_path is None or not source_path.exists():
            LOGGER.warning("Audio inexistente para %s: %s", sample_id, row["audio_path"])
            error_rows.append(
                {
                    "sample_id": sample_id,
                    "audio_path": row["audio_path"],
                    "error_message": "audio_path inexistente",
                }
            )
            continue
        output_path = output_root / f"{sample_id}.npy"
        try:
            if output_path.exists() and not args.overwrite:
                vector = np.load(output_path, mmap_mode="r")
                embedding_dim = int(vector.shape[-1])
            else:
                vector = embed_audio(
                    source_path,
                    feature_extractor,
                    model,
                    device,
                    config.embeddings.max_chunk_seconds,
                )
                np.save(output_path, vector)
                embedding_dim = int(vector.shape[-1])
        except Exception as exc:
            LOGGER.exception("Falha ao extrair embedding de audio para %s: %s", sample_id, source_path)
            error_rows.append(
                {
                    "sample_id": sample_id,
                    "audio_path": as_project_path(source_path, config.project_root),
                    "error_message": str(exc),
                }
            )
            if args.fail_fast:
                raise
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "audio_embedding_path": as_project_path(output_path, config.project_root),
                "audio_embedding_dim": embedding_dim,
                "audio_embedding_model": model_name,
            }
        )

    manifest = pd.DataFrame(rows)
    output_path = config.processed_root / "metadata" / "audio_embeddings.parquet"
    write_parquet(manifest, output_path)
    LOGGER.info("Manifesto de embeddings de audio salvo em %s (%d linhas).", output_path, len(manifest))
    if error_rows:
        errors_path = config.processed_root / "metadata" / "audio_embedding_errors.parquet"
        write_parquet(pd.DataFrame(error_rows), errors_path)
        LOGGER.warning(
            "%d audios falharam e foram registrados em %s.",
            len(error_rows),
            errors_path,
        )


if __name__ == "__main__":
    main()
