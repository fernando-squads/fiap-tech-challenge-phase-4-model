from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_config
from io_utils import ensure_dir, read_parquet
from logging_utils import configure_logging
from multimodal_data import (
    MultimodalEmbeddingDataset,
    filter_rows_for_training,
    fit_metadata_stats,
    infer_embedding_dims,
    parse_modalities,
)
from multimodal_transformer import MultimodalTransformerClassifier

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina um transformer multimodal sobre embeddings de audio/texto/metadados."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--target", type=str, default="depression_label")
    parser.add_argument("--modalities", type=str, default="audio,text")
    parser.add_argument("--require-all-modalities", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def positive_class_weight(labels: pd.Series) -> torch.Tensor | None:
    positives = int((labels.astype(int) == 1).sum())
    negatives = int((labels.astype(int) == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    return torch.tensor([negatives / positives], dtype=torch.float32)


def binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = None
    return metrics


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> dict[str, float | None]:
    model.eval()
    losses: list[float] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch)
        label = batch["label"]
        loss = criterion(logits, label)
        losses.append(float(loss.detach().cpu()))
        labels.append(label.detach().cpu().numpy())
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())

    if not labels:
        raise ValueError("DataLoader vazio durante avaliacao.")
    y_true = np.concatenate(labels).astype(int)
    y_prob = np.concatenate(probabilities)
    metrics = binary_metrics(y_true, y_prob, threshold)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def build_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    project_root: Path,
    modalities: list[str],
    target: str,
    require_all_modalities: bool,
) -> tuple[
    MultimodalEmbeddingDataset,
    MultimodalEmbeddingDataset,
    MultimodalEmbeddingDataset,
    dict[str, int],
    dict[str, Any],
]:
    train_filtered = filter_rows_for_training(
        train_df,
        project_root,
        modalities,
        target,
        require_all_modalities=require_all_modalities,
    )
    validation_filtered = filter_rows_for_training(
        validation_df,
        project_root,
        modalities,
        target,
        require_all_modalities=require_all_modalities,
    )
    test_filtered = filter_rows_for_training(
        test_df,
        project_root,
        modalities,
        target,
        require_all_modalities=require_all_modalities,
    )

    if train_filtered.empty or validation_filtered.empty:
        raise ValueError(
            "Treino e validacao precisam conter labels e pelo menos uma modalidade. "
            f"train={len(train_filtered)}, validation={len(validation_filtered)}."
        )
    if train_filtered[target].nunique() < 2:
        raise ValueError(
            f"O treino precisa de duas classes para {target}; "
            f"classes={train_filtered[target].value_counts(dropna=False).to_dict()}."
        )

    embedding_dims = infer_embedding_dims(
        [train_filtered, validation_filtered, test_filtered],
        project_root,
        modalities,
    )
    input_dims = dict(embedding_dims)
    metadata_stats = None
    if "metadata" in modalities:
        metadata_stats = fit_metadata_stats(train_filtered)
        input_dims["metadata"] = max(1, len(metadata_stats.columns))

    dataset_kwargs = {
        "project_root": project_root,
        "modalities": modalities,
        "embedding_dims": embedding_dims,
        "target": target,
        "metadata_stats": metadata_stats,
    }
    train_dataset = MultimodalEmbeddingDataset(train_filtered, **dataset_kwargs)
    validation_dataset = MultimodalEmbeddingDataset(validation_filtered, **dataset_kwargs)
    test_dataset = MultimodalEmbeddingDataset(test_filtered, **dataset_kwargs)

    metadata_payload: dict[str, Any] = {
        "train_samples": int(len(train_filtered)),
        "validation_samples": int(len(validation_filtered)),
        "test_samples": int(len(test_filtered)),
        "input_dims": input_dims,
        "metadata_columns": metadata_stats.columns if metadata_stats else [],
        "metadata_means": metadata_stats.means if metadata_stats else {},
        "metadata_stds": metadata_stats.stds if metadata_stats else {},
    }
    return train_dataset, validation_dataset, test_dataset, input_dims, metadata_payload


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.random_seed
    set_seed(seed)

    modalities = parse_modalities(args.modalities)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Treinando multimodal transformer em %s com modalidades=%s.", device, modalities)

    train_df = read_parquet(config.unified_root / "train.parquet")
    validation_df = read_parquet(config.unified_root / "validation.parquet")
    test_df = read_parquet(config.unified_root / "test.parquet")

    train_dataset, validation_dataset, test_dataset, input_dims, data_info = build_datasets(
        train_df,
        validation_df,
        test_df,
        config.project_root,
        modalities,
        args.target,
        args.require_all_modalities,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = MultimodalTransformerClassifier(
        input_dims=input_dims,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    class_weight = positive_class_weight(train_dataset.df[args.target])
    if class_weight is not None:
        class_weight = class_weight.to(device)
        LOGGER.info("Usando pos_weight=%.4f.", float(class_weight.item()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_metric = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            args.threshold,
        )
        monitor = validation_metrics["roc_auc"]
        if monitor is None:
            monitor = validation_metrics["f1"] or 0.0
        epoch_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation_metrics,
        }
        history.append(epoch_payload)
        LOGGER.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_f1=%.4f val_auc=%s",
            epoch,
            train_loss,
            float(validation_metrics["loss"] or 0.0),
            float(validation_metrics["f1"] or 0.0),
            validation_metrics["roc_auc"],
        )

        if float(monitor) > best_metric:
            best_metric = float(monitor)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            LOGGER.info("Early stopping acionado na epoch %d.", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_metrics = evaluate(model, validation_loader, criterion, device, args.threshold)
    test_metrics = evaluate(model, test_loader, criterion, device, args.threshold) if len(test_dataset) else {}

    model_dir = ensure_dir(config.processed_root / "models" / "multimodal_transformer")
    checkpoint_path = model_dir / "model.pt"
    metrics_path = model_dir / "metrics.json"
    config_path = model_dir / "training_config.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dims": input_dims,
            "modalities": modalities,
            "target": args.target,
            "model_hparams": {
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
            },
            "data_info": data_info,
        },
        checkpoint_path,
    )

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_validation_monitor": best_metric,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "modalities": modalities,
                "target": args.target,
                "seed": seed,
                "require_all_modalities": args.require_all_modalities,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "threshold": args.threshold,
                **data_info,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Checkpoint salvo em %s.", checkpoint_path)
    LOGGER.info("Metricas salvas em %s.", metrics_path)


if __name__ == "__main__":
    main()

