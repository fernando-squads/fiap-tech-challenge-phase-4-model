from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import PipelineConfig, load_config
from io_utils import ensure_dir, read_parquet, write_json
from logging_utils import configure_logging
from multimodal_data import (
    EMBEDDING_COLUMNS,
    SAFE_METADATA_COLUMNS,
    infer_embedding_dims,
    parse_modalities,
    resolve_existing_path,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatrixResult:
    x: np.ndarray
    kept: pd.DataFrame
    skipped_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treina um detector de anomalias nao supervisionado sobre embeddings "
            "multimodais do dataset unificado."
        )
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--modalities", type=str, default="audio,text")
    parser.add_argument("--metadata-columns", type=str, default=",".join(SAFE_METADATA_COLUMNS))
    parser.add_argument("--require-all-modalities", action="store_true")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-samples", type=str, default="auto")
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--normal-label-column", type=str, default=None)
    parser.add_argument("--normal-label-value", type=str, default="0")
    parser.add_argument("--target-eval-column", type=str, default="depression_label")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _split_columns(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_max_samples(value: str) -> str | int | float:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    if "." in text:
        return float(text)
    return int(text)


def _normal_label_value(value: str) -> int | float | str:
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _fit_vector(vector: np.ndarray, expected_dim: int) -> np.ndarray:
    flat = vector.astype(np.float32).reshape(-1)
    if flat.shape[0] == expected_dim:
        return flat
    if flat.shape[0] > expected_dim:
        return flat[:expected_dim]
    padded = np.zeros(expected_dim, dtype=np.float32)
    padded[: flat.shape[0]] = flat
    return padded


def _load_embedding(path: Path | None, expected_dim: int) -> tuple[np.ndarray, bool]:
    if path is None:
        return np.zeros(expected_dim, dtype=np.float32), False
    try:
        vector = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        LOGGER.warning("Embedding invalido ignorado: %s (%s)", path, exc)
        return np.zeros(expected_dim, dtype=np.float32), False
    return _fit_vector(vector, expected_dim), True


def fit_metadata_fill_values(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    fill_values: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(df.get(column), errors="coerce")
        if values is None or not values.notna().any():
            fill_values[column] = 0.0
            continue
        fill_values[column] = float(values.median())
    return fill_values


def _metadata_value(row: pd.Series, column: str, fill_values: dict[str, float]) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    if pd.isna(value):
        return float(fill_values.get(column, 0.0))
    return float(value)


def feature_count(
    modalities: list[str],
    embedding_dims: dict[str, int],
    metadata_columns: list[str],
) -> int:
    count = 0
    for modality in modalities:
        if modality in EMBEDDING_COLUMNS:
            count += int(embedding_dims[modality]) + 1
    if "metadata" in modalities:
        count += len(metadata_columns)
    return count


def build_feature_blocks(
    modalities: list[str],
    embedding_dims: dict[str, int],
    metadata_columns: list[str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for modality in modalities:
        if modality in EMBEDDING_COLUMNS:
            blocks.append(
                {
                    "name": f"{modality}_embedding",
                    "kind": "embedding",
                    "dimension": int(embedding_dims[modality]),
                }
            )
            blocks.append(
                {
                    "name": f"{modality}_present",
                    "kind": "presence_flag",
                    "dimension": 1,
                }
            )
    if "metadata" in modalities:
        blocks.append(
            {
                "name": "metadata",
                "kind": "numeric_metadata",
                "columns": metadata_columns,
                "dimension": len(metadata_columns),
            }
        )
    return blocks


def build_anomaly_matrix(
    df: pd.DataFrame,
    project_root: Path,
    modalities: list[str],
    embedding_dims: dict[str, int],
    metadata_columns: list[str],
    metadata_fill_values: dict[str, float],
    require_all_modalities: bool,
) -> MatrixResult:
    expected_feature_count = feature_count(modalities, embedding_dims, metadata_columns)
    if expected_feature_count == 0:
        raise ValueError("Nenhuma feature disponivel para o detector de anomalias.")

    requested_embedding_modalities = [item for item in modalities if item in EMBEDDING_COLUMNS]
    rows: list[np.ndarray] = []
    kept_rows: list[dict[str, Any]] = []
    skipped_rows = 0

    for _, row in df.iterrows():
        vectors: list[np.ndarray] = []
        modality_present: dict[str, bool] = {}

        for modality in requested_embedding_modalities:
            column = EMBEDDING_COLUMNS[modality]
            path = resolve_existing_path(row.get(column), project_root)
            vector, present = _load_embedding(path, embedding_dims[modality])
            modality_present[modality] = present
            vectors.append(vector)
            vectors.append(np.asarray([1.0 if present else 0.0], dtype=np.float32))

        if requested_embedding_modalities:
            present_values = [modality_present[item] for item in requested_embedding_modalities]
            has_required_embeddings = (
                all(present_values) if require_all_modalities else any(present_values)
            )
            if not has_required_embeddings:
                skipped_rows += 1
                continue

        if "metadata" in modalities and metadata_columns:
            metadata_vector = np.asarray(
                [
                    _metadata_value(row, column, metadata_fill_values)
                    for column in metadata_columns
                ],
                dtype=np.float32,
            )
            vectors.append(metadata_vector)

        rows.append(np.concatenate(vectors).astype(np.float32))
        kept_rows.append(row.to_dict())

    if not rows:
        return MatrixResult(
            x=np.empty((0, expected_feature_count), dtype=np.float32),
            kept=pd.DataFrame(),
            skipped_rows=skipped_rows,
        )

    return MatrixResult(
        x=np.vstack(rows).astype(np.float32),
        kept=pd.DataFrame(kept_rows),
        skipped_rows=skipped_rows,
    )


def filter_training_rows(
    df: pd.DataFrame,
    normal_label_column: str | None,
    normal_label_value: str,
) -> pd.DataFrame:
    if normal_label_column is None:
        return df
    if normal_label_column not in df.columns:
        raise ValueError(f"Coluna normal-label-column nao encontrada: {normal_label_column}")
    expected = _normal_label_value(normal_label_value)
    if isinstance(expected, (int, float)):
        values = pd.to_numeric(df[normal_label_column], errors="coerce")
        mask = values == expected
    else:
        mask = df[normal_label_column].astype(str) == str(expected)
    filtered = df[mask].copy()
    if filtered.empty:
        raise ValueError(
            "Filtro de treinamento normal removeu todas as amostras: "
            f"{normal_label_column}={expected}."
        )
    LOGGER.info(
        "Treinando apenas com amostras normais: coluna=%s valor=%s amostras=%d.",
        normal_label_column,
        expected,
        len(filtered),
    )
    return filtered


def build_pipeline(args: argparse.Namespace, seed: int, n_features: int, n_samples: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    pca_components = int(args.pca_components)
    if pca_components > 0:
        max_components = min(pca_components, n_features, max(1, n_samples - 1))
        if max_components >= 2:
            steps.append(("pca", PCA(n_components=max_components, random_state=seed)))
        else:
            LOGGER.info("PCA ignorado: amostras/features insuficientes.")

    steps.append(
        (
            "isolation_forest",
            IsolationForest(
                n_estimators=int(args.n_estimators),
                contamination=float(args.contamination),
                max_samples=_parse_max_samples(args.max_samples),
                random_state=seed,
                n_jobs=int(args.n_jobs),
            ),
        )
    )
    return Pipeline(steps)


def raw_anomaly_scores(pipeline: Pipeline, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.asarray([], dtype=np.float32)
    return (-pipeline.decision_function(x)).astype(np.float32)


def normalize_scores(raw_scores: np.ndarray, raw_min: float, raw_max: float) -> np.ndarray:
    if raw_scores.size == 0:
        return np.asarray([], dtype=np.float32)
    span = raw_max - raw_min
    if not np.isfinite(span) or span <= 0:
        return np.zeros_like(raw_scores, dtype=np.float32)
    return np.clip((raw_scores - raw_min) / span, 0.0, 1.0).astype(np.float32)


def score_summary(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    if scores.size == 0:
        return {
            "samples": 0,
            "anomaly_count": 0,
            "anomaly_rate": 0.0,
            "score_min": None,
            "score_mean": None,
            "score_max": None,
            "score_p50": None,
            "score_p90": None,
            "score_p95": None,
            "score_p99": None,
        }
    predictions = scores >= threshold
    return {
        "samples": int(scores.shape[0]),
        "anomaly_count": int(predictions.sum()),
        "anomaly_rate": float(predictions.mean()),
        "score_min": float(np.min(scores)),
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
        "score_p50": float(np.quantile(scores, 0.50)),
        "score_p90": float(np.quantile(scores, 0.90)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_p99": float(np.quantile(scores, 0.99)),
    }


def label_metrics(
    kept_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    target_column: str,
) -> dict[str, float | None]:
    if kept_df.empty or target_column not in kept_df.columns or scores.size == 0:
        return {}
    labels = pd.to_numeric(kept_df[target_column], errors="coerce")
    mask = labels.notna().to_numpy()
    if not mask.any():
        return {}
    y_true = labels[mask].astype(int).to_numpy()
    y_score = scores[mask]
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, float | None] = {
        "accuracy_against_label": float(accuracy_score(y_true, y_pred)),
        "f1_against_label": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc_against_label"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc_against_label"] = None
    return metrics


def split_report(
    kept_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    target_column: str,
) -> dict[str, Any]:
    payload = score_summary(scores, threshold)
    payload.update(label_metrics(kept_df, scores, threshold, target_column))
    return payload


def selected_metadata_columns(df: pd.DataFrame, raw_columns: str | None) -> list[str]:
    requested = _split_columns(raw_columns)
    return [column for column in requested if column in df.columns]


def train_anomaly_detector(config: PipelineConfig, args: argparse.Namespace) -> dict[str, Path]:
    seed = int(args.seed if args.seed is not None else config.random_seed)
    if not 0.0 < float(args.contamination) < 0.5:
        raise ValueError("--contamination deve ser maior que 0 e menor que 0.5.")
    if args.threshold is not None and not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError("--threshold deve estar entre 0 e 1.")

    modalities = parse_modalities(args.modalities)
    LOGGER.info("Treinando detector de anomalias com modalidades=%s.", modalities)

    train_df = read_parquet(config.unified_root / "train.parquet")
    validation_df = read_parquet(config.unified_root / "validation.parquet")
    test_df = read_parquet(config.unified_root / "test.parquet")
    train_for_fit = filter_training_rows(
        train_df,
        args.normal_label_column,
        args.normal_label_value,
    )

    embedding_dims = infer_embedding_dims(
        [train_for_fit, validation_df, test_df],
        config.project_root,
        modalities,
    )
    metadata_columns = (
        selected_metadata_columns(train_for_fit, args.metadata_columns)
        if "metadata" in modalities
        else []
    )
    metadata_fill_values = fit_metadata_fill_values(train_for_fit, metadata_columns)

    train_matrix = build_anomaly_matrix(
        train_for_fit,
        config.project_root,
        modalities,
        embedding_dims,
        metadata_columns,
        metadata_fill_values,
        args.require_all_modalities,
    )
    if train_matrix.x.shape[0] < 2:
        raise ValueError(
            "O detector de anomalias precisa de pelo menos 2 amostras validas para treino. "
            f"Amostras validas={train_matrix.x.shape[0]}."
        )

    validation_matrix = build_anomaly_matrix(
        validation_df,
        config.project_root,
        modalities,
        embedding_dims,
        metadata_columns,
        metadata_fill_values,
        args.require_all_modalities,
    )
    test_matrix = build_anomaly_matrix(
        test_df,
        config.project_root,
        modalities,
        embedding_dims,
        metadata_columns,
        metadata_fill_values,
        args.require_all_modalities,
    )

    pipeline = build_pipeline(
        args,
        seed=seed,
        n_features=train_matrix.x.shape[1],
        n_samples=train_matrix.x.shape[0],
    )
    pipeline.fit(train_matrix.x)

    train_raw_scores = raw_anomaly_scores(pipeline, train_matrix.x)
    raw_min = float(np.min(train_raw_scores))
    raw_max = float(np.max(train_raw_scores))
    train_scores = normalize_scores(train_raw_scores, raw_min, raw_max)
    if args.threshold is None:
        threshold = (
            1.0
            if np.allclose(train_scores, 0.0)
            else float(np.quantile(train_scores, 1.0 - float(args.contamination)))
        )
    else:
        threshold = float(args.threshold)

    validation_scores = normalize_scores(
        raw_anomaly_scores(pipeline, validation_matrix.x),
        raw_min,
        raw_max,
    )
    test_scores = normalize_scores(
        raw_anomaly_scores(pipeline, test_matrix.x),
        raw_min,
        raw_max,
    )

    model_dir = ensure_dir(config.processed_root / "models" / "anomaly_detector")
    artifact_path = model_dir / "anomaly_detector.joblib"
    metrics_path = model_dir / "anomaly_metrics.json"
    config_path = model_dir / "anomaly_config.json"

    feature_blocks = build_feature_blocks(modalities, embedding_dims, metadata_columns)
    artifact = {
        "schema_version": 1,
        "task": "unsupervised_anomaly_detection",
        "model_type": "isolation_forest",
        "pipeline": pipeline,
        "modalities": modalities,
        "embedding_dims": {key: int(value) for key, value in embedding_dims.items()},
        "metadata_columns": metadata_columns,
        "metadata_fill_values": metadata_fill_values,
        "feature_blocks": feature_blocks,
        "feature_count": int(train_matrix.x.shape[1]),
        "require_all_modalities": bool(args.require_all_modalities),
        "threshold": float(threshold),
        "calibration": {
            "raw_min": raw_min,
            "raw_max": raw_max,
            "normalized_min": float(np.min(train_scores)),
            "normalized_max": float(np.max(train_scores)),
        },
        "contamination": float(args.contamination),
        "target_eval_column": args.target_eval_column,
        "normal_label_column": args.normal_label_column,
        "normal_label_value": args.normal_label_value if args.normal_label_column else None,
        "models": {
            "audio_embedding_model": config.embeddings.audio_model_name,
            "text_embedding_model": config.embeddings.text_model_name,
        },
        "sklearn_version": sklearn.__version__,
        "created_by": "src/train_anomaly_detector.py",
    }
    joblib.dump(artifact, artifact_path)

    metrics_payload = {
        "threshold": float(threshold),
        "contamination": float(args.contamination),
        "modalities": modalities,
        "feature_count": int(train_matrix.x.shape[1]),
        "train": split_report(
            train_matrix.kept,
            train_scores,
            threshold,
            args.target_eval_column,
        ),
        "validation": split_report(
            validation_matrix.kept,
            validation_scores,
            threshold,
            args.target_eval_column,
        ),
        "test": split_report(
            test_matrix.kept,
            test_scores,
            threshold,
            args.target_eval_column,
        ),
        "skipped_rows": {
            "train": int(train_matrix.skipped_rows),
            "validation": int(validation_matrix.skipped_rows),
            "test": int(test_matrix.skipped_rows),
        },
    }
    config_payload = {
        key: value
        for key, value in artifact.items()
        if key != "pipeline"
    }
    write_json(metrics_payload, metrics_path)
    write_json(config_payload, config_path)

    LOGGER.info("Detector de anomalias salvo em %s.", artifact_path)
    LOGGER.info("Metricas salvas em %s.", metrics_path)
    return {
        "artifact": artifact_path,
        "metrics": metrics_path,
        "config": config_path,
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    train_anomaly_detector(config, args)


if __name__ == "__main__":
    main()
