from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import load_config
from io_utils import ensure_dir, read_parquet, resolve_project_path
from logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
MODALITY_COLUMNS = {
    "audio": "audio_embedding_path",
    "text": "text_embedding_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina baseline sklearn para depression_label usando embeddings.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--features", type=str, default="auto", help="auto, audio, text ou audio,text")
    parser.add_argument("--target", type=str, default="depression_label")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _available_modalities(df: pd.DataFrame, project_root: Path) -> list[str]:
    modalities: list[str] = []
    for modality, column in MODALITY_COLUMNS.items():
        if column not in df.columns:
            continue
        paths = [resolve_project_path(value, project_root) for value in df[column].dropna().tolist()]
        if any(path and path.exists() for path in paths):
            modalities.append(modality)
    return modalities


def _usable_count(
    df: pd.DataFrame,
    project_root: Path,
    modalities: list[str],
    target: str,
) -> int:
    if target not in df.columns:
        return 0
    count = 0
    labeled = df[df[target].notna()]
    for _, row in labeled.iterrows():
        usable = True
        for modality in modalities:
            path = resolve_project_path(row.get(MODALITY_COLUMNS[modality]), project_root)
            if path is None or not path.exists():
                usable = False
                break
        if usable:
            count += 1
    return count


def _dataset_diagnostics(
    df: pd.DataFrame,
    project_root: Path,
    target: str,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "rows": int(len(df)),
        "labeled_rows": int(df[target].notna().sum()) if target in df.columns else 0,
    }
    for modality, column in MODALITY_COLUMNS.items():
        if column not in df.columns:
            diagnostics[f"{modality}_paths"] = 0
            diagnostics[f"{modality}_existing_files"] = 0
            diagnostics[f"{modality}_usable_labeled_rows"] = 0
            continue
        paths = [resolve_project_path(value, project_root) for value in df[column].dropna().tolist()]
        diagnostics[f"{modality}_paths"] = int(len(paths))
        diagnostics[f"{modality}_existing_files"] = int(sum(bool(path and path.exists()) for path in paths))
        diagnostics[f"{modality}_usable_labeled_rows"] = _usable_count(
            df,
            project_root,
            [modality],
            target,
        )
    return diagnostics


def _select_auto_modalities(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    project_root: Path,
    target: str,
) -> list[str]:
    candidates = [["audio"], ["text"], ["audio", "text"]]
    scored: list[tuple[int, int, int, list[str]]] = []
    for index, modalities in enumerate(candidates):
        train_count = _usable_count(train_df, project_root, modalities, target)
        validation_count = _usable_count(validation_df, project_root, modalities, target)
        if train_count > 0 and validation_count > 0:
            scored.append((min(train_count, validation_count), train_count + validation_count, -index, modalities))
    if not scored:
        return []
    scored.sort(reverse=True)
    return scored[0][3]


def _load_matrix(df: pd.DataFrame, project_root: Path, modalities: list[str], target: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    kept_rows: list[int] = []
    missing_modalities = [modality for modality in modalities if modality not in MODALITY_COLUMNS]
    if missing_modalities:
        raise ValueError(f"Modalidades desconhecidas: {missing_modalities}")
    if target not in df.columns:
        raise ValueError(f"Coluna target nao encontrada: {target}")
    labeled = df[df[target].notna()].copy()
    for index, row in labeled.iterrows():
        vectors: list[np.ndarray] = []
        missing = False
        for modality in modalities:
            path = resolve_project_path(row.get(MODALITY_COLUMNS[modality]), project_root)
            if path is None or not path.exists():
                missing = True
                break
            vectors.append(np.load(path).astype("float32").reshape(-1))
        if missing or not vectors:
            continue
        rows.append(np.concatenate(vectors))
        labels.append(int(row[target]))
        kept_rows.append(index)
    if not rows:
        diagnostics = _dataset_diagnostics(df, project_root, target)
        raise ValueError(
            "Nenhuma amostra com embeddings e label foi encontrada para o baseline. "
            f"Modalidades solicitadas={modalities}. Diagnostico={diagnostics}. "
            "Reexecute build_metadata.py, create_splits.py, export_unified_dataset.py "
            "depois de gerar os embeddings."
        )
    return np.vstack(rows), np.asarray(labels), labeled.loc[kept_rows]


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = None
    return metrics


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)

    train_df = read_parquet(config.unified_root / "train.parquet")
    validation_df = read_parquet(config.unified_root / "validation.parquet")
    test_df = read_parquet(config.unified_root / "test.parquet")

    if args.features == "auto":
        modalities = _select_auto_modalities(
            train_df,
            validation_df,
            config.project_root,
            args.target,
        )
    else:
        modalities = [item.strip() for item in args.features.split(",") if item.strip()]
    if not modalities:
        raise ValueError(
            "Nenhuma modalidade de embedding disponivel com labels em treino e validacao. "
            f"Treino={_dataset_diagnostics(train_df, config.project_root, args.target)}. "
            f"Validacao={_dataset_diagnostics(validation_df, config.project_root, args.target)}."
        )
    LOGGER.info("Treinando baseline com modalidades: %s", modalities)

    x_train, y_train, kept_train = _load_matrix(train_df, config.project_root, modalities, args.target)
    x_val, y_val, kept_val = _load_matrix(validation_df, config.project_root, modalities, args.target)
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"O treino precisa de pelo menos duas classes em {args.target}; classes encontradas={np.unique(y_train).tolist()}."
        )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=config.random_seed,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    validation_probabilities = model.predict_proba(x_val)[:, 1]
    validation_predictions = (validation_probabilities >= 0.5).astype(int)
    metrics = {
        "modalities": modalities,
        "target": args.target,
        "train_samples": int(len(kept_train)),
        "validation_samples": int(len(kept_val)),
        "validation": _metrics(y_val, validation_probabilities, validation_predictions),
    }

    try:
        x_test, y_test, kept_test = _load_matrix(test_df, config.project_root, modalities, args.target)
        test_probabilities = model.predict_proba(x_test)[:, 1]
        test_predictions = (test_probabilities >= 0.5).astype(int)
        metrics["test_samples"] = int(len(kept_test))
        metrics["test"] = _metrics(y_test, test_probabilities, test_predictions)
    except ValueError as exc:
        LOGGER.warning("Pulando avaliacao de teste: %s", exc)

    model_dir = ensure_dir(config.processed_root / "models")
    model_path = model_dir / "depression_baseline.joblib"
    metrics_path = model_dir / "depression_baseline_metrics.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Modelo salvo em %s.", model_path)
    LOGGER.info("Metricas salvas em %s: %s", metrics_path, metrics)


if __name__ == "__main__":
    main()
