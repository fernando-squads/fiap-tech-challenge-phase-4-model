from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from voice_risk import (
    MULTILABEL_TARGETS,
    VoiceRiskTrainingConfig,
    build_training_features,
    load_dataset_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def positive_probabilities(pipeline: Pipeline, features: list[dict[str, float]]) -> np.ndarray:
    probabilities = pipeline.predict_proba(features)
    classes = list(pipeline.named_steps["classifier"].classes_)
    if 1 in classes:
        positive_index = classes.index(1)
        return probabilities[:, positive_index]
    return np.zeros(len(features), dtype=float)


def train_binary_model(
    features: list[dict[str, float]],
    rows: list[dict[str, str]],
    *,
    output_path: Path,
    random_state: int,
    threshold: float,
) -> dict[str, Any]:
    y = []
    for row in rows:
        if "binary_risk" not in row:
            raise ValueError("Dataset deve conter a coluna binary_risk.")
        y.append(int(float(row["binary_risk"])))

    if len(set(y)) < 2:
        raise ValueError("Dataset binario deve conter exemplos das classes 0 e 1.")

    pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(features, y)
    ensure_logistic_regression_backward_compatibility(pipeline)

    predictions = pipeline.predict(features)
    probabilities = positive_probabilities(pipeline, features)
    metrics = {"accuracy": float(accuracy_score(y, predictions))}
    if len(set(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, probabilities))

    artifact = {
        "task": "binary",
        "pipeline": pipeline,
        "threshold": threshold,
        "feature_count": len(pipeline.named_steps["vectorizer"].feature_names_),
        "sklearn_version": sklearn.__version__,
    }
    ensure_dir(output_path.parent)
    joblib.dump(artifact, output_path)
    return {
        "task": "binary",
        "model_path": str(output_path),
        "rows": len(rows),
        "features": artifact["feature_count"],
        "metrics": metrics,
        "labels": ["nao_risco", "risco"],
    }


def train_multilabel_model(
    features: list[dict[str, float]],
    rows: list[dict[str, str]],
    *,
    output_path: Path,
    random_state: int,
    threshold: float,
) -> dict[str, Any]:
    missing = [target for target in MULTILABEL_TARGETS if target not in rows[0]]
    if missing:
        raise ValueError("Dataset deve conter as colunas multilabel: " + ", ".join(missing))

    y = np.asarray(
        [
            [int(float(row[target])) for target in MULTILABEL_TARGETS]
            for row in rows
        ],
        dtype=int,
    )
    labels_without_variance = [
        MULTILABEL_TARGETS[index]
        for index in range(y.shape[1])
        if len(set(y[:, index])) < 2
    ]
    if labels_without_variance:
        raise ValueError(
            "Cada label multilabel precisa ter exemplos positivos e negativos. "
            "Labels sem variacao: " + ", ".join(labels_without_variance)
        )

    pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    )
                ),
            ),
        ]
    )
    pipeline.fit(features, y)
    ensure_logistic_regression_backward_compatibility(pipeline)

    predicted = pipeline.predict(features)
    metrics = {
        "subset_accuracy": float(np.mean(np.all(predicted == y, axis=1))),
        "label_accuracy": float(np.mean(predicted == y)),
    }

    artifact = {
        "task": "multilabel",
        "pipeline": pipeline,
        "labels": MULTILABEL_TARGETS,
        "threshold": threshold,
        "feature_count": len(pipeline.named_steps["vectorizer"].feature_names_),
        "sklearn_version": sklearn.__version__,
    }
    ensure_dir(output_path.parent)
    joblib.dump(artifact, output_path)
    return {
        "task": "multilabel",
        "model_path": str(output_path),
        "rows": len(rows),
        "features": artifact["feature_count"],
        "metrics": metrics,
        "labels": MULTILABEL_TARGETS,
    }


def train_voice_risk_models(
    *,
    dataset_path: Path,
    output_dir: Path,
    random_state: int,
    config: VoiceRiskTrainingConfig,
) -> dict[str, Any]:
    rows = load_dataset_rows(dataset_path)
    features = build_training_features(dataset_path, rows, config)
    ensure_dir(output_dir)

    binary = train_binary_model(
        features,
        rows,
        output_path=output_dir / "binary_risk_model.joblib",
        random_state=random_state,
        threshold=config.binary_risk_threshold,
    )
    multilabel = train_multilabel_model(
        features,
        rows,
        output_path=output_dir / "multilabel_risk_model.joblib",
        random_state=random_state,
        threshold=config.multilabel_risk_threshold,
    )
    metrics = {
        "dataset_path": str(dataset_path),
        "rows": len(rows),
        "binary": binary,
        "multilabel": multilabel,
    }
    (output_dir / "voice_risk_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def ensure_logistic_regression_backward_compatibility(pipeline: Pipeline) -> None:
    classifier = pipeline.named_steps["classifier"]
    estimators = [classifier]
    estimators.extend(list(getattr(classifier, "estimators_", [])))
    for estimator in estimators:
        if estimator.__class__.__name__ == "LogisticRegression" and not hasattr(estimator, "multi_class"):
            estimator.multi_class = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina os modelos de risco de voz consumidos pela API FastAPI."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="CSV com audio_path, transcription, binary_risk e labels multilabel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretorio onde os artefatos .joblib serao salvos.",
    )
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--binary-threshold", type=float, default=0.65)
    parser.add_argument("--multilabel-threshold", type=float, default=0.55)
    parser.add_argument("--allow-missing-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        from config import load_config

        pipeline_config = load_config(args.config)
        processed_root = pipeline_config.processed_root
        sample_rate = pipeline_config.audio.sample_rate
        default_random_state = pipeline_config.random_seed
    else:
        processed_root = PROJECT_ROOT / "processed"
        sample_rate = 22_050
        default_random_state = 42

    dataset_path = args.dataset or processed_root / "voice_risk" / "training_dataset.csv"
    output_dir = args.output_dir or processed_root / "models" / "voice_risk"
    random_state = args.random_state if args.random_state is not None else default_random_state

    metrics = train_voice_risk_models(
        dataset_path=dataset_path,
        output_dir=output_dir,
        random_state=random_state,
        config=VoiceRiskTrainingConfig(
            audio_sample_rate=sample_rate,
            binary_risk_threshold=args.binary_threshold,
            multilabel_risk_threshold=args.multilabel_threshold,
            require_audio_files=not args.allow_missing_audio,
        ),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
