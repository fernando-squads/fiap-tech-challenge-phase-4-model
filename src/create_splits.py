from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from config import load_config
from io_utils import write_parquet, read_parquet
from logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria splits por participant_id para evitar data leakage.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _can_stratify(labels: pd.Series, test_size: float) -> bool:
    if labels.isna().any():
        return False
    labels = labels.dropna()
    if labels.nunique() < 2:
        return False
    counts = labels.value_counts()
    if counts.min() < 2:
        return False
    n = len(labels)
    requested = max(1, int(round(n * test_size)))
    return requested >= labels.nunique()


def _manual_split(
    participants: pd.DataFrame,
    train_size: float,
    validation_size: float,
    test_size: float,
) -> pd.DataFrame:
    n_participants = len(participants)
    if n_participants == 0:
        participants["split"] = []
        return participants
    if n_participants == 1:
        participants["split"] = ["train"]
        return participants
    if n_participants == 2:
        participants["split"] = ["train", "test"]
        return participants

    n_test = max(1, round(n_participants * test_size))
    n_validation = max(1, round(n_participants * validation_size))
    n_train = n_participants - n_validation - n_test
    if n_train < 1:
        n_train = 1
        if n_validation > n_test and n_validation > 1:
            n_validation -= 1
        elif n_test > 1:
            n_test -= 1

    split_names = (
        ["train"] * n_train
        + ["validation"] * n_validation
        + ["test"] * n_test
    )
    split_names = split_names[:n_participants]
    while len(split_names) < n_participants:
        split_names.append("train")
    participants["split"] = split_names
    return participants


def _split_participants(participants: pd.DataFrame, config_seed: int, train_size: float, validation_size: float, test_size: float, stratify_by: str) -> pd.DataFrame:
    total_size = train_size + validation_size + test_size
    if abs(total_size - 1.0) > 1e-6:
        raise ValueError(f"As proporcoes de split devem somar 1.0; soma atual={total_size:.4f}")
    participants = participants.sample(frac=1.0, random_state=config_seed).reset_index(drop=True)
    if len(participants) < 6:
        return _manual_split(participants, train_size, validation_size, test_size)

    temp_size = validation_size + test_size
    stratify = participants[stratify_by] if stratify_by in participants and _can_stratify(participants[stratify_by], temp_size) else None
    train_df, temp_df = train_test_split(
        participants,
        test_size=temp_size,
        random_state=config_seed,
        stratify=stratify,
    )

    relative_test_size = test_size / temp_size if temp_size else 0.5
    temp_stratify = temp_df[stratify_by] if stratify_by in temp_df and _can_stratify(temp_df[stratify_by], relative_test_size) else None
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=config_seed,
        stratify=temp_stratify,
    )

    train_df = train_df.assign(split="train")
    validation_df = validation_df.assign(split="validation")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, validation_df, test_df], ignore_index=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    metadata = read_parquet(config.processed_root / "metadata" / "metadata.parquet")

    if metadata.empty:
        LOGGER.warning("Metadata vazia; salvando splits vazios.")
        write_parquet(pd.DataFrame(columns=["sample_id", "participant_id", "split"]), config.processed_root / "metadata" / "splits.parquet")
        return

    metadata["participant_id"] = metadata["participant_id"].fillna(metadata["sample_id"])
    stratify_by = config.splits.stratify_by
    aggregations = {"sample_id": "count"}
    if stratify_by in metadata.columns:
        aggregations[stratify_by] = "max"
    participants = (
        metadata.groupby("participant_id", as_index=False)
        .agg(aggregations)
        .rename(columns={"sample_id": "n_samples"})
    )

    split_participants = _split_participants(
        participants,
        config.random_seed,
        config.splits.train_size,
        config.splits.validation_size,
        config.splits.test_size,
        stratify_by,
    )
    sample_splits = metadata[["sample_id", "participant_id"]].merge(
        split_participants[["participant_id", "split"]],
        on="participant_id",
        how="left",
    )
    output_path = config.processed_root / "metadata" / "splits.parquet"
    write_parquet(sample_splits, output_path)
    counts = sample_splits["split"].value_counts(dropna=False).to_dict()
    LOGGER.info("Splits salvos em %s. Contagem por amostra: %s", output_path, counts)


if __name__ == "__main__":
    main()
