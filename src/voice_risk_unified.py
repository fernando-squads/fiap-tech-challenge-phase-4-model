from __future__ import annotations

import csv
import shutil
from pathlib import Path

from io_utils import ensure_dir, sha1_short, slugify
from voice_risk import MULTILABEL_TARGETS


VOICE_RISK_METADATA_COLUMNS = [
    "sample_id",
    "participant_id",
    "audio_path",
    "transcript",
    "language",
    "voice_risk_label",
    "anxiety_label",
    "postpartum_depression_label",
    "hormonal_fatigue_label",
    "domestic_violence_label",
]


def materialize_voice_risk_raw_dataset(
    source_csv: Path,
    raw_output_dir: Path,
    dataset_key: str,
    language: str = "pt-BR",
    copy_audio: bool = True,
) -> Path:
    """Create a raw/ dataset compatible with the Unified pipeline."""

    ensure_dir(raw_output_dir)
    audio_dir = ensure_dir(raw_output_dir / "audio")
    metadata_path = raw_output_dir / "metadata.csv"
    rows: list[dict[str, str]] = []

    with source_csv.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for index, source_row in enumerate(reader, start=1):
            source_audio_path = Path(source_row["audio_path"])
            if not source_audio_path.is_absolute():
                source_audio_path = (source_csv.parent / source_audio_path).resolve()
            if not source_audio_path.exists():
                continue

            participant_id = _participant_id_from_audio(source_audio_path)
            sample_seed = f"{dataset_key}:{source_audio_path}:{index}"
            sample_id = f"{dataset_key}_{slugify(source_audio_path.stem)}_{sha1_short(sample_seed, 8)}"
            target_audio_path = audio_dir / f"{sample_id}{source_audio_path.suffix.lower()}"
            if copy_audio and source_audio_path.resolve() != target_audio_path.resolve():
                shutil.copy2(source_audio_path, target_audio_path)

            row = {
                "sample_id": sample_id,
                "participant_id": participant_id,
                "audio_path": f"audio/{target_audio_path.name}",
                "transcript": source_row.get("transcription", ""),
                "language": language,
                "voice_risk_label": source_row.get("binary_risk", ""),
                "anxiety_label": source_row.get("anxiety", ""),
                "postpartum_depression_label": source_row.get("postpartum_depression", ""),
                "hormonal_fatigue_label": source_row.get("hormonal_fatigue", ""),
                "domestic_violence_label": source_row.get("domestic_violence", ""),
            }
            for target in MULTILABEL_TARGETS:
                row.setdefault(f"{target}_label", source_row.get(target, ""))
            rows.append(row)

    with metadata_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=VOICE_RISK_METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return metadata_path


def _participant_id_from_audio(audio_path: Path) -> str:
    stem = audio_path.stem
    first_token = stem.split("_", 1)[0]
    if first_token.lower() == "synthetic":
        return slugify(stem)
    if first_token:
        return first_token
    return slugify(stem)
