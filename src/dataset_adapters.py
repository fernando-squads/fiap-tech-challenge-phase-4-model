from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from config import DatasetConfig, PipelineConfig
from io_utils import (
    as_project_path,
    ensure_dir,
    list_files,
    normalize_column_name,
    read_table,
    sha1_short,
    slugify,
)
from labels import (
    clean_text,
    missing,
    normalize_gender,
    parse_float,
    parse_int_label,
    standardize_binary_label,
)

LOGGER = logging.getLogger(__name__)


DEFAULT_COLUMN_ALIASES: dict[str, list[str]] = {
    "sample_id": ["sample_id", "sample", "recording_id", "record_id", "session_id", "uid"],
    "participant_id": [
        "participant_id",
        "participant",
        "subject_id",
        "subject",
        "speaker_id",
        "patient_id",
        "user_id",
        "person_id",
        "interview_id",
        "id",
    ],
    "audio_file": [
        "audio_file",
        "audio_filename",
        "audio_path",
        "file",
        "filename",
        "wav_file",
        "wav_path",
        "media_path",
    ],
    "transcript": [
        "transcript",
        "transcription",
        "text",
        "utterance",
        "sentence",
        "content",
        "response",
        "answer",
        "value",
    ],
    "language": ["language", "lang", "locale"],
    "gender": ["gender", "sex", "biological_sex"],
    "age": ["age", "idade"],
    "phq_score": [
        "phq_score",
        "phq8_score",
        "phq_8",
        "phq9_score",
        "phq_9",
        "phq",
        "depression_score",
    ],
    "gad_score": [
        "gad_score",
        "gad7_score",
        "gad_7",
        "gad",
        "anxiety_score",
    ],
    "depression_label": [
        "depression_label",
        "depression_binary",
        "depressed",
        "phq_binary",
        "phq8_binary",
        "phq9_binary",
        "depression_class",
    ],
    "anxiety_label": [
        "anxiety_label",
        "anxiety_binary",
        "anxious",
        "gad_binary",
        "gad7_binary",
        "anxiety",
        "anxiety_class",
    ],
    "emotion_label": ["emotion_label", "emotion", "sentiment", "affect", "mood"],
}

TEXT_EXTENSIONS = {".txt", ".cha", ".trs"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json"}
INTERVIEWER_SPEAKERS = {
    "ellie",
    "interviewer",
    "interview",
    "therapist",
    "doctor",
    "clinician",
    "researcher",
}


def build_sample_id(dataset_key: str, raw_path: Path, raw_root: Path) -> str:
    try:
        relative = raw_path.resolve().relative_to(raw_root.resolve()).as_posix()
    except ValueError:
        relative = raw_path.as_posix()
    return f"{dataset_key}_{slugify(raw_path.stem)}_{sha1_short(relative, 8)}"


def infer_participant_id(raw_path: Path) -> str:
    stem_prefix = raw_path.stem.split("_", 1)[0]
    if re.fullmatch(r"[a-fA-F0-9]{8,}", stem_prefix):
        return stem_prefix.lower()
    for part in reversed(raw_path.parts):
        if re.fullmatch(r"[tv]_\d+", part, flags=re.IGNORECASE):
            return part
        named_match = re.search(
            r"(?:participant|subject|subj|patient|speaker|user|person|interview|id)[-_ ]?([a-zA-Z0-9]{2,})",
            part,
            flags=re.IGNORECASE,
        )
        if named_match:
            return named_match.group(1)
        short_match = re.fullmatch(r"[pP][-_ ]?([a-zA-Z0-9]{2,})", part)
        if short_match:
            return short_match.group(1)
        numeric = re.search(r"\d{2,}", part)
        if numeric:
            return numeric.group(0)

    stem = raw_path.stem
    return re.split(r"[_\-\s]+", stem)[0] or stem


def audio_files_for_dataset(dataset_config: DatasetConfig) -> list[Path]:
    return list_files(dataset_config.raw_dir, dataset_config.audio_globs)


def _combined_aliases(dataset_config: DatasetConfig) -> dict[str, list[str]]:
    aliases = {key: list(values) for key, values in DEFAULT_COLUMN_ALIASES.items()}
    for key, values in dataset_config.column_aliases.items():
        aliases.setdefault(key, [])
        aliases[key].extend(values)
    return aliases


def _normalize_key(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return normalize_column_name(Path(text).name)


def _identifier_tokens(value: Any) -> set[str]:
    text = clean_text(value)
    if text is None:
        return set()
    path = Path(text)
    tokens = {
        normalize_column_name(text),
        normalize_column_name(path.name),
        normalize_column_name(path.stem),
        normalize_column_name(f"{path.parent.name}_{path.stem}"),
    }
    return {token for token in tokens if token}


def _emotion_from_audio_stem(path_value: Any) -> str | None:
    text = clean_text(path_value)
    if text is None:
        return None
    stem = Path(text).stem.lower()
    stem = stem.removesuffix("_out")
    if stem in {"positive", "negative", "neutral"}:
        return stem
    return None


def _normalize_row(
    row: pd.Series,
    aliases: dict[str, list[str]],
    source_path: Path,
    row_number: int,
) -> dict[str, Any]:
    normalized_columns = {normalize_column_name(column): column for column in row.index}
    result: dict[str, Any] = {
        "source_file": source_path.as_posix(),
        "source_row_number": row_number,
    }
    for target, candidates in aliases.items():
        for candidate in candidates:
            actual = normalized_columns.get(normalize_column_name(candidate))
            if actual is not None:
                value = row.get(actual)
                if not missing(value):
                    result[target] = value
                break
    return result


def _table_has_metadata_signal(df: pd.DataFrame, aliases: dict[str, list[str]]) -> bool:
    normalized_columns = {normalize_column_name(column) for column in df.columns}
    metadata_targets = {
        "participant_id",
        "sample_id",
        "audio_file",
        "gender",
        "age",
        "phq_score",
        "gad_score",
        "depression_label",
        "anxiety_label",
        "voice_risk_label",
        "postpartum_depression_label",
        "hormonal_fatigue_label",
        "domestic_violence_label",
        "emotion_label",
    }
    for target in metadata_targets:
        for alias in aliases.get(target, []):
            if normalize_column_name(alias) in normalized_columns:
                return True
    return False


def _extract_transcript_from_table(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    normalized_columns = {normalize_column_name(column): column for column in df.columns}
    text_column = None
    for candidate in DEFAULT_COLUMN_ALIASES["transcript"]:
        text_column = normalized_columns.get(normalize_column_name(candidate))
        if text_column is not None:
            break
    if text_column is None:
        return None

    filtered = df
    speaker_column = None
    for candidate in ["speaker", "role", "speaker_role", "participant_type"]:
        speaker_column = normalized_columns.get(candidate)
        if speaker_column is not None:
            break
    if speaker_column is not None:
        speaker_values = filtered[speaker_column].astype(str).str.lower().str.strip()
        filtered = filtered[~speaker_values.isin(INTERVIEWER_SPEAKERS)]

    utterances = [clean_text(value) for value in filtered[text_column].tolist()]
    utterances = [value for value in utterances if value]
    return " ".join(utterances) if utterances else None


class GenericDatasetAdapter:
    def __init__(self, config: PipelineConfig, dataset_config: DatasetConfig) -> None:
        self.config = config
        self.dataset_config = dataset_config
        self.aliases = _combined_aliases(dataset_config)

    def load_metadata_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in list_files(self.dataset_config.raw_dir, self.dataset_config.metadata_globs):
            if path.suffix.lower() not in TABLE_EXTENSIONS:
                continue
            if "transcript" in normalize_column_name(path.stem):
                continue
            try:
                df = read_table(path)
            except Exception as exc:
                LOGGER.warning("Ignorando metadata invalido %s: %s", path, exc)
                continue
            if not _table_has_metadata_signal(df, self.aliases):
                continue
            for row_number, (_, row) in enumerate(df.iterrows(), start=1):
                records.append(_normalize_row(row, self.aliases, path, row_number))
        return records

    def load_transcripts(self) -> dict[str, str]:
        transcripts: dict[str, str] = {}
        for path in list_files(self.dataset_config.raw_dir, self.dataset_config.transcript_globs):
            text: str | None = None
            if path.suffix.lower() in TEXT_EXTENSIONS:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception as exc:
                    LOGGER.warning("Ignorando transcript invalido %s: %s", path, exc)
                    continue
            elif path.suffix.lower() in TABLE_EXTENSIONS:
                try:
                    text = _extract_transcript_from_table(read_table(path))
                except Exception as exc:
                    LOGGER.warning("Ignorando transcript tabular invalido %s: %s", path, exc)
                    continue
            text = clean_text(text)
            if not text:
                continue
            for token in _identifier_tokens(path.name) | _identifier_tokens(path.stem):
                transcripts[token] = text
        return transcripts

    def _metadata_indexes(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        by_sample: dict[str, dict[str, Any]] = {}
        by_audio_token: dict[str, dict[str, Any]] = {}
        by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for record in records:
            sample_key = _normalize_key(record.get("sample_id"))
            if sample_key:
                by_sample.setdefault(sample_key, record)
            for token in _identifier_tokens(record.get("audio_file")):
                by_audio_token.setdefault(token, record)
            participant_key = _normalize_key(record.get("participant_id"))
            if participant_key:
                by_participant[participant_key].append(record)
        return by_sample, by_audio_token, by_participant

    def _match_metadata(
        self,
        audio_row: pd.Series,
        by_sample: dict[str, dict[str, Any]],
        by_audio_token: dict[str, dict[str, Any]],
        by_participant: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        sample_key = _normalize_key(audio_row.get("sample_id"))
        if sample_key and sample_key in by_sample:
            return by_sample[sample_key]

        audio_tokens = set()
        audio_tokens.update(_identifier_tokens(audio_row.get("raw_audio_path")))
        audio_tokens.update(_identifier_tokens(audio_row.get("audio_path")))
        for token in audio_tokens:
            if token in by_audio_token:
                return by_audio_token[token]

        participant_key = _normalize_key(audio_row.get("participant_id"))
        if participant_key and len(by_participant.get(participant_key, [])) == 1:
            return by_participant[participant_key][0]
        return None

    def _find_transcript(
        self,
        record: dict[str, Any],
        audio_row: pd.Series | None,
        transcripts: dict[str, str],
    ) -> str | None:
        direct = clean_text(record.get("transcript"))
        if direct:
            return direct

        tokens: set[str] = set()
        if audio_row is not None:
            tokens.update(_identifier_tokens(audio_row.get("sample_id")))
            tokens.update(_identifier_tokens(audio_row.get("raw_audio_path")))
            tokens.update(_identifier_tokens(audio_row.get("audio_path")))
            tokens.update(_identifier_tokens(audio_row.get("participant_id")))
        tokens.update(_identifier_tokens(record.get("sample_id")))
        tokens.update(_identifier_tokens(record.get("audio_file")))
        tokens.update(_identifier_tokens(record.get("participant_id")))

        for token in tokens:
            if token in transcripts:
                return transcripts[token]
        return None

    def _base_unified_record(
        self,
        record: dict[str, Any],
        transcript: str | None,
        audio_row: pd.Series | None = None,
    ) -> dict[str, Any]:
        depression_threshold = float(
            self.config.label_thresholds.get("depression", {}).get("phq_score", 10)
        )
        anxiety_threshold = float(
            self.config.label_thresholds.get("anxiety", {}).get("gad_score", 10)
        )

        sample_id = clean_text(record.get("sample_id"))
        participant_id = clean_text(record.get("participant_id"))
        audio_path = None
        duration_seconds = None

        if audio_row is not None:
            sample_id = clean_text(audio_row.get("sample_id")) or sample_id
            participant_id = participant_id or clean_text(audio_row.get("participant_id"))
            audio_path = clean_text(audio_row.get("audio_path"))
            duration_seconds = parse_float(audio_row.get("duration_seconds"))

        if not sample_id:
            seed = f"{self.dataset_config.key}:{participant_id}:{record.get('source_file')}:{record.get('source_row_number')}"
            sample_id = f"{self.dataset_config.key}_{slugify(participant_id, 'metadata')}_{sha1_short(seed, 8)}"
        if not participant_id:
            participant_id = sample_id

        phq_score = parse_float(record.get("phq_score"))
        gad_score = parse_float(record.get("gad_score"))

        return {
            "sample_id": sample_id,
            "dataset_source": self.dataset_config.source_name,
            "participant_id": participant_id,
            "audio_path": audio_path,
            "transcript": transcript,
            "language": clean_text(record.get("language")) or self.dataset_config.language,
            "gender": normalize_gender(record.get("gender")),
            "age": parse_float(record.get("age")),
            "phq_score": phq_score,
            "gad_score": gad_score,
            "depression_label": standardize_binary_label(
                record.get("depression_label"),
                phq_score,
                depression_threshold,
            ),
            "anxiety_label": standardize_binary_label(
                record.get("anxiety_label"),
                gad_score,
                anxiety_threshold,
            ),
            "voice_risk_label": parse_int_label(record.get("voice_risk_label")),
            "postpartum_depression_label": parse_int_label(
                record.get("postpartum_depression_label")
            ),
            "hormonal_fatigue_label": parse_int_label(record.get("hormonal_fatigue_label")),
            "domestic_violence_label": parse_int_label(record.get("domestic_violence_label")),
            "emotion_label": clean_text(record.get("emotion_label")),
            "duration_seconds": duration_seconds,
            "audio_embedding_path": None,
            "text_embedding_path": None,
        }

    def build_unified_records(self, audio_manifest: pd.DataFrame) -> list[dict[str, Any]]:
        metadata_records = self.load_metadata_records()
        transcripts = self.load_transcripts()
        by_sample, by_audio_token, by_participant = self._metadata_indexes(metadata_records)
        matched_record_ids: set[int] = set()
        unified_records: list[dict[str, Any]] = []

        dataset_audio_manifest = audio_manifest[
            audio_manifest["dataset_key"].astype(str) == self.dataset_config.key
        ] if not audio_manifest.empty and "dataset_key" in audio_manifest.columns else pd.DataFrame()

        for _, audio_row in dataset_audio_manifest.iterrows():
            metadata_record = self._match_metadata(
                audio_row,
                by_sample,
                by_audio_token,
                by_participant,
            ) or {}
            if metadata_record:
                matched_record_ids.add(id(metadata_record))
            transcript = self._find_transcript(metadata_record, audio_row, transcripts)
            unified_records.append(
                self._base_unified_record(metadata_record, transcript, audio_row)
            )

        for metadata_record in metadata_records:
            if id(metadata_record) in matched_record_ids:
                continue
            transcript = self._find_transcript(metadata_record, None, transcripts)
            has_signal = any(
                not missing(metadata_record.get(field))
                for field in (
                    "transcript",
                    "phq_score",
                    "gad_score",
                    "depression_label",
                    "anxiety_label",
                    "voice_risk_label",
                    "postpartum_depression_label",
                    "hormonal_fatigue_label",
                    "domestic_violence_label",
                    "emotion_label",
                )
            )
            if transcript or has_signal:
                unified_records.append(self._base_unified_record(metadata_record, transcript))

        return unified_records


class EATDDatasetAdapter(GenericDatasetAdapter):
    def load_metadata_records(self) -> list[dict[str, Any]]:
        threshold = float(self.dataset_config.label_rules.get("depression_score_threshold", 53))
        score_files = list(
            self.dataset_config.label_rules.get("score_files", ["new_label.txt", "label.txt"])
        )
        records: list[dict[str, Any]] = []
        for participant_dir in sorted(self.dataset_config.raw_dir.glob("**/*_*")):
            if not participant_dir.is_dir():
                continue
            if not re.fullmatch(r"[tv]_\d+", participant_dir.name, flags=re.IGNORECASE):
                continue
            score_path = None
            for score_file in score_files:
                candidate = participant_dir / score_file
                if candidate.exists():
                    score_path = candidate
                    break
            if score_path is None:
                continue
            score = parse_float(score_path.read_text(encoding="utf-8", errors="ignore"))
            if score is None:
                LOGGER.warning("Label EATD invalido em %s.", score_path)
                continue
            records.append(
                {
                    "source_file": score_path.as_posix(),
                    "source_row_number": 1,
                    "participant_id": participant_dir.name,
                    "depression_label": int(score >= threshold),
                    "depression_score": score,
                }
            )
        return records

    def _find_transcript(
        self,
        record: dict[str, Any],
        audio_row: pd.Series | None,
        transcripts: dict[str, str],
    ) -> str | None:
        if audio_row is not None:
            raw_audio_path = clean_text(audio_row.get("raw_audio_path"))
            if raw_audio_path:
                path = Path(raw_audio_path)
                if not path.is_absolute():
                    path = self.config.project_root / path
                stem = path.stem.removesuffix("_out")
                transcript_path = path.with_name(f"{stem}.txt")
                if transcript_path.exists():
                    return clean_text(
                        transcript_path.read_text(encoding="utf-8", errors="ignore")
                    )
        return super()._find_transcript(record, audio_row, transcripts)

    def _base_unified_record(
        self,
        record: dict[str, Any],
        transcript: str | None,
        audio_row: pd.Series | None = None,
    ) -> dict[str, Any]:
        result = super()._base_unified_record(record, transcript, audio_row)
        if audio_row is not None and not result.get("emotion_label"):
            result["emotion_label"] = _emotion_from_audio_stem(audio_row.get("raw_audio_path"))
        return result


def dataset_adapter_for(config: PipelineConfig, dataset_config: DatasetConfig) -> GenericDatasetAdapter:
    if dataset_config.key == "eatd":
        return EATDDatasetAdapter(config, dataset_config)
    return GenericDatasetAdapter(config, dataset_config)


def write_normalized_transcripts(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    result = df.copy()
    transcript_root = ensure_dir(config.processed_root / "transcripts")
    transcript_paths: list[str | None] = []
    for _, row in result.iterrows():
        transcript = clean_text(row.get("transcript"))
        if not transcript:
            transcript_paths.append(None)
            continue
        dataset_source = slugify(row.get("dataset_source"), "dataset")
        sample_id = slugify(row.get("sample_id"), "sample")
        path = transcript_root / dataset_source / f"{sample_id}.txt"
        ensure_dir(path.parent)
        path.write_text(transcript + "\n", encoding="utf-8")
        transcript_paths.append(as_project_path(path, config.project_root))
    result["transcript_path"] = transcript_paths
    return result
