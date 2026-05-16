#!/usr/bin/env python3
"""Build the supervised voice-risk CSV from approved source datasets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config import load_config
from voice_risk_unified import materialize_voice_risk_raw_dataset
from voice_risk import OUTPUT_COLUMNS


@dataclass(frozen=True)
class BuildStats:
    source: str
    included: int
    skipped: int
    note: str


WOMAN_HEALTH_LABELS = {
    "prenatal_normal": {
        "binary_risk": 0,
        "anxiety": 0,
        "postpartum_depression": 0,
        "hormonal_fatigue": 0,
        "domestic_violence": 0,
    },
    "pos_parto_normal": {
        "binary_risk": 0,
        "anxiety": 0,
        "postpartum_depression": 0,
        "hormonal_fatigue": 0,
        "domestic_violence": 0,
    },
    "ginecologica_normal": {
        "binary_risk": 0,
        "anxiety": 0,
        "postpartum_depression": 0,
        "hormonal_fatigue": 0,
        "domestic_violence": 0,
    },
    "prenatal_ansiedade": {
        "binary_risk": 1,
        "anxiety": 1,
        "postpartum_depression": 0,
        "hormonal_fatigue": 0,
        "domestic_violence": 0,
    },
    "pos_parto_depressao": {
        "binary_risk": 1,
        "anxiety": 0,
        "postpartum_depression": 1,
        "hormonal_fatigue": 0,
        "domestic_violence": 0,
    },
    "ginecologica_suspeita_violencia": {
        "binary_risk": 1,
        "anxiety": 0,
        "postpartum_depression": 0,
        "hormonal_fatigue": 0,
        "domestic_violence": 1,
    },
    "menopausa": {
        "binary_risk": 1,
        "anxiety": 0,
        "postpartum_depression": 0,
        "hormonal_fatigue": 1,
        "domestic_violence": 0,
    },
}


def build_womanhealthfiap(root: Path) -> tuple[list[dict[str, str]], BuildStats]:
    metadata_path = root / "dataset_pacientes.csv"
    if not metadata_path.exists():
        return [], BuildStats(
            source="WomanHealthFIAP",
            included=0,
            skipped=0,
            note=f"metadata nao encontrado em {metadata_path}",
        )

    rows: list[dict[str, str]] = []
    skipped = 0
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for source_row in reader:
            category = (source_row.get("categoria_consulta") or "").strip()
            labels = WOMAN_HEALTH_LABELS.get(category)
            audio_relative = (source_row.get("audio_path") or "").strip()
            transcription = (source_row.get("relato_texto") or "").strip()
            audio_path = root / audio_relative

            if not labels or not transcription or not audio_path.exists():
                skipped += 1
                continue

            row = {
                "audio_path": str(audio_path),
                "transcription": transcription,
            }
            row.update({key: str(value) for key, value in labels.items()})
            rows.append(row)

    return rows, BuildStats(
        source="WomanHealthFIAP",
        included=len(rows),
        skipped=skipped,
        note="incluido como base principal conforme categorias clinicas do dataset",
    )


def build_depac(root: Path) -> tuple[list[dict[str, str]], BuildStats]:
    metadata_files = _metadata_files(root)
    if not metadata_files:
        return [], BuildStats("DEPAC", 0, 0, f"metadata nao encontrado em {root}")

    rows: list[dict[str, str]] = []
    skipped = 0
    for metadata_path in metadata_files:
        with metadata_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for source_row in reader:
                audio_value = _first_present(
                    source_row,
                    ["audio_path", "audio", "filename", "file", "path", "wav_path"],
                )
                transcription = _first_present(
                    source_row,
                    ["transcription", "transcript", "text", "utterance", "relato_texto"],
                )
                audio_path = _resolve_source_path(metadata_path.parent, audio_value)

                if not audio_path or not audio_path.exists() or not transcription:
                    skipped += 1
                    continue

                labels = _clinical_labels_from_row(source_row)
                if labels is None:
                    skipped += 1
                    continue

                row = {"audio_path": str(audio_path), "transcription": transcription}
                row.update({key: str(value) for key, value in labels.items()})
                rows.append(row)

    return rows, BuildStats(
        "DEPAC",
        len(rows),
        skipped,
        "incluido quando metadata local contem audio, transcricao e labels clinicos",
    )


def report_auxiliary_emotion_dataset(source: str, root: Path) -> BuildStats:
    if not root.exists():
        return BuildStats(source, 0, 0, f"diretorio nao encontrado em {root}")

    audio_count = sum(1 for path in root.rglob("*") if path.suffix.lower() in {".wav", ".mp3", ".m4a"})
    return BuildStats(
        source,
        0,
        audio_count,
        (
            "nao incluido no CSV clinico supervisionado: labels de emocao/prosodia "
            "nao equivalem a ansiedade, depressao pos-parto, fadiga hormonal ou violencia domestica"
        ),
    )


def write_training_dataset(output_path: Path, rows: Iterable[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: (row["binary_risk"], row["audio_path"]))
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted_rows)


def write_report(report_path: Path, stats: Iterable[BuildStats], output_path: Path, total_rows: int) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Voice Risk Training Dataset Build Report",
        "",
        f"Arquivo gerado: `{output_path}`",
        f"Linhas incluidas: {total_rows}",
        "",
    ]
    for item in stats:
        lines.extend(
            [
                f"## {item.source}",
                "",
                f"- Incluidas: {item.included}",
                f"- Ignoradas/nao incluidas: {item.skipped}",
                f"- Nota: {item.note}",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _metadata_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    preferred_names = {
        "metadata.csv",
        "dataset.csv",
        "labels.csv",
        "annotations.csv",
        "train.csv",
    }
    files = [path for path in root.rglob("*.csv") if path.name.lower() in preferred_names]
    return files or list(root.rglob("*.csv"))


def _first_present(row: dict[str, str], candidates: list[str]) -> str:
    normalized = {key.lower(): value for key, value in row.items()}
    for candidate in candidates:
        value = normalized.get(candidate.lower())
        if value:
            return value.strip()
    return ""


def _resolve_source_path(metadata_dir: Path, audio_value: str) -> Path | None:
    if not audio_value:
        return None
    candidate = Path(audio_value)
    if candidate.is_absolute():
        return candidate
    return metadata_dir / candidate


def _clinical_labels_from_row(row: dict[str, str]) -> dict[str, int] | None:
    anxiety = _binary_value(row, ["anxiety", "anxiety_label", "gad_positive", "gad7_binary"])
    postpartum = _binary_value(
        row,
        ["postpartum_depression", "postpartum", "ppd", "ppd_label"],
    )
    hormonal = _binary_value(row, ["hormonal_fatigue", "hormonal", "fatigue_hormonal"])
    domestic = _binary_value(row, ["domestic_violence", "violence", "dv", "ipv"])
    binary = _binary_value(row, ["binary_risk", "risk", "label", "clinical_risk"])

    depression = _binary_value(row, ["depression", "depressive", "phq_positive", "phq9_binary"])
    if binary is None:
        binary = any(value == 1 for value in [anxiety, postpartum, hormonal, domestic, depression])

    if binary is None:
        return None

    return {
        "binary_risk": int(binary),
        "anxiety": int(anxiety or 0),
        "postpartum_depression": int(postpartum or 0),
        "hormonal_fatigue": int(hormonal or 0),
        "domestic_violence": int(domestic or 0),
    }


def _binary_value(row: dict[str, str], columns: list[str]):
    normalized = {key.lower(): value for key, value in row.items()}
    for column in columns:
        value = normalized.get(column.lower())
        if value is None or str(value).strip() == "":
            continue
        value = str(value).strip().lower()
        if value in {"1", "true", "yes", "sim", "positive", "risk", "risco"}:
            return 1
        if value in {"0", "false", "no", "nao", "não", "negative", "normal", "nao_risco"}:
            return 0
        try:
            return 1 if float(value) >= 0.5 else 0
        except ValueError:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--skip-raw-materialization", action="store_true")
    parser.add_argument("--womanhealthfiap-root", type=Path, default=None)
    parser.add_argument("--depac-root", type=Path, default=None)
    parser.add_argument("--cremad-root", type=Path, default=None)
    parser.add_argument("--msp-podcast-root", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output = args.output or config.processed_root / "voice_risk" / "training_dataset.csv"
    report = args.report or config.processed_root / "voice_risk" / "training_dataset_build_report.md"
    raw_output_dir = args.raw_output_dir or config.raw_root / "voice_risk_training"
    womanhealthfiap_root = args.womanhealthfiap_root or config.raw_root / "womanhealthfiap"
    depac_root = args.depac_root or config.raw_root / "depac"
    cremad_root = args.cremad_root or config.raw_root / "crema-d"
    msp_podcast_root = args.msp_podcast_root or config.raw_root / "msp-podcast"

    all_rows: list[dict[str, str]] = []
    stats: list[BuildStats] = []

    woman_rows, woman_stats = build_womanhealthfiap(womanhealthfiap_root)
    all_rows.extend(woman_rows)
    stats.append(woman_stats)

    depac_rows, depac_stats = build_depac(depac_root)
    all_rows.extend(depac_rows)
    stats.append(depac_stats)

    stats.append(report_auxiliary_emotion_dataset("CREMA-D", cremad_root))
    stats.append(report_auxiliary_emotion_dataset("MSP-Podcast", msp_podcast_root))

    write_training_dataset(output, all_rows)
    write_report(report, stats, output, len(all_rows))
    if not args.skip_raw_materialization:
        metadata_path = materialize_voice_risk_raw_dataset(
            source_csv=output,
            raw_output_dir=raw_output_dir,
            dataset_key="voice_risk_training",
        )
        print(f"Dataset raw Unified: {metadata_path}")

    print(f"Gerado {output} com {len(all_rows)} linhas.")
    print(f"Relatorio: {report}")


if __name__ == "__main__":
    main()
