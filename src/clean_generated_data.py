from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config import PipelineConfig, load_config
from logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)
Mode = Literal["path", "contents"]
KEEP_NAMES = {".gitkeep"}


@dataclass(frozen=True)
class CleanTarget:
    path: Path
    mode: Mode
    description: str


@dataclass(frozen=True)
class CleanSummary:
    planned: int
    removed: int
    skipped: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove artefatos gerados pelo pipeline para permitir uma "
            "regeneracao completa dos dados e modelos."
        )
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lista o que seria removido, sem apagar arquivos.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirma a remocao dos artefatos gerados.",
    )
    parser.add_argument(
        "--keep-voice-risk-raw",
        action="store_true",
        help="Preserva raw/voice_risk_training e raw/voice_risk_synthetic.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_clean_targets(
    config: PipelineConfig,
    *,
    keep_voice_risk_raw: bool = False,
) -> list[CleanTarget]:
    targets: list[CleanTarget] = []
    if not keep_voice_risk_raw:
        targets.extend(
            [
                CleanTarget(
                    config.raw_root / "voice_risk_training",
                    "path",
                    "dataset raw gerado por build_voice_risk_dataset.py",
                ),
                CleanTarget(
                    config.raw_root / "voice_risk_synthetic",
                    "path",
                    "dataset raw gerado por generate_voice_risk_synthetic_data.py",
                ),
            ]
        )

    targets.extend(
        [
            CleanTarget(
                config.processed_root / "voice_risk",
                "path",
                "CSVs e audios sinteticos de voice risk",
            ),
            CleanTarget(
                config.processed_root / "audio",
                "contents",
                "audios WAV padronizados por prepare_audio.py",
            ),
            CleanTarget(
                config.processed_root / "transcripts",
                "contents",
                "transcricoes normalizadas por build_metadata.py",
            ),
            CleanTarget(
                config.processed_root / "metadata",
                "contents",
                "manifestos, relatorios, metadata e splits do pipeline",
            ),
            CleanTarget(
                config.processed_root / "labels",
                "contents",
                "labels padronizados por build_metadata.py",
            ),
            CleanTarget(
                config.processed_root / "embeddings",
                "contents",
                "embeddings de audio e texto",
            ),
            CleanTarget(
                config.processed_root / "models",
                "contents",
                "modelos e metricas gerados pelos scripts de treino",
            ),
            CleanTarget(
                config.unified_root,
                "contents",
                "Parquets finais e schema.json do Unified",
            ),
        ]
    )
    return targets


def clean_targets(
    targets: list[CleanTarget],
    config: PipelineConfig,
    *,
    dry_run: bool,
) -> CleanSummary:
    planned = 0
    removed = 0
    skipped = 0
    safe_roots = _safe_roots(config)

    for target in targets:
        _assert_safe_target(target, safe_roots, config.project_root)
        LOGGER.info("%s: %s", "Planejando remover" if dry_run else "Removendo", target.description)
        current_planned, current_removed, current_skipped = _clean_target(target, dry_run=dry_run)
        planned += current_planned
        removed += current_removed
        skipped += current_skipped

    return CleanSummary(planned=planned, removed=removed, skipped=skipped)


def _clean_target(target: CleanTarget, *, dry_run: bool) -> tuple[int, int, int]:
    path = target.path
    if target.mode == "path":
        return _remove_path(path, dry_run=dry_run)
    return _remove_contents(path, dry_run=dry_run)


def _remove_contents(root: Path, *, dry_run: bool) -> tuple[int, int, int]:
    if not root.exists():
        LOGGER.info("Ignorando inexistente: %s", root)
        return 0, 0, 1

    planned = 0
    removed = 0
    skipped = 0
    for child in sorted(root.iterdir(), key=lambda item: item.as_posix()):
        if child.name in KEEP_NAMES:
            skipped += 1
            LOGGER.info("Preservando: %s", child)
            continue
        child_planned, child_removed, child_skipped = _remove_path(child, dry_run=dry_run)
        planned += child_planned
        removed += child_removed
        skipped += child_skipped
    return planned, removed, skipped


def _remove_path(path: Path, *, dry_run: bool) -> tuple[int, int, int]:
    if not path.exists():
        LOGGER.info("Ignorando inexistente: %s", path)
        return 0, 0, 1
    if dry_run:
        LOGGER.info("Removeria: %s", path)
        return 1, 0, 0
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    LOGGER.info("Removido: %s", path)
    return 1, 1, 0


def _safe_roots(config: PipelineConfig) -> list[Path]:
    return [
        config.raw_root.resolve(),
        config.processed_root.resolve(),
        config.unified_root.resolve(),
    ]


def _assert_safe_target(target: CleanTarget, safe_roots: list[Path], project_root: Path) -> None:
    path = target.path.resolve()
    project_root = project_root.resolve()
    if path in {Path("/"), project_root, project_root.parent}:
        raise ValueError(f"Alvo de limpeza muito amplo: {path}")

    matching_roots = [root for root in safe_roots if path == root or path.is_relative_to(root)]
    if not matching_roots:
        raise ValueError(f"Alvo fora das raizes configuradas do projeto: {path}")

    if target.mode == "path" and path in safe_roots:
        raise ValueError(f"Modo path nao pode remover uma raiz completa do projeto: {path}")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if not args.dry_run and not args.yes:
        raise SystemExit(
            "Use --dry-run para visualizar a limpeza ou --yes para confirmar a remocao."
        )

    config = load_config(args.config)
    targets = build_clean_targets(config, keep_voice_risk_raw=args.keep_voice_risk_raw)
    summary = clean_targets(targets, config, dry_run=args.dry_run)

    if args.dry_run:
        LOGGER.info(
            "Dry-run concluido: %d alvos seriam removidos, %d preservados/ignorados.",
            summary.planned,
            summary.skipped,
        )
    else:
        LOGGER.info(
            "Limpeza concluida: %d alvos removidos, %d preservados/ignorados.",
            summary.removed,
            summary.skipped,
        )


if __name__ == "__main__":
    main()
