from __future__ import annotations

import hashlib
import logging
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from config import DatasetConfig, PipelineConfig
from io_utils import as_project_path, ensure_dir

LOGGER = logging.getLogger(__name__)


class DatasetDownloadError(RuntimeError):
    """Erro recuperavel ao tentar baixar ou preparar um dataset."""


@dataclass(frozen=True)
class DownloadResult:
    dataset_key: str
    dataset_source: str
    status: str
    raw_dir: str
    archive_path: str | None = None
    message: str | None = None


class DatasetDownloader:
    """Baixa datasets ausentes usando a configuracao declarada em datasets.yaml."""

    PLACEHOLDER_NAMES = {".gitkeep", ".gitignore", "README.md", "README.txt"}

    def __init__(
        self,
        config: PipelineConfig,
        cache_dir: Path | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir or config.processed_root / "downloads"
        self.chunk_size = chunk_size

    def dataset_exists(self, dataset_config: DatasetConfig) -> bool:
        raw_dir = dataset_config.raw_dir
        if not raw_dir.exists():
            return False

        expected_files = dataset_config.download.expected_files
        if expected_files:
            return all(any(raw_dir.glob(pattern)) for pattern in expected_files)

        return any(
            path.is_file() and path.name not in self.PLACEHOLDER_NAMES
            for path in raw_dir.rglob("*")
        )

    def ensure_available(
        self,
        dataset_keys: list[str] | None = None,
        force: bool = False,
        fail_fast: bool = False,
    ) -> list[DownloadResult]:
        keys = dataset_keys or list(self.config.datasets.keys())
        results: list[DownloadResult] = []
        for dataset_key in keys:
            try:
                results.append(self.download_if_missing(dataset_key, force=force))
            except Exception as exc:
                LOGGER.exception("Falha ao preparar dataset %s.", dataset_key)
                if fail_fast:
                    raise
                dataset_config = self.config.datasets[dataset_key]
                results.append(
                    DownloadResult(
                        dataset_key=dataset_key,
                        dataset_source=dataset_config.source_name,
                        status="error",
                        raw_dir=as_project_path(dataset_config.raw_dir, self.config.project_root),
                        message=str(exc),
                    )
                )
        return results

    def download_if_missing(self, dataset_key: str, force: bool = False) -> DownloadResult:
        if dataset_key not in self.config.datasets:
            raise KeyError(f"Dataset desconhecido: {dataset_key}")

        dataset_config = self.config.datasets[dataset_key]
        ensure_dir(dataset_config.raw_dir)

        if self.dataset_exists(dataset_config) and not force:
            message = "Dataset ja encontrado no projeto."
            LOGGER.info("%s: %s", dataset_config.source_name, message)
            return DownloadResult(
                dataset_key=dataset_key,
                dataset_source=dataset_config.source_name,
                status="present",
                raw_dir=as_project_path(dataset_config.raw_dir, self.config.project_root),
                message=message,
            )

        download_config = dataset_config.download
        if download_config.source_type in {
            "manual",
            "manual_huggingface_loader",
            "restricted_portal",
        }:
            instructions = download_config.manual_instructions or (
                f"Baixe o dataset manualmente em {download_config.url} e coloque os arquivos em "
                f"{dataset_config.raw_dir}."
            )
            LOGGER.warning("%s: %s", dataset_config.source_name, instructions)
            return DownloadResult(
                dataset_key=dataset_key,
                dataset_source=dataset_config.source_name,
                status="manual_required",
                raw_dir=as_project_path(dataset_config.raw_dir, self.config.project_root),
                message=instructions,
            )

        if download_config.source_type == "huggingface_dataset":
            self._download_huggingface_dataset(dataset_config)
            status = "downloaded"
            archive_path = None
        else:
            if not download_config.url:
                instructions = (
                    "Nenhuma URL de download configurada. Coloque os arquivos manualmente em "
                    f"{dataset_config.raw_dir} ou configure datasets.{dataset_key}.download.url."
                )
                LOGGER.warning("%s: %s", dataset_config.source_name, instructions)
                return DownloadResult(
                    dataset_key=dataset_key,
                    dataset_source=dataset_config.source_name,
                    status="manual_required",
                    raw_dir=as_project_path(dataset_config.raw_dir, self.config.project_root),
                    message=instructions,
                )
            archive_path = self._download_archive(dataset_config)
            if download_config.sha256:
                self._verify_sha256(archive_path, download_config.sha256)
            if download_config.md5:
                self._verify_md5(archive_path, download_config.md5)

            if download_config.extract:
                self._extract_archive(archive_path, dataset_config.raw_dir)
                status = "downloaded_and_extracted"
            else:
                destination = dataset_config.raw_dir / archive_path.name
                if archive_path.resolve() != destination.resolve():
                    shutil.copy2(archive_path, destination)
                status = "downloaded"

        message = "Dataset baixado com sucesso."
        LOGGER.info("%s: %s", dataset_config.source_name, message)
        return DownloadResult(
            dataset_key=dataset_key,
            dataset_source=dataset_config.source_name,
            status=status,
            raw_dir=as_project_path(dataset_config.raw_dir, self.config.project_root),
            archive_path=as_project_path(archive_path, self.config.project_root)
            if archive_path
            else None,
            message=message,
        )

    def _download_huggingface_dataset(self, dataset_config: DatasetConfig) -> None:
        download_config = dataset_config.download
        if not download_config.hf_dataset_name:
            raise DatasetDownloadError("hf_dataset_name ausente para source_type=huggingface_dataset.")

        token = os.getenv(download_config.auth_token_env) if download_config.auth_token_env else None
        ensure_dir(dataset_config.raw_dir)
        LOGGER.info(
            "%s: sincronizando Hugging Face dataset %s em %s.",
            dataset_config.source_name,
            download_config.hf_dataset_name,
            dataset_config.raw_dir,
        )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise DatasetDownloadError(
                "Instale huggingface-hub para baixar datasets do Hugging Face."
            ) from exc

        snapshot_download(
            repo_id=download_config.hf_dataset_name,
            repo_type="dataset",
            local_dir=str(dataset_config.raw_dir),
            token=token,
        )

    def _download_archive(self, dataset_config: DatasetConfig) -> Path:
        download_config = dataset_config.download
        if not download_config.url:
            raise DatasetDownloadError("URL de download ausente.")

        ensure_dir(self.cache_dir)
        filename = download_config.filename or self._filename_from_url(
            download_config.url,
            fallback=f"{dataset_config.key}.download",
        )
        archive_path = self.cache_dir / filename

        request = urllib.request.Request(download_config.url)
        if download_config.auth_token_env:
            token = os.getenv(download_config.auth_token_env)
            if not token:
                raise DatasetDownloadError(
                    f"Variavel de ambiente {download_config.auth_token_env} nao configurada."
                )
            request.add_header("Authorization", f"Bearer {token}")

        LOGGER.info(
            "%s: baixando %s para %s.",
            dataset_config.source_name,
            download_config.url,
            archive_path,
        )
        try:
            with urllib.request.urlopen(request) as response:
                total = int(response.headers.get("Content-Length", "0") or "0")
                self._stream_to_file(response, archive_path, total)
        except urllib.error.URLError as exc:
            raise DatasetDownloadError(f"Falha no download de {download_config.url}: {exc}") from exc

        return archive_path

    def _stream_to_file(self, response: object, output_path: Path, total: int) -> None:
        from tqdm import tqdm

        with output_path.open("wb") as output_file:
            with tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            ) as progress:
                while True:
                    chunk = response.read(self.chunk_size)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    progress.update(len(chunk))

    def _verify_sha256(self, path: Path, expected_sha256: str) -> None:
        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(self.chunk_size), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256.lower() != expected_sha256.lower():
            raise DatasetDownloadError(
                f"Checksum invalido para {path.name}: esperado={expected_sha256}, obtido={actual_sha256}"
            )

    def _verify_md5(self, path: Path, expected_md5: str) -> None:
        digest = hashlib.md5()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(self.chunk_size), b""):
                digest.update(chunk)
        actual_md5 = digest.hexdigest()
        if actual_md5.lower() != expected_md5.lower():
            raise DatasetDownloadError(
                f"MD5 invalido para {path.name}: esperado={expected_md5}, obtido={actual_md5}"
            )

    def _extract_archive(self, archive_path: Path, raw_dir: Path) -> None:
        ensure_dir(raw_dir)
        try:
            shutil.unpack_archive(str(archive_path), str(raw_dir))
        except (shutil.ReadError, ValueError) as exc:
            raise DatasetDownloadError(
                f"Arquivo {archive_path.name} nao parece ser um pacote suportado. "
                "Use download.extract=false para baixar arquivos individuais."
            ) from exc

    def _filename_from_url(self, url: str, fallback: str) -> str:
        parsed = urlparse(url)
        name = unquote(Path(parsed.path).name)
        return name or fallback
