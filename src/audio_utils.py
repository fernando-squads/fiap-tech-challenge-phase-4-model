from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

LOGGER = logging.getLogger(__name__)


def _is_torchcodec_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    if "torchcodec" in message:
        return True
    cause = getattr(exc, "__cause__", None)
    return bool(cause and "torchcodec" in str(cause).lower())


def convert_to_wav_16k_mono(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return

    if shutil.which("ffmpeg"):
        command = [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]
        subprocess.run(command, check=True)
        return

    LOGGER.warning("ffmpeg nao encontrado; usando soundfile/torchaudio para conversao.")
    import soundfile as sf

    waveform = load_audio_16k_mono(input_path)
    sf.write(output_path, waveform.cpu().numpy(), 16000, subtype="PCM_16")


def get_audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        with sf.SoundFile(path) as audio_file:
            return float(len(audio_file) / audio_file.samplerate)
    except Exception:
        pass

    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav_file:
            return float(wav_file.getnframes() / wav_file.getframerate())

    import torchaudio

    info = torchaudio.info(str(path))
    return float(info.num_frames / info.sample_rate)


def _resample_waveform(
    waveform: "torch.Tensor",
    source_sample_rate: int,
    target_sample_rate: int = 16000,
) -> "torch.Tensor":
    if source_sample_rate == target_sample_rate:
        return waveform

    try:
        import torchaudio

        return torchaudio.functional.resample(
            waveform.unsqueeze(0),
            source_sample_rate,
            target_sample_rate,
        ).squeeze(0)
    except ModuleNotFoundError as exc:
        if exc.name != "torchcodec":
            raise
        LOGGER.warning(
            "torchaudio solicitou torchcodec para resample; usando fallback com torch.interpolate."
        )
    except Exception:
        LOGGER.warning("Falha no resample via torchaudio; usando fallback com torch.interpolate.")

    import torch

    target_length = max(1, round(waveform.numel() * target_sample_rate / source_sample_rate))
    return torch.nn.functional.interpolate(
        waveform.view(1, 1, -1),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).view(-1)


def _load_audio_with_soundfile(path: Path) -> tuple["torch.Tensor", int]:
    import soundfile as sf
    import torch

    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"Audio vazio: {path}")
    mono_audio = np.mean(audio, axis=1)
    waveform = torch.from_numpy(mono_audio.copy())
    return waveform, int(sample_rate)


def _load_wav_pcm_with_wave(path: Path) -> tuple["torch.Tensor", int]:
    import torch

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if not frames:
        raise ValueError(f"Audio vazio: {path}")

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        audio = audio / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        audio_int = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        audio_int = np.where(audio_int & 0x800000, audio_int | ~0xFFFFFF, audio_int)
        audio = audio_int.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32)
        audio = audio / 2147483648.0
    else:
        raise ValueError(f"Largura de amostra WAV nao suportada: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(audio.copy()), int(sample_rate)


def load_audio_16k_mono(path: Path) -> "torch.Tensor":
    soundfile_error: Exception | None = None
    wave_error: Exception | None = None

    try:
        waveform, sample_rate = _load_audio_with_soundfile(path)
        return _resample_waveform(waveform, sample_rate, 16000)
    except Exception as exc:
        soundfile_error = exc
        LOGGER.debug("soundfile nao carregou %s: %s", path, soundfile_error)

    if path.suffix.lower() == ".wav":
        try:
            waveform, sample_rate = _load_wav_pcm_with_wave(path)
            return _resample_waveform(waveform, sample_rate, 16000)
        except Exception as exc:
            wave_error = exc
            LOGGER.debug("wave nao carregou %s: %s", path, wave_error)
            raise RuntimeError(
                "Nao foi possivel carregar WAV via soundfile nem via leitor PCM nativo. "
                f"Arquivo: {path}. "
                f"Erro soundfile: {soundfile_error}. "
                f"Erro wave: {wave_error}. "
                "Rode `python src/prepare_audio.py --overwrite` para recriar WAV PCM 16 kHz."
            ) from exc

    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        return _resample_waveform(waveform, sample_rate, 16000)
    except (ImportError, ModuleNotFoundError) as exc:
        if _is_torchcodec_error(exc):
            raise RuntimeError(
                "Nao foi possivel carregar o audio sem torchcodec. "
                f"Arquivo: {path}. "
                f"Erro soundfile: {soundfile_error}. "
                f"Erro wave: {wave_error}. "
                "Rode `python src/prepare_audio.py --overwrite` para recriar WAV PCM 16 kHz "
                "ou instale um torchcodec compativel com sua versao do PyTorch."
            ) from exc
        raise
