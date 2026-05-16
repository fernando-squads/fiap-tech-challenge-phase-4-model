from __future__ import annotations

import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Iterable, Optional, Union

import librosa
import numpy as np
from scipy.stats import iqr, variation


MULTILABEL_TARGETS = [
    "anxiety",
    "postpartum_depression",
    "hormonal_fatigue",
    "domestic_violence",
]

OUTPUT_COLUMNS = [
    "audio_path",
    "transcription",
    "binary_risk",
    *MULTILABEL_TARGETS,
]


@dataclass(frozen=True)
class VoiceRiskTrainingConfig:
    audio_sample_rate: int = 22_050
    silence_top_db: int = 30
    min_pause_duration_seconds: float = 0.35
    long_pause_threshold_seconds: float = 1.2
    binary_risk_threshold: float = 0.65
    multilabel_risk_threshold: float = 0.55
    require_audio_files: bool = True


@dataclass(frozen=True)
class TextEvidence:
    category: str
    term: str
    snippet: str


@dataclass(frozen=True)
class NLPAnalysis:
    score: float
    categories: dict[str, int]
    category_scores: dict[str, float]
    evidences: list[TextEvidence]
    risk_terms_total: int
    token_count: int


class TextRiskAnalyzer:
    terms: dict[str, list[str]] = {
        "fear": [
            "medo",
            "fear",
            "afraid",
            "scared",
            "frightened",
            "apavorada",
            "assustada",
            "panico",
            "pânico",
            "ameaca",
            "ameaça",
            "tenho medo",
        ],
        "sadness": [
            "triste",
            "sad",
            "sadness",
            "tristeza",
            "depressed",
            "depression",
            "hopeless",
            "hollow",
            "chorar",
            "choro",
            "sem esperança",
            "sem esperanca",
            "vazio",
            "culpa",
            "vontade de sumir",
        ],
        "exhaustion": [
            "cansada",
            "tired",
            "exhausted",
            "exhaustion",
            "exausta",
            "esgotada",
            "no energy",
            "sem energia",
            "can't sleep",
            "cannot sleep",
            "não durmo",
            "nao durmo",
            "insônia",
            "insonia",
            "fadiga",
        ],
        "anxiety": [
            "ansiosa",
            "anxious",
            "anxiety",
            "panic",
            "intrusive thoughts",
            "ansiedade",
            "nervosa",
            "coração acelerado",
            "coracao acelerado",
            "crise",
            "preocupação",
            "preocupacao",
            "taquicardia",
        ],
        "isolation": [
            "sozinha",
            "alone",
            "lonely",
            "isolated",
            "isolada",
            "ninguém",
            "ninguem",
            "sem apoio",
            "abandono",
            "não tenho ajuda",
            "nao tenho ajuda",
        ],
        "coercion": [
            "controla",
            "controls me",
            "controlling",
            "trapped",
            "doesn't listen when i say no",
            "me controla",
            "proíbe",
            "proibe",
            "obrigada",
            "ameaça",
            "ameaca",
            "ciúmes",
            "ciumes",
            "vigia",
        ],
        "violence": [
            "bateu",
            "bruises",
            "grabbed me",
            "hits",
            "hit me",
            "violence",
            "bleeding after sex",
            "bater",
            "agressão",
            "agressao",
            "violência",
            "violencia",
            "empurrou",
            "machucou",
            "gritou",
            "lesão",
            "lesao",
        ],
        "insecurity": [
            "insegura",
            "unsafe",
            "not safe",
            "safe again",
            "scared in my own home",
            "insegurança",
            "inseguranca",
            "perigo",
            "não me sinto segura",
            "nao me sinto segura",
            "medo de voltar",
        ],
        "postpartum": [
            "puerpério",
            "postpartum",
            "post-partum",
            "after birth",
            "puerperio",
            "pós-parto",
            "pos-parto",
            "parto",
            "bebê",
            "bebe",
            "amamentação",
            "amamentacao",
        ],
        "hormonal": [
            "hormônio",
            "hormone",
            "hormonal",
            "menopause",
            "hot flashes",
            "night sweats",
            "hormonio",
            "hormonal",
            "menstruação",
            "menstruacao",
            "ondas de calor",
            "climatério",
            "climaterio",
        ],
    }

    weights: dict[str, float] = {
        "violence": 1.5,
        "coercion": 1.4,
        "fear": 1.2,
        "insecurity": 1.1,
        "sadness": 1.0,
        "anxiety": 1.0,
        "exhaustion": 0.9,
        "isolation": 1.0,
        "postpartum": 0.7,
        "hormonal": 0.6,
    }

    def analyze(self, transcription: str) -> NLPAnalysis:
        text = transcription or ""
        normalized_text = self._normalize(text)
        token_count = len(re.findall(r"\b[\wÀ-ÿ]+\b", text.lower()))

        categories: dict[str, int] = {category: 0 for category in self.terms}
        evidences: list[TextEvidence] = []
        weighted_hits = 0.0

        for category, terms in self.terms.items():
            for term in terms:
                normalized_term = self._normalize(term)
                matches = list(re.finditer(rf"\b{re.escape(normalized_term)}\b", normalized_text))
                if not matches and " " in normalized_term and normalized_term in normalized_text:
                    matches = [re.search(re.escape(normalized_term), normalized_text)]  # type: ignore[list-item]
                for match in [item for item in matches if item is not None]:
                    categories[category] += 1
                    weighted_hits += self.weights.get(category, 1.0)
                    if len(evidences) < 12:
                        evidences.append(
                            TextEvidence(
                                category=category,
                                term=term,
                                snippet=self._snippet(text, match.start(), match.end()),
                            )
                        )

        risk_terms_total = sum(categories.values())
        diversity = sum(1 for count in categories.values() if count > 0)
        score = 1 - math.exp(-weighted_hits / 7.0)
        score = min(1.0, score + min(diversity, 6) * 0.035)

        category_scores = {
            category: min(1.0, count / 3.0)
            for category, count in categories.items()
        }

        return NLPAnalysis(
            score=round(score, 4),
            categories=categories,
            category_scores=category_scores,
            evidences=evidences,
            risk_terms_total=risk_terms_total,
            token_count=token_count,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value.lower())
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _snippet(text: str, start: int, end: int) -> str:
        if not text:
            return ""
        left = max(0, start - 45)
        right = min(len(text), end + 45)
        snippet = text[left:right].strip()
        return re.sub(r"\s+", " ", snippet)


class AcousticFeatureExtractor:
    def __init__(self, config: VoiceRiskTrainingConfig):
        self.config = config

    def extract(self, audio_path: Path, transcription: Optional[str] = None) -> dict[str, float]:
        try:
            y, sr = librosa.load(
                str(audio_path),
                sr=self.config.audio_sample_rate,
                mono=True,
            )
        except Exception as exc:
            raise RuntimeError("Nao foi possivel carregar o audio para extracao.") from exc

        if y.size == 0:
            raise RuntimeError("Audio vazio ou ilegivel.")

        y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        duration = float(librosa.get_duration(y=y, sr=sr))
        features: dict[str, float] = {
            "duration_seconds": duration,
            "sample_rate": float(sr),
        }

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = np.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)
        for idx in range(13):
            coeff = mfcc[idx]
            features[f"mfcc_{idx + 1}_mean"] = self._safe_float(np.mean(coeff))
            features[f"mfcc_{idx + 1}_std"] = self._safe_float(np.std(coeff))

        rms = librosa.feature.rms(y=y)[0]
        rms = np.nan_to_num(rms, nan=0.0, posinf=0.0, neginf=0.0)
        features.update(
            {
                "energy_mean": self._safe_float(np.mean(rms)),
                "energy_std": self._safe_float(np.std(rms)),
                "energy_iqr": self._safe_float(iqr(rms)) if rms.size else 0.0,
                "energy_max": self._safe_float(np.max(rms)) if rms.size else 0.0,
            }
        )

        pitch_values = self._extract_pitch(y, sr)
        features.update(
            {
                "pitch_mean_hz": self._safe_float(np.mean(pitch_values)) if pitch_values.size else 0.0,
                "pitch_std_hz": self._safe_float(np.std(pitch_values)) if pitch_values.size else 0.0,
                "pitch_min_hz": self._safe_float(np.min(pitch_values)) if pitch_values.size else 0.0,
                "pitch_max_hz": self._safe_float(np.max(pitch_values)) if pitch_values.size else 0.0,
            }
        )

        features.update(self._extract_silence_features(y, sr, duration))
        features["speech_rate_wpm"] = self._speech_rate_wpm(transcription, duration, y, sr)
        features["prosodic_variation"] = self._prosodic_variation(pitch_values, rms)
        features["hesitation_score"] = self._hesitation_score(features)
        return {key: self._safe_float(value) for key, value in features.items()}

    def _extract_pitch(self, y: np.ndarray, sr: int) -> np.ndarray:
        try:
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
            f0 = np.asarray(f0, dtype=float)
            f0 = f0[np.isfinite(f0)]
            return f0[(f0 >= 50) & (f0 <= 500)]
        except Exception:
            return np.asarray([], dtype=float)

    def _extract_silence_features(
        self,
        y: np.ndarray,
        sr: int,
        duration: float,
    ) -> dict[str, float]:
        intervals = librosa.effects.split(y, top_db=self.config.silence_top_db)
        silence_durations: list[float] = []
        cursor = 0
        for start, end in intervals:
            if start > cursor:
                silence_durations.append((start - cursor) / sr)
            cursor = end
        if cursor < len(y):
            silence_durations.append((len(y) - cursor) / sr)

        pauses = [
            value
            for value in silence_durations
            if value >= self.config.min_pause_duration_seconds
        ]
        total_silence = float(sum(silence_durations))

        return {
            "pause_count": float(len(pauses)),
            "mean_pause_duration_seconds": float(np.mean(pauses)) if pauses else 0.0,
            "max_pause_duration_seconds": float(np.max(pauses)) if pauses else 0.0,
            "silence_duration_seconds": total_silence,
            "silence_ratio": total_silence / duration if duration > 0 else 0.0,
        }

    def _speech_rate_wpm(
        self,
        transcription: Optional[str],
        duration: float,
        y: np.ndarray,
        sr: int,
    ) -> float:
        if duration <= 0:
            return 0.0

        if transcription:
            words = re.findall(r"\b[\wÀ-ÿ]+\b", transcription.lower())
            if words:
                return float(len(words) / (duration / 60))

        try:
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
            estimated_words = max(len(onsets), 1) / 2.2
            return float(estimated_words / (duration / 60))
        except Exception:
            return 0.0

    def _prosodic_variation(self, pitch_values: np.ndarray, rms: np.ndarray) -> float:
        pitch_component = 0.0
        if pitch_values.size and np.mean(pitch_values) > 0:
            pitch_component = self._safe_float(variation(pitch_values))

        energy_component = 0.0
        if rms.size and np.mean(rms) > 0:
            energy_component = self._safe_float(variation(rms))

        return float(max(0.0, min(1.0, (pitch_component * 0.6) + (energy_component * 0.4))))

    def _hesitation_score(self, features: dict[str, float]) -> float:
        pause_component = min(features.get("pause_count", 0.0) / 8.0, 1.0) * 0.35
        silence_component = min(features.get("silence_ratio", 0.0), 1.0) * 0.40
        long_pause_component = (
            0.25
            if features.get("max_pause_duration_seconds", 0.0)
            >= self.config.long_pause_threshold_seconds
            else 0.0
        )
        return float(min(1.0, pause_component + silence_component + long_pause_component))

    @staticmethod
    def _safe_float(value: Union[float, int, np.floating]) -> float:
        result = float(value)
        if not np.isfinite(result):
            return 0.0
        return result


def build_feature_dict(
    acoustic_features: dict[str, float],
    nlp_analysis: NLPAnalysis,
) -> dict[str, float]:
    features: dict[str, float] = {}

    for key, value in acoustic_features.items():
        features[f"acoustic_{key}"] = _safe_number(value)

    features["text_score"] = _safe_number(nlp_analysis.score)
    features["text_risk_terms_total"] = float(nlp_analysis.risk_terms_total)
    features["text_token_count"] = float(nlp_analysis.token_count)

    for category, count in nlp_analysis.categories.items():
        features[f"text_category_{category}"] = float(count)

    for category, score in nlp_analysis.category_scores.items():
        features[f"text_category_score_{category}"] = _safe_number(score)

    return features


def load_dataset_rows(dataset_path: Path) -> list[dict[str, str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError("Dataset vazio.")
    return rows


def write_dataset(output_path: Path, rows: Iterable[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def resolve_audio_path(dataset_path: Path, audio_path: Optional[str]) -> Optional[Path]:
    if not audio_path:
        return None

    candidate = Path(audio_path)
    if candidate.is_absolute():
        return candidate

    dataset_relative = dataset_path.parent / candidate
    if dataset_relative.exists():
        return dataset_relative

    return Path.cwd() / candidate


def build_training_features(
    dataset_path: Path,
    rows: list[dict[str, str]],
    config: VoiceRiskTrainingConfig,
) -> list[dict[str, float]]:
    analyzer = TextRiskAnalyzer()
    extractor = AcousticFeatureExtractor(config)
    features: list[dict[str, float]] = []
    for row in rows:
        transcription = (row.get("transcription", "") or "").strip()
        if not transcription:
            raise ValueError("Todas as linhas de treinamento devem conter transcription.")

        nlp_analysis = analyzer.analyze(transcription)
        audio_path = resolve_audio_path(dataset_path, row.get("audio_path"))
        if audio_path and audio_path.exists():
            acoustic_features = extractor.extract(audio_path, transcription)
        elif config.require_audio_files:
            raise ValueError("Audio de treinamento nao encontrado para uma linha do dataset.")
        else:
            acoustic_features = fallback_acoustic_features()

        features.append(build_feature_dict(acoustic_features, nlp_analysis))
    return features


def fallback_acoustic_features() -> dict[str, float]:
    return {
        "duration_seconds": 0.0,
        "energy_mean": 0.0,
        "max_pause_duration_seconds": 0.0,
        "silence_ratio": 0.0,
        "hesitation_score": 0.0,
        "prosodic_variation": 0.0,
        "speech_rate_wpm": 0.0,
    }


def _safe_number(value: object) -> float:
    if not isinstance(value, Number):
        return 0.0
    result = float(value)
    if not math.isfinite(result):
        return 0.0
    return result
