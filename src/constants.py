from __future__ import annotations

from typing import Final


UNIVERSAL_SCHEMA_COLUMNS: Final[list[str]] = [
    "sample_id",
    "dataset_source",
    "participant_id",
    "audio_path",
    "transcript",
    "language",
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
    "duration_seconds",
    "audio_embedding_path",
    "text_embedding_path",
]


SCHEMA_DESCRIPTIONS: Final[dict[str, str]] = {
    "sample_id": "Identificador unico e estavel da amostra no dataset unificado.",
    "dataset_source": "Nome do dataset de origem.",
    "participant_id": "Identificador do participante; usado para splits sem vazamento.",
    "audio_path": "Caminho relativo para o audio WAV mono 16 kHz PCM processado.",
    "transcript": "Transcricao textual associada a amostra, quando disponivel.",
    "language": "Idioma principal da amostra em tag BCP-47.",
    "gender": "Genero normalizado: female, male, non_binary, unknown ou nulo.",
    "age": "Idade do participante, quando disponivel.",
    "phq_score": "Pontuacao PHQ/PHQ-8/PHQ-9 normalizada como numero.",
    "gad_score": "Pontuacao GAD/GAD-7 normalizada como numero.",
    "depression_label": "Rotulo binario de depressao: 1 positivo, 0 negativo.",
    "anxiety_label": "Rotulo binario de ansiedade: 1 positivo, 0 negativo.",
    "voice_risk_label": "Rotulo binario geral de risco de voz: 1 positivo, 0 negativo.",
    "postpartum_depression_label": "Rotulo binario de risco de depressao pos-parto.",
    "hormonal_fatigue_label": "Rotulo binario de risco de fadiga hormonal.",
    "domestic_violence_label": "Rotulo binario de risco de violencia domestica.",
    "emotion_label": "Rotulo emocional original ou normalizado, quando existir.",
    "duration_seconds": "Duracao do audio processado em segundos.",
    "audio_embedding_path": "Caminho relativo para embedding .npy extraido com WavLM.",
    "text_embedding_path": "Caminho relativo para embedding .npy extraido com MPNet.",
}


SCHEMA_DTYPES: Final[dict[str, str]] = {
    "sample_id": "string",
    "dataset_source": "string",
    "participant_id": "string",
    "audio_path": "string",
    "transcript": "string",
    "language": "string",
    "gender": "string",
    "age": "float",
    "phq_score": "float",
    "gad_score": "float",
    "depression_label": "integer_nullable",
    "anxiety_label": "integer_nullable",
    "voice_risk_label": "integer_nullable",
    "postpartum_depression_label": "integer_nullable",
    "hormonal_fatigue_label": "integer_nullable",
    "domestic_violence_label": "integer_nullable",
    "emotion_label": "string",
    "duration_seconds": "float",
    "audio_embedding_path": "string",
    "text_embedding_path": "string",
}


DATASET_SOURCE_NAMES: Final[dict[str, str]] = {
    "womanhealthfiap": "WomanHealthFIAP",
    "eatd": "EATD-Corpus",
    "emu": "EMU",
    "voice_risk_training": "VoiceRiskTraining",
    "voice_risk_synthetic": "VoiceRiskSynthetic",
}
