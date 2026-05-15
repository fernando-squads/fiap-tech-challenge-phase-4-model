# Unified Mental Health Dataset

Projeto Python para construir um dataset unificado chamado **Unified**, combinando WomanHealthFIAP, EATD-Corpus e EMU para tarefas de deteccao de depressao e ansiedade com voz, texto e metadados.

O repositorio **nao inclui datasets reais**. Coloque os arquivos originais manualmente em `raw/` antes de executar o pipeline.

## Estrutura

```text
mental-health-dataset/
|-- raw/
|   |-- womanhealthfiap/
|   |-- eatd/
|   `-- emu/
|-- processed/
|   |-- audio/
|   |-- transcripts/
|   |-- metadata/
|   |-- embeddings/
|   |-- labels/
|   `-- models/
|-- unified/
|   |-- train.parquet
|   |-- validation.parquet
|   |-- test.parquet
|   `-- schema.json
|-- config/
|   `-- datasets.yaml
|-- src/
`-- README.md
```

## Schema Universal

As saidas `unified/train.parquet`, `unified/validation.parquet` e `unified/test.parquet` seguem estas colunas:

```text
sample_id
dataset_source
participant_id
audio_path
transcript
language
gender
age
phq_score
gad_score
depression_label
anxiety_label
emotion_label
duration_seconds
audio_embedding_path
text_embedding_path
```

Labels binarios usam `1` para positivo e `0` para negativo. Quando nao ha label explicito, o pipeline usa os limiares configurados em `config/datasets.yaml`: `phq_score >= 10` para depressao e `gad_score >= 10` para ansiedade.

## Instalacao

Recomendado usar ambiente virtual.

```bash
cd mental-health-dataset
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para conversao de audio, instale `ffmpeg` no sistema quando possivel. Se `ffmpeg` nao estiver disponivel, o script tenta usar `torchaudio`.

Se `extract_audio_embeddings.py` acusar `ModuleNotFoundError: No module named 'torchcodec'`, rode novamente a etapa de preparo de audio para garantir WAV PCM 16 kHz:

```bash
python src/prepare_audio.py --overwrite
python src/extract_audio_embeddings.py
```

Em ambientes com `torchaudio` recente, tambem pode ser necessario instalar um `torchcodec` compativel com a versao do PyTorch usada pelo ambiente.

## Configuracao

Edite `config/datasets.yaml` para ajustar:

- caminhos dos datasets em `raw/`;
- URLs autorizadas de download em `download.url`;
- idiomas padrao;
- padroes de busca de audio, metadados e transcricoes;
- aliases de colunas para cada dataset;
- proporcoes de split;
- modelos de embedding.

Os adaptadores sao genericos por padrao. Quando um corpus tiver nomes de coluna especificos, adicione aliases em `column_aliases` sem alterar codigo.

Exemplo de bloco de download:

```yaml
download:
  source_type: "direct_url"
  url: "https://exemplo-autorizado.org/dataset.zip"
  hf_dataset_name:
  hf_data_dir:
  load_dataset_example:
  filename: "dataset.zip"
  sha256: "hash_sha256_opcional"
  md5: "md5_opcional"
  extract: true
  auth_token_env: "DATASET_TOKEN"
  expected_files:
    - "**/*.wav"
  manual_instructions: "Mensagem exibida quando nao houver URL configurada."
```

Quando `auth_token_env` estiver definido, o downloader usa `Authorization: Bearer <token>`.

Tipos suportados em `source_type`:

- `direct_url`: baixa um arquivo direto em `download.url`.
- `github_archive`: baixa um arquivo compactado publico, como `refs/heads/master.zip`.
- `huggingface_dataset`: sincroniza um dataset publico com `huggingface_hub.snapshot_download`.
- `manual_huggingface_loader`: registra um loader do Hugging Face que exige `data_dir` local, como EATD-Corpus.
- `restricted_portal`: registra portal de acesso com EULA/login, caso algum dataset futuro exija isso.

## Execucao do Pipeline

Execute a partir da pasta `mental-health-dataset/`.

```bash
python src/download_datasets.py
python src/validate_raw_files.py
python src/prepare_audio.py
python src/build_metadata.py
python src/extract_audio_embeddings.py
python src/extract_text_embeddings.py
python src/create_splits.py
python src/export_unified_dataset.py
python src/train_multimodal_transformer.py
```

Tambem e possivel processar um dataset por vez:

```bash
python src/download_datasets.py --dataset womanhealthfiap
python src/prepare_audio.py --dataset womanhealthfiap
python src/build_metadata.py --dataset womanhealthfiap
```

## O que Cada Etapa Faz

`download_datasets.py` usa `DatasetDownloader` para verificar `raw/<dataset>`, baixar fontes diretas ou datasets do Hugging Face quando possivel, validar `sha256`/`md5` opcionais e extrair pacotes `.zip`/`.tar`.

`validate_raw_files.py` gera `processed/metadata/raw_validation_report.json` com contagens e erros basicos.

`prepare_audio.py` converte audios brutos para WAV mono 16 kHz PCM e salva `processed/metadata/audio_manifest.parquet`.

`build_metadata.py` normaliza metadados, transcricoes e labels, salvando `processed/metadata/metadata.parquet`, `processed/labels/labels.parquet` e transcricoes limpas em `processed/transcripts/`.
No EATD-Corpus, `label.txt`/`new_label.txt` sao propagados para os audios do mesmo participante e `depression_label` usa o limiar configurado em `datasets.eatd.label_rules.depression_score_threshold`.

`extract_audio_embeddings.py` usa `microsoft/wavlm-large` e salva vetores `.npy` em `processed/embeddings/audio/`.
Audios invalidos ou vazios sao ignorados por padrao e registrados em `processed/metadata/audio_embedding_errors.parquet`; use `--fail-fast` para interromper no primeiro erro.

`extract_text_embeddings.py` usa `sentence-transformers/all-mpnet-base-v2` e salva vetores `.npy` em `processed/embeddings/text/`.

`create_splits.py` cria splits por `participant_id`, evitando vazamento entre treino, validacao e teste.

`export_unified_dataset.py` exporta os Parquets finais em `unified/` e atualiza `unified/schema.json`.

`train_multimodal_transformer.py` treina um transformer multimodal de fusao sobre tokens de audio, texto e metadados derivados dos embeddings extraidos.

## Baseline de Treino

Depois de gerar embeddings e exportar o dataset:

```bash
python src/train_baseline.py
```

O baseline usa `LogisticRegression` com `StandardScaler` para prever `depression_label` a partir dos embeddings disponiveis. Artefatos:

- `processed/models/depression_baseline.joblib`
- `processed/models/depression_baseline_metrics.json`

Para escolher modalidades manualmente:

```bash
python src/train_baseline.py --features text
python src/train_baseline.py --features audio
python src/train_baseline.py --features audio,text
```

Com `--features auto`, o script escolhe a modalidade ou combinacao com maior sobreposicao real entre embeddings existentes e labels nos splits de treino e validacao.

## Multimodal Transformer

O treino multimodal usa os embeddings congelados como tokens de entrada:

- token `[CLS]` treinavel;
- token de audio projetado a partir do embedding WavLM;
- token de texto projetado a partir do embedding MPNet;
- token opcional de metadados seguros, como `age` e `duration_seconds`;
- `TransformerEncoder` para fusao;
- cabeca binaria para `depression_label`.

Comando padrao:

```bash
python src/train_multimodal_transformer.py
```

Exemplos:

```bash
python src/train_multimodal_transformer.py --modalities audio,text
python src/train_multimodal_transformer.py --modalities audio,text,metadata
python src/train_multimodal_transformer.py --target anxiety_label --modalities audio,text
python src/train_multimodal_transformer.py --epochs 50 --batch-size 16 --d-model 384 --num-layers 3
```

Artefatos:

- `processed/models/multimodal_transformer/model.pt`
- `processed/models/multimodal_transformer/metrics.json`
- `processed/models/multimodal_transformer/training_config.json`

Esse modelo treina a fusao multimodal. WavLM e MPNet continuam congelados como extratores de embeddings; para fine-tuning end-to-end, seria necessario adicionar um treino que carregue audio/texto bruto diretamente.

## Usar o Modelo em Outro Projeto

Para usar o transformer multimodal fora deste repositorio, copie ou publique estes artefatos:

- `processed/models/multimodal_transformer/model.pt`
- `processed/models/multimodal_transformer/training_config.json`
- `src/multimodal_transformer.py`

O projeto consumidor tambem precisa gerar embeddings com os mesmos modelos usados no treino:

- audio: `microsoft/wavlm-large`
- texto: `sentence-transformers/all-mpnet-base-v2`

Exemplo minimo de inferencia:

```python
from pathlib import Path

import numpy as np
import torch

from multimodal_transformer import MultimodalTransformerClassifier


MODEL_DIR = Path("processed/models/multimodal_transformer")
checkpoint = torch.load(MODEL_DIR / "model.pt", map_location="cpu")

model = MultimodalTransformerClassifier(
    input_dims=checkpoint["input_dims"],
    **checkpoint["model_hparams"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

audio_embedding = np.load("audio_embedding.npy").astype("float32")
text_embedding = np.load("text_embedding.npy").astype("float32")

batch = {
    "audio": torch.from_numpy(audio_embedding).unsqueeze(0),
    "audio_present": torch.tensor([True]),
    "text": torch.from_numpy(text_embedding).unsqueeze(0),
    "text_present": torch.tensor([True]),
}

with torch.inference_mode():
    logits = model(batch)
    probability = torch.sigmoid(logits).item()

print({"depression_probability": probability, "depression_label": int(probability >= 0.5)})
```

Se o modelo foi treinado com `--modalities audio,text,metadata`, inclua tambem o token de metadados. Use as medias e desvios salvos em `checkpoint["data_info"]` para normalizar os valores:

```python
data_info = checkpoint["data_info"]
metadata_columns = data_info["metadata_columns"]
means = data_info["metadata_means"]
stds = data_info["metadata_stds"]

raw_metadata = {
    "age": 32,
    "duration_seconds": 48.5,
}

metadata_values = [
    (float(raw_metadata.get(column, means[column])) - means[column]) / stds[column]
    for column in metadata_columns
]

batch["metadata"] = torch.tensor([metadata_values], dtype=torch.float32)
batch["metadata_present"] = torch.tensor([True])
```

Se uma modalidade estiver ausente, envie um vetor de zeros com a dimensao esperada e marque `<modalidade>_present=False`:

```python
input_dims = checkpoint["input_dims"]
batch["text"] = torch.zeros(1, input_dims["text"])
batch["text_present"] = torch.tensor([False])
```

Para gerar os embeddings no projeto consumidor, reutilize a mesma logica de `src/extract_audio_embeddings.py` e `src/extract_text_embeddings.py`. O modelo multimodal espera vetores ja extraidos; ele nao recebe audio bruto nem texto bruto diretamente.

## Boas Praticas de Dados

- Nao coloque dados brutos, embeddings ou Parquets finais em controle de versao.
- Mantenha `participant_id` consistente, pois ele controla o split sem vazamento.
- Revise `processed/metadata/metadata.parquet` antes de treinar modelos.
- Para datasets sensiveis, armazene dados apenas em ambientes autorizados e criptografados.
