# Unificador de dataset's e treinador de modelos

Projeto Python para construir, unificar e treinar modelos sobre datasets de saude mental baseados em voz, texto e metadados.

O fluxo real do projeto e:

1. Baixar ou localizar os datasets brutos configurados em `config/datasets.yaml`.
2. Processar os datasets baixados, gerar datasets derivados de voice risk e materializar tudo em um formato bruto padronizado.
3. Unificar os dados processados no dataset **Unified**, com splits por `participant_id` para evitar data leakage.
4. Treinar modelos usando o dataset unificado e os embeddings extraidos.
5. Mostrar como carregar o modelo treinado em outro projeto para inferencia.

O repositorio **nao versiona datasets reais, embeddings, Parquets finais nem modelos treinados**. Esses artefatos sao gerados localmente dentro de `raw/`, `processed/` e `unified/`.

## Estrutura

```text
fiap-tech-challenge-phase-4-model/
|-- raw/
|   |-- womanhealthfiap/
|   |-- eatd/
|   |-- emu/
|   |-- voice_risk_training/
|   `-- voice_risk_synthetic/
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

## Datasets

O projeto trabalha com estas fontes:

| Dataset | Origem | Papel no pipeline |
| --- | --- | --- |
| `WomanHealthFIAP` | Hugging Face `brunoretiro/womanhealthfiap` | Dataset real baixado para `raw/womanhealthfiap/`. |
| `EATD-Corpus` | Corpus externo referenciado pelo loader `jimregan/eatd_corpus` | Dataset real que exige arquivos locais em `raw/eatd/EATD-Corpus/`. |
| `EMU` | GitHub `mltlachac/EMU` | Dataset real baixado e extraido em `raw/emu/`. |
| `VoiceRiskTraining` | Gerado por `src/build_voice_risk_dataset.py` | Dataset derivado local para treino de risco de voz, materializado em `raw/voice_risk_training/`. |
| `VoiceRiskSynthetic` | Gerado por `src/generate_voice_risk_synthetic_data.py` | Dataset sintetico de desenvolvimento, materializado em `raw/voice_risk_synthetic/`. |

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
voice_risk_label
postpartum_depression_label
hormonal_fatigue_label
domestic_violence_label
emotion_label
duration_seconds
audio_embedding_path
text_embedding_path
```

Labels binarios usam `1` para positivo e `0` para negativo. Quando nao ha label explicito, o pipeline usa os limiares configurados em `config/datasets.yaml`: `phq_score >= 10` para depressao e `gad_score >= 10` para ansiedade. Os campos `voice_risk_label`, `postpartum_depression_label`, `hormonal_fatigue_label` e `domestic_violence_label` sao preservados para treinos de voice risk.

## Instalacao

Recomendado usar ambiente virtual.

```bash
cd fiap-tech-challenge-phase-4-model
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

## Execucao Completa

Execute os comandos a partir da raiz do repositorio.

### 0. Limpar Artefatos Gerados

Quando quiser rodar o pipeline do zero, limpe os artefatos gerados pelas etapas de processamento, unificacao e treino:

```bash
python src/clean_generated_data.py --dry-run
python src/clean_generated_data.py --yes
```

O `--dry-run` mostra o que seria removido sem apagar nada. O `--yes` confirma a remocao.

Esse script remove:

- `raw/voice_risk_training/` e `raw/voice_risk_synthetic/`, que sao datasets raw gerados localmente;
- `processed/voice_risk/`;
- audios processados em `processed/audio/`;
- transcricoes, metadados, labels, manifestos e splits em `processed/`;
- embeddings em `processed/embeddings/`;
- modelos e metricas em `processed/models/`;
- Parquets finais e `schema.json` em `unified/`.

Ele preserva os datasets reais baixados ou colocados manualmente em `raw/womanhealthfiap/`, `raw/eatd/` e `raw/emu/`, alem dos arquivos `.gitkeep`. Para preservar tambem os datasets raw gerados de voice risk, use:

```bash
python src/clean_generated_data.py --yes --keep-voice-risk-raw
```

### 1. Baixar Datasets

Primeiro o projeto tenta baixar ou localizar os datasets configurados:

```bash
python src/download_datasets.py
```

O downloader verifica `raw/<dataset>` antes de baixar. Quando o dataset ja existe e contem os arquivos esperados, ele nao baixa novamente.

Comportamento por dataset:

- `womanhealthfiap`: baixado via Hugging Face com `snapshot_download`.
- `emu`: baixado como arquivo `.zip` publico do GitHub e extraido em `raw/emu/`.
- `eatd`: registrado como fonte manual, porque o loader publico exige o `EATD-Corpus` local em `raw/eatd/EATD-Corpus/`.
- `voice_risk_training` e `voice_risk_synthetic`: nao sao baixados da internet; sao gerados pelos scripts locais da etapa seguinte.

Tambem e possivel baixar apenas um dataset:

```bash
python src/download_datasets.py --dataset womanhealthfiap
```

### 2. Gerar Datasets Derivados

Depois do download, o projeto gera os datasets locais de voice risk. Eles entram no Unified como qualquer outro dataset porque tambem sao materializados em `raw/`.

```bash
python src/build_voice_risk_dataset.py
python src/generate_voice_risk_synthetic_data.py
```

Esses comandos geram:

- `processed/voice_risk/training_dataset.csv`
- `processed/voice_risk/synthetic_dataset.csv`
- `raw/voice_risk_training/metadata.csv`
- `raw/voice_risk_training/audio/`
- `raw/voice_risk_synthetic/metadata.csv`
- `raw/voice_risk_synthetic/audio/`

O dataset sintetico e util para desenvolvimento e validacao do fluxo, mas nao substitui dados reais em avaliacao clinica ou producao.

### 3. Processar e Unificar

Com os dados brutos e derivados disponiveis, execute o pipeline de processamento:

```bash
python src/validate_raw_files.py
python src/prepare_audio.py
python src/build_metadata.py
python src/extract_audio_embeddings.py
python src/extract_text_embeddings.py
python src/create_splits.py
python src/export_unified_dataset.py
```

Essa etapa faz a padronizacao de audio, metadados, labels, embeddings e splits. Ao final, o dataset unificado fica em:

- `unified/train.parquet`
- `unified/validation.parquet`
- `unified/test.parquet`
- `unified/schema.json`

Tambem e possivel processar um dataset por vez:

```bash
python src/validate_raw_files.py --dataset womanhealthfiap
python src/prepare_audio.py --dataset womanhealthfiap
python src/build_metadata.py --dataset womanhealthfiap
```

### 4. Treinar Modelos

Apos a unificacao, treine os modelos conforme o objetivo do experimento:

```bash
python src/train_baseline.py
python src/train_multimodal_transformer.py
python src/train_voice_risk.py
```

`train_baseline.py` e `train_multimodal_transformer.py` usam os Parquets do Unified e os embeddings extraidos. `train_voice_risk.py` usa o CSV supervisionado `processed/voice_risk/training_dataset.csv`, gerado na etapa de datasets derivados.

### 5. Usar o Modelo Treinado

Depois do treino, use os artefatos salvos em `processed/models/`. A secao [Usar o Modelo em Outro Projeto](#usar-o-modelo-em-outro-projeto) mostra um exemplo de inferencia carregando o checkpoint do transformer multimodal em outro codigo Python.

## O Que Cada Etapa Faz

`download_datasets.py` usa `DatasetDownloader` para verificar `raw/<dataset>`, baixar fontes diretas, arquivos do GitHub ou datasets do Hugging Face quando possivel, validar `sha256`/`md5` opcionais e extrair pacotes `.zip`/`.tar`.

`clean_generated_data.py` remove artefatos locais gerados pelo pipeline, preservando datasets reais baixados/manualizados em `raw/` e arquivos `.gitkeep`.

`build_voice_risk_dataset.py` gera o CSV supervisionado de voice risk a partir das fontes locais aprovadas e materializa `raw/voice_risk_training/` para que ele entre no Unified.

`generate_voice_risk_synthetic_data.py` gera dados sinteticos de desenvolvimento para voice risk e materializa `raw/voice_risk_synthetic/`.

`validate_raw_files.py` gera `processed/metadata/raw_validation_report.json` com contagens e erros basicos.

`prepare_audio.py` converte audios brutos para WAV mono 16 kHz PCM e salva `processed/metadata/audio_manifest.parquet`.

`build_metadata.py` normaliza metadados, transcricoes e labels, salvando `processed/metadata/metadata.parquet`, `processed/labels/labels.parquet` e transcricoes limpas em `processed/transcripts/`.
No EATD-Corpus, `label.txt`/`new_label.txt` sao propagados para os audios do mesmo participante e `depression_label` usa o limiar configurado em `datasets.eatd.label_rules.depression_score_threshold`.

`extract_audio_embeddings.py` usa `microsoft/wavlm-large` e salva vetores `.npy` em `processed/embeddings/audio/`.
Audios invalidos ou vazios sao ignorados por padrao e registrados em `processed/metadata/audio_embedding_errors.parquet`; use `--fail-fast` para interromper no primeiro erro.

`extract_text_embeddings.py` usa `sentence-transformers/all-mpnet-base-v2` e salva vetores `.npy` em `processed/embeddings/text/`.

`create_splits.py` cria splits por `participant_id`, evitando vazamento entre treino, validacao e teste.

`export_unified_dataset.py` exporta os Parquets finais em `unified/` e atualiza `unified/schema.json`.

`train_baseline.py` treina um baseline classico com scikit-learn usando embeddings extraidos.

`train_multimodal_transformer.py` treina um transformer multimodal de fusao sobre tokens de audio, texto e metadados derivados dos embeddings extraidos.

`train_voice_risk.py` treina e exporta os modelos `binary_risk_model.joblib` e `multilabel_risk_model.joblib` consumidos pela API FastAPI.

## Treinamento dos Modelos

Depois de exportar o dataset unificado, o projeto pode treinar tres familias de modelos:

- baseline classico com scikit-learn para `depression_label`;
- transformer multimodal com PyTorch para `depression_label`, `anxiety_label` ou `voice_risk_label`;
- modelos especificos de voice risk usados pela API externa.

### Baseline Classico

Treina uma regressao logistica sobre embeddings ja extraidos:

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

### Multimodal Transformer

Depois do baseline classico, treine o transformer multimodal sobre os mesmos embeddings exportados:

```bash
python src/train_multimodal_transformer.py
```

O treino multimodal usa os embeddings congelados como tokens de entrada:

- token `[CLS]` treinavel;
- token de audio projetado a partir do embedding WavLM;
- token de texto projetado a partir do embedding MPNet;
- token opcional de metadados seguros, como `age` e `duration_seconds`;
- `TransformerEncoder` para fusao;
- cabeca binaria para `depression_label`.

Exemplos:

```bash
python src/train_multimodal_transformer.py --modalities audio,text
python src/train_multimodal_transformer.py --modalities audio,text,metadata
python src/train_multimodal_transformer.py --target anxiety_label --modalities audio,text
python src/train_multimodal_transformer.py --target voice_risk_label --modalities audio,text
python src/train_multimodal_transformer.py --epochs 50 --batch-size 16 --d-model 384 --num-layers 3
```

Artefatos:

- `processed/models/multimodal_transformer/model.pt`
- `processed/models/multimodal_transformer/metrics.json`
- `processed/models/multimodal_transformer/training_config.json`

Esse modelo treina a fusao multimodal. WavLM e MPNet continuam congelados como extratores de embeddings; para fine-tuning end-to-end, seria necessario adicionar um treino que carregue audio/texto bruto diretamente.

### Modelos Voice Risk Para A API

Este repositorio tambem concentra o treinamento dos modelos consumidos pela API `/Users/fernando/development/repo/fiap-tech-challenge-phase-4`.

Para gerar o dataset supervisionado a partir das fontes locais aprovadas:

```bash
python src/build_voice_risk_dataset.py
```

Saidas:

- `processed/voice_risk/training_dataset.csv`
- `processed/voice_risk/training_dataset_build_report.md`
- `raw/voice_risk_training/metadata.csv`
- `raw/voice_risk_training/audio/`

Para gerar dados sinteticos apenas em desenvolvimento:

```bash
python src/generate_voice_risk_synthetic_data.py
```

Saidas sinteticas:

- `processed/voice_risk/synthetic_dataset.csv`
- `raw/voice_risk_synthetic/metadata.csv`
- `raw/voice_risk_synthetic/audio/`

Para treinar os modelos binario e multilabel:

```bash
python src/train_voice_risk.py
```

Como os artefatos `joblib` dependem da versao do `scikit-learn`, gere os modelos voice-risk com uma versao compativel com a API consumidora. No ambiente local atual, a geracao compativel foi validada usando o Python da API:

```bash
PYTHONPATH=src /caminho/do/seu/projeto/.venv/bin/python \
  src/train_voice_risk.py \
  --dataset processed/voice_risk/training_dataset.csv \
  --output-dir processed/models/voice_risk
```

Artefatos:

- `processed/models/voice_risk/binary_risk_model.joblib`
- `processed/models/voice_risk/multilabel_risk_model.joblib`
- `processed/models/voice_risk/voice_risk_metrics.json`

O CSV de treino deve conter:

```text
audio_path
transcription
binary_risk
anxiety
postpartum_depression
hormonal_fatigue
domestic_violence
```

Por padrao, o treino exige arquivos de audio existentes. Para experimentos sem audio, use `--allow-missing-audio`; essa opcao nao deve ser usada para modelos de producao.

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

## Copiar Modelos Para A API

A copia dos artefatos treinados para a API e manual nesta etapa:

```bash
cp processed/models/voice_risk/binary_risk_model.joblib \
  /caminho/do/seu/projeto/artifacts/models/binary_risk_model.joblib

cp processed/models/voice_risk/multilabel_risk_model.joblib \
  /caminho/do/seu/projeto/artifacts/models/multilabel_risk_model.joblib

cp processed/models/multimodal_transformer/model.pt \
  /caminho/do/seu/projeto/artifacts/models/mental_health/multimodal_transformer/model.pt

cp processed/models/multimodal_transformer/training_config.json \
  /caminho/do/seu/projeto/artifacts/models/mental_health/multimodal_transformer/training_config.json

cp processed/models/multimodal_transformer/metrics.json \
  /caminho/do/seu/projeto/artifacts/models/mental_health/multimodal_transformer/metrics.json
```

## Boas Praticas de Dados

- Nao coloque dados brutos, embeddings ou Parquets finais em controle de versao.
- Mantenha `participant_id` consistente, pois ele controla o split sem vazamento.
- Revise `processed/metadata/metadata.parquet` antes de treinar modelos.
- Para datasets sensiveis, armazene dados apenas em ambientes autorizados e criptografados.
