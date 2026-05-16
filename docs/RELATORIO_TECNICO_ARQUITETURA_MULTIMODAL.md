# Relatorio Tecnico Corporativo: Arquitetura Multimodal de Saude da Mulher

Data da analise: 2026-05-16  
Escopo analisado:

- `https://github.com/fernando-squads/fiap-tech-challenge-phase-4-model`
- `https://github.com/fernando-squads/fiap-tech-challenge-phase-4-api`
- `https://github.com/fernando-squads/fiap-tech-challenge-phase-4-web`

Este documento foi produzido a partir da leitura direta do codigo-fonte, configuracoes, READMEs, testes e artefatos locais encontrados nos tres repositorios. Quando uma capacidade e mencionada como nao implementada, isso significa que nao foi localizada no codigo analisado, mesmo que seja desejavel dentro do contexto do Tech Challenge.

# Resumo Executivo

O sistema esta organizado em tres repositorios com responsabilidades bem separadas:

| Repositorio | Responsabilidade principal | Papel na arquitetura |
| --- | --- | --- |
| `fiap-tech-challenge-phase-4-model` | Pipeline de dados, treinamento, geracao de embeddings, modelos e artefatos | Camada offline de MLOps e treinamento |
| `fiap-tech-challenge-phase-4-api` | API FastAPI para inferencia em audio enviado pelo usuario | Camada backend de inferencia e integracao cloud |
| `fiap-tech-challenge-phase-4-web` | Interface React para upload, visualizacao e explicabilidade | Camada frontend de operacao e demonstracao |

A solucao implementa monitoramento e triagem vocal para saude da mulher com foco em:

- sinais gerais de risco vocal;
- ansiedade;
- depressao pos-parto;
- fadiga hormonal;
- violencia domestica;
- predicao multimodal de saude mental para `depression_label`;
- deteccao nao supervisionada de anomalias multimodais;
- explicabilidade baseada em termos textuais, indicadores acusticos e scores de modelos.

O pipeline multimodal real e composto por duas trilhas:

1. Trilha offline de treinamento: baixa datasets, normaliza audio/texto/metadados, gera embeddings com WavLM e MPNet, cria o dataset Unified, treina modelos scikit-learn, transformer multimodal e detector de anomalias.
2. Trilha online de inferencia: recebe audio no frontend, envia para a API, transcreve com Whisper local, extrai features acusticas, calcula sinais NLP, executa modelos joblib, transformer multimodal e detector de anomalias, gera alertas e retorna resultado para exibicao.

Componentes implementados de forma concreta:

- upload multipart de arquivos `.wav`, `.mp3` e `.m4a`;
- validacao de tamanho, tipo e duracao;
- armazenamento temporario em S3 ou disco local;
- exclusao do objeto S3 ao final da requisicao;
- transcricao com `faster-whisper`;
- extracao de MFCC, pitch, energia, pausas, silencio, taxa de fala, variacao prosodica e hesitacao;
- NLP heuristico para termos de medo, tristeza, exaustao, ansiedade, isolamento, coercao, violencia, inseguranca, pos-parto e sinais hormonais;
- modelos scikit-learn binario e multilabel;
- transformer multimodal PyTorch sobre embeddings congelados de audio/texto;
- detector nao supervisionado `IsolationForest` para anomalias multimodais;
- frontend com dashboards, cards de risco, card de anomalia, graficos acusticos e paineis de explicabilidade;
- integracao efetiva com S3 para storage temporario;
- Dockerfile e orientacoes para ECS/Fargate;
- infraestrutura CloudFront, IaC de ECS e monitoramento CloudWatch.

# Visao Arquitetural

Diagrama textual da arquitetura implementada:

```text
Usuario
  |
  v
Frontend React/Vite
  - upload de audio
  - modo API real ou mock
  - dashboard de scores, alertas e features
  |
  | POST /audio/analyze multipart/form-data
  v
Backend FastAPI
  - validacao do upload
  - persistencia temporaria local
  - upload para S3
  - Whisper local
  - features acusticas
  - NLP heuristico
  - modelos joblib
  - transformer multimodal
  - detector de anomalias
  - geracao de alertas
  |
  | le artefatos treinados
  v
Artifacts da API
  - binary_risk_model.joblib
  - multilabel_risk_model.joblib
  - mental_health/multimodal_transformer/model.pt
  - training_config.json
  - metrics.json
  - anomaly_detector/anomaly_detector.joblib
  ^
  |
Projeto de Modelos
  - download e validacao de datasets
  - processamento de audio
  - normalizacao de metadados
  - embeddings WavLM e MPNet
  - dataset Unified
  - treinamento scikit-learn, PyTorch e anomalia nao supervisionada
```

Diagrama textual da comunicacao entre repositorios:

```text
fiap-tech-challenge-phase-4-model
  gera:
    processed/models/voice_risk/binary_risk_model.joblib
    processed/models/voice_risk/multilabel_risk_model.joblib
    processed/models/multimodal_transformer/model.pt
    processed/models/multimodal_transformer/training_config.json
    processed/models/multimodal_transformer/metrics.json
    processed/models/anomaly_detector/anomaly_detector.joblib
  copia manualmente para:
    fiap-tech-challenge-phase-4/artifacts/models/

fiap-tech-challenge-phase-4-api
  expoe:
    GET /health
    POST /audio/analyze
  e consumido por:
    fiap-tech-challenge-phase-4-web

fiap-tech-challenge-phase-4-web
  configura:
    VITE_API_BASE_URL=/api ou http://localhost:8000
    VITE_API_PROXY_TARGET=http://localhost:8000
  envia:
    audio multipart/form-data
  renderiza:
    transcricao, scores, anomalia, alertas, evidencias e graficos
```

# Introducao

O Tech Challenge Fase 4 foi implementado como uma solucao de triagem e apoio a decisao para saude da mulher, usando sinais multimodais de voz e texto. O sistema nao executa diagnostico medico. Ele produz indicadores quantitativos e alertas para priorizacao de revisao humana por profissional qualificado.

A arquitetura encontrada separa treinamento e inferencia. O repositorio de modelos concentra dados, preprocessamento, embeddings, treinamento e exportacao de artefatos. O backend FastAPI concentra inferencia operacional e integracao com storage. O frontend React concentra experiencia de uso, visualizacao e explicabilidade.

Essa separacao e coerente com uma arquitetura corporativa de IA aplicada:

- a camada de treinamento pode ser executada de forma controlada, reprodutivel e offline;
- a camada de inferencia fica enxuta e consome somente artefatos versionaveis;
- a camada web nao conhece detalhes internos dos modelos, apenas o contrato da API.

# Objetivos do Projeto

Objetivos funcionais implementados:

- receber gravacoes de voz em formato `.wav`, `.mp3` ou `.m4a`;
- validar tipo, tamanho e duracao do audio;
- transcrever audio com Whisper local;
- extrair atributos acusticos relevantes para comportamento vocal;
- detectar termos textuais associados a sofrimento, violencia e risco psicologico;
- calcular risco geral binario;
- calcular scores multilabel para ansiedade, depressao pos-parto, fadiga hormonal e violencia domestica;
- calcular predicao multimodal de saude mental com transformer treinado externamente;
- gerar alertas com severidade e justificativas;
- exibir resultados no frontend com cards, graficos e evidencias.

Objetivos de MLOps implementados:

- baixar ou localizar datasets brutos configurados em YAML;
- validar arquivos raw;
- converter audio para WAV mono 16 kHz PCM;
- normalizar metadados e labels;
- gerar dataset Unified em Parquet;
- gerar embeddings WavLM e MPNet;
- criar splits por `participant_id`;
- treinar baseline scikit-learn;
- treinar transformer multimodal;
- treinar modelos voice risk para consumo pela API;
- limpar artefatos gerados para reexecucao do pipeline.

# Arquitetura Geral

## Responsabilidades por camada

| Camada | Repositorio | Responsabilidade |
| --- | --- | --- |
| Dados e treinamento | `fiap-tech-challenge-phase-4-model` | Construcao do Unified, embeddings, treinamento e exportacao de modelos |
| Inferencia e API | `fiap-tech-challenge-phase-4-api` | Processar uploads, extrair sinais, executar modelos, gerar alertas |
| Interface | `fiap-tech-challenge-phase-4-web` | Upload de audio, consumo da API, visualizacao de resultados |

## Fluxo macro implementado

```text
1. Preparacao offline
   download_datasets.py
   build_voice_risk_dataset.py
   generate_voice_risk_synthetic_data.py
   validate_raw_files.py
   prepare_audio.py
   build_metadata.py
   extract_audio_embeddings.py
   extract_text_embeddings.py
   create_splits.py
   export_unified_dataset.py

2. Treinamento offline
   train_baseline.py
   train_multimodal_transformer.py
   train_anomaly_detector.py
   train_voice_risk.py

3. Publicacao manual dos artefatos
   cp processed/models/... -> backend/artifacts/models/...

4. Inferencia online
   frontend -> POST /audio/analyze -> backend -> modelos -> frontend
```

## Decisoes arquiteturais observadas

- O backend nao treina modelos. Ele falha com `503` quando artefatos obrigatorios nao existem ou sao incompativeis, exceto se fallback heuristico for explicitamente permitido.
- O dataset Unified usa Parquet e schema universal, com splits por `participant_id` para reduzir data leakage.
- O transformer multimodal usa embeddings pre-extraidos e congelados. Nao ha fine-tuning end-to-end de WavLM, MPNet ou Whisper.
- A deteccao dedicada de anomalias usa `IsolationForest` nao supervisionado sobre embeddings WavLM/MPNet e metadados opcionais, com score calibrado no conjunto de treino.
- O audio de inferencia e temporario. Mesmo quando enviado ao S3, o objeto e excluido ao final da requisicao.
- O frontend tem modo mock configuravel por `VITE_USE_MOCK=true`, util para demonstracao sem backend.

# Arquitetura Multimodal

A arquitetura multimodal possui duas formas de fusao:

1. Fusao tabular para modelos scikit-learn de voice risk:
   - features acusticas extraidas com `librosa`;
   - features textuais heuristicas extraidas por `TextRiskAnalyzer`;
   - `DictVectorizer`;
   - `StandardScaler`;
   - `LogisticRegression` ou `OneVsRestClassifier(LogisticRegression)`.

2. Fusao neural com transformer:
   - embedding de audio WavLM, dimensao observada 1024;
   - embedding de texto MPNet, dimensao observada 768;
   - metadados opcionais, como idade e duracao;
   - token `[CLS]` treinavel;
   - projecoes lineares por modalidade;
   - embeddings de modalidade;
   - `TransformerEncoder`;
   - cabeca binaria com `BCEWithLogitsLoss`.

Fluxo multimodal offline:

```text
audio bruto
  -> WAV mono 16 kHz PCM
  -> WavLM large
  -> audio_embedding.npy

texto/transcricao
  -> normalizacao
  -> all-mpnet-base-v2
  -> text_embedding.npy

metadados
  -> schema Unified
  -> age, duration_seconds

Unified + embeddings
  -> MultimodalEmbeddingDataset
  -> MultimodalTransformerClassifier
  -> model.pt + metrics.json + training_config.json
```

Fluxo multimodal online:

```text
audio enviado pelo usuario
  -> Whisper local
  -> transcricao
  -> embedding WavLM do audio
  -> embedding MPNet da transcricao
  -> metadata de duracao
  -> transformer multimodal carregado de artifacts/models
  -> mental_health_prediction
  -> alerta se probabilidade >= threshold
```

# Arquitetura dos Repositorios

## 1. `fiap-tech-challenge-phase-4-model`

Objetivo principal: preparar datasets, construir o Unified, extrair embeddings e treinar modelos.

Papel dentro da arquitetura geral: camada offline de dados e modelos. E o ponto de origem dos artefatos consumidos pelo backend.

Estrutura relevante:

```text
config/datasets.yaml
raw/
processed/
unified/
src/
tests/
README.md
```

Principais modulos:

| Modulo | Responsabilidade |
| --- | --- |
| `dataset_downloader.py` | Download ou orientacao manual dos datasets configurados |
| `prepare_audio.py` | Conversao para WAV mono 16 kHz PCM |
| `dataset_adapters.py` | Adaptacao generica e especifica do EATD para schema Unified |
| `build_metadata.py` | Normalizacao de metadados, transcripts e labels |
| `extract_audio_embeddings.py` | Embeddings WavLM |
| `extract_text_embeddings.py` | Embeddings MPNet |
| `create_splits.py` | Splits por participante |
| `export_unified_dataset.py` | Exportacao Parquet e schema |
| `voice_risk.py` | Extracao acustica/textual e features para modelos voice risk |
| `train_voice_risk.py` | Treinamento dos modelos joblib consumidos pela API |
| `multimodal_transformer.py` | Arquitetura PyTorch do transformer multimodal |
| `train_multimodal_transformer.py` | Treinamento do transformer multimodal |
| `train_anomaly_detector.py` | Treinamento do detector nao supervisionado de anomalias |
| `clean_generated_data.py` | Limpeza segura de artefatos gerados |

Tecnologias e dependencias principais:

- Python;
- pandas, pyarrow, numpy, scipy;
- scikit-learn, joblib;
- PyTorch, torchaudio, transformers;
- sentence-transformers;
- librosa, soundfile;
- huggingface-hub, datasets;
- tqdm, pyyaml, openpyxl;
- pytest.

Datasets configurados:

| Dataset | Situacao no codigo |
| --- | --- |
| `WomanHealthFIAP` | Hugging Face `brunoretiro/womanhealthfiap` |
| `EATD-Corpus` | exige corpus local em `raw/eatd/EATD-Corpus` |
| `EMU` | GitHub archive `mltlachac/EMU` |
| `VoiceRiskTraining` | gerado localmente por `build_voice_risk_dataset.py` |
| `VoiceRiskSynthetic` | gerado localmente por `generate_voice_risk_synthetic_data.py` |

Persistencia:

- `raw/`: dados brutos baixados, manuais ou gerados;
- `processed/audio/`: WAV padronizado;
- `processed/metadata/`: manifestos, metadata, relatorios, splits, manifests de embeddings;
- `processed/embeddings/`: vetores `.npy`;
- `processed/models/`: modelos e metricas;
- `unified/`: `train.parquet`, `validation.parquet`, `test.parquet`, `schema.json`.

## 2. `fiap-tech-challenge-phase-4-api`

Objetivo principal: expor uma API de inferencia para analise de gravacoes de voz e transcricoes de pacientes em consultas medicas.

Papel dentro da arquitetura geral: camada operacional de inferencia, storage temporario, integracao S3 e geracao de alertas.

Estrutura relevante:

```text
app/api/
app/alerts/
app/core/
app/features/
app/models/
app/nlp/
app/schemas/
app/services/
artifacts/models/
Dockerfile
requirements.txt
tests/
```

Principais modulos:

| Modulo | Responsabilidade |
| --- | --- |
| `app/main.py` | Instancia FastAPI, CORS, Swagger e lifecycle |
| `app/api/routes.py` | Endpoints `/health` e `/audio/analyze` |
| `app/services/storage.py` | Validacao, S3/local storage, cleanup |
| `app/services/transcription.py` | Whisper local via `faster-whisper` |
| `app/features/acoustic.py` | Features acusticas |
| `app/nlp/rules.py` | NLP heuristico |
| `app/models/binary.py` | Inferencia binaria |
| `app/models/multilabel.py` | Inferencia multilabel |
| `app/services/mental_health.py` | Inferencia com transformer multimodal |
| `app/alerts/rules.py` | Alertas e severidade |
| `app/core/logging.py` | Logs estruturados JSON |

Tecnologias e dependencias principais:

- FastAPI, uvicorn, python-multipart;
- pydantic e pydantic-settings;
- boto3 para S3;
- faster-whisper;
- librosa, soundfile, scipy, numpy;
- scikit-learn, joblib;
- PyTorch, transformers, sentence-transformers;
- pytest, httpx.

APIs disponiveis:

| Metodo | Endpoint | Descricao |
| --- | --- | --- |
| `GET` | `/` | redireciona para `/docs` |
| `GET` | `/health` | retorna `OK` em `text/plain` |
| `POST` | `/audio/analyze` | recebe audio multipart e retorna analise multimodal |

Persistencia:

- audio local temporario em `/tmp/voice-risk-screening-api`;
- upload temporario em S3 quando `STORAGE_BACKEND=s3`;
- remocao do objeto S3 ao final do processamento;
- artefatos de modelo em `artifacts/models/`;
- cache local de Whisper em `artifacts/whisper/`.

## 3. `fiap-tech-challenge-phase-4-web`

Objetivo principal: oferecer interface de upload e visualizacao dos resultados da API.

Papel dentro da arquitetura geral: camada de apresentacao e experiencia do usuario.

Estrutura relevante:

```text
src/pages/Dashboard.tsx
src/components/
src/services/api.ts
src/types/analysis.ts
src/utils/
src/styles/global.css
vite.config.ts
package.json
```

Tecnologias e dependencias principais:

- React 18;
- Vite;
- TypeScript;
- lucide-react;
- Recharts;
- CSS global.

Funcionalidades implementadas:

- upload por selecao ou drag-and-drop;
- validacao local de extensao `.wav`, `.mp3`, `.m4a`;
- envio multipart para `POST /audio/analyze`;
- modo mock por `VITE_USE_MOCK=true`;
- exibicao de transcricao, metadados, score binario, scores multilabel, alertas, evidencias e graficos acusticos;
- proxy local do Vite para contornar CORS em desenvolvimento.

# Fluxo de Dados

## Fluxo offline de dados e treinamento

```text
1. Download/localizacao
   config/datasets.yaml
   -> download_datasets.py
   -> raw/<dataset>

2. Geracao local de datasets derivados
   build_voice_risk_dataset.py
   generate_voice_risk_synthetic_data.py
   -> processed/voice_risk/*.csv
   -> raw/voice_risk_training
   -> raw/voice_risk_synthetic

3. Validacao e audio
   validate_raw_files.py
   prepare_audio.py
   -> processed/audio
   -> processed/metadata/audio_manifest.parquet

4. Metadata e labels
   build_metadata.py
   -> processed/metadata/metadata.parquet
   -> processed/labels/labels.parquet
   -> processed/transcripts

5. Embeddings
   extract_audio_embeddings.py
   -> processed/embeddings/audio/*.npy
   extract_text_embeddings.py
   -> processed/embeddings/text/*.npy

6. Splits e exportacao
   create_splits.py
   -> processed/metadata/splits.parquet
   export_unified_dataset.py
   -> unified/train.parquet
   -> unified/validation.parquet
   -> unified/test.parquet
   -> unified/schema.json

7. Treinamento
   train_baseline.py
   train_multimodal_transformer.py
   train_anomaly_detector.py
   train_voice_risk.py
   -> processed/models
```

## Fluxo online de inferencia

```text
Frontend
  -> usuario seleciona audio
  -> analyzeAudio(file)
  -> POST /audio/analyze

Backend
  -> valida upload
  -> grava arquivo local temporario
  -> envia para S3, se configurado
  -> transcreve com Whisper
  -> extrai features acusticas
  -> extrai features textuais
  -> executa modelo binario
  -> executa modelo multilabel
  -> executa transformer multimodal
  -> executa detector de anomalias
  -> gera alertas
  -> remove S3/local temporario
  -> retorna JSON

Frontend
  -> renderiza resultado, evidencias, alertas e graficos
```

# Fluxo de Audio

## No treinamento

Entrada:

- arquivos `.wav`, `.mp3`, `.flac`, `.m4a`, `.mp4`, `.ogg`, `.webm`, `.aac`, conforme `config/datasets.yaml`.

Processamento:

1. `prepare_audio.py` localiza audios por dataset.
2. Cada audio recebe `sample_id`.
3. O audio e convertido para WAV mono 16 kHz PCM.
4. A duracao e extraida e persistida no manifesto.
5. `extract_audio_embeddings.py` carrega o WAV 16 kHz mono.
6. WavLM large gera embeddings por chunks.
7. Os embeddings sao agregados pela media dos chunks e salvos em `.npy`.

Saidas:

- `processed/audio/<dataset>/<sample_id>.wav`;
- `processed/metadata/audio_manifest.parquet`;
- `processed/embeddings/audio/<sample_id>.npy`;
- `processed/metadata/audio_embeddings.parquet`.

## Na inferencia

Entrada:

- `UploadFile` enviado por `multipart/form-data`.

Processamento:

1. Validacao de extensao: `.wav`, `.mp3`, `.m4a`.
2. Validacao de content-type.
3. Validacao de tamanho maximo por `MAX_AUDIO_SIZE_MB`.
4. Validacao de duracao minima e maxima.
5. Escrita local temporaria.
6. Upload para S3 quando `STORAGE_BACKEND=s3`.
7. Transcricao Whisper.
8. Extracao acustica com `librosa`.
9. Embedding WavLM para o transformer multimodal e para o detector de anomalias.
10. Exclusao do objeto S3 e do arquivo local temporario.

Features acusticas extraidas:

- `duration_seconds`;
- `sample_rate`;
- media e desvio de 13 MFCCs;
- `energy_mean`, `energy_std`, `energy_iqr`, `energy_max`;
- `pitch_mean_hz`, `pitch_std_hz`, `pitch_min_hz`, `pitch_max_hz`;
- `pause_count`;
- `mean_pause_duration_seconds`;
- `max_pause_duration_seconds`;
- `silence_duration_seconds`;
- `silence_ratio`;
- `speech_rate_wpm`;
- `prosodic_variation`;
- `hesitation_score`.

# Fluxo Textual

## No treinamento

Entradas:

- transcricoes existentes em datasets;
- textos em arquivos `.txt`, `.cha`, `.trs`;
- colunas tabulares como `transcript`, `transcription`, `text`, `utterance`, `answer`;
- textos sinteticos/rotulados dos datasets de voice risk.

Processamento:

1. `dataset_adapters.py` extrai transcricoes diretas ou por tabela.
2. `build_metadata.py` normaliza e salva transcricoes em `processed/transcripts`.
3. `extract_text_embeddings.py` usa `sentence-transformers/all-mpnet-base-v2`.
4. Embeddings sao salvos em `.npy`.

Saidas:

- `processed/transcripts/<dataset>/<sample_id>.txt`;
- `processed/embeddings/text/<sample_id>.npy`;
- `processed/metadata/text_embeddings.parquet`.

## Na inferencia

Entradas:

- transcricao gerada pelo Whisper local.

Processamento:

1. `TextRiskAnalyzer` normaliza texto removendo acentos e padronizando espacos.
2. Procura termos por categoria:
   - medo;
   - tristeza;
   - exaustao;
   - ansiedade;
   - isolamento;
   - coercao;
   - violencia;
   - inseguranca;
   - pos-parto;
   - hormonal.
3. Calcula score textual por pesos e diversidade de categorias.
4. Gera evidencias com termo, categoria e snippet.
5. Para o transformer e para o detector de anomalias, gera embedding MPNet da transcricao.

Saidas:

- `nlp_analysis.score`;
- `nlp_analysis.categories`;
- `nlp_analysis.category_scores`;
- `nlp_analysis.evidences`;
- embedding textual para o transformer.

# Pipeline de Inferencia

Endpoint principal: `POST /audio/analyze`.

Contrato de entrada:

```text
multipart/form-data
  file: audio .wav, .mp3 ou .m4a
  run_multilabel: boolean, default true
```

Contrato de saida:

```text
AnalyzeResponse
  file_id
  filename
  content_type
  duration_seconds
  transcription
  acoustic_features
  nlp_analysis
  binary_prediction
  multilabel_prediction
  mental_health_prediction
  anomaly_prediction
  alerts
```

Sequencia operacional:

```text
POST /audio/analyze
  -> AudioStorageService.save_upload
       valida tipo, tamanho e duracao
       salva temporario local
       opcionalmente envia ao S3
  -> WhisperLocalSpeechToTextProvider.transcribe
       carrega faster-whisper
       retorna texto, idioma, confianca e duracao
  -> AcousticFeatureExtractor.extract
       calcula features acusticas
  -> TextRiskAnalyzer.analyze
       calcula score textual e evidencias
  -> BinaryRiskModelService.predict
       carrega joblib e retorna risco/nao risco
  -> MultilabelRiskModelService.predict
       carrega joblib e retorna scores por categoria
  -> MentalHealthModelService.predict
       gera WavLM e MPNet online
       carrega model.pt
       retorna probabilidade binaria
  -> AnomalyModelService.predict
       gera WavLM e MPNet online
       carrega anomaly_detector.joblib
       retorna score nao supervisionado
  -> AlertGenerator.generate
       aplica thresholds e justifica alertas
  -> cleanup_processed_audio
       remove S3 e arquivo local
```

Tratamento de erros:

| Erro | Status HTTP |
| --- | --- |
| Upload invalido | `400` |
| Falha de storage | `503` |
| Falha de transcricao | `503` |
| Falha de features acusticas | `422` |
| Modelo ausente/incompativel | `503` |
| Falha de cleanup apos resposta montada | `503` |

# Integracao com AWS

## S3

S3 e a integracao AWS efetivamente implementada no codigo.

Variaveis relevantes:

| Variavel | Uso |
| --- | --- |
| `STORAGE_BACKEND` | `s3` ou `local` |
| `S3_BUCKET_NAME` | bucket privado para upload temporario |
| `S3_PREFIX` | prefixo de objetos |
| `AWS_REGION` | regiao do cliente S3 |
| `AWS_ENDPOINT_URL` | endpoint alternativo para LocalStack/MinIO |
| `AWS_ACCESS_KEY_ID` | credencial obrigatoria em modo S3 |
| `AWS_SECRET_ACCESS_KEY` | credencial obrigatoria em modo S3 |
| `AWS_SESSION_TOKEN` | token opcional |
| `S3_SERVER_SIDE_ENCRYPTION` | criptografia server-side, default `AES256` |

Fluxo S3 implementado:

```text
audio local temporario
  -> boto3.upload_file
  -> s3://<bucket>/<prefix>/<yyyy>/<mm>/<dd>/<file_id>.<ext>
  -> processamento local
  -> boto3.delete_object
```

O backend exige `s3:PutObject` e `s3:DeleteObject` no prefixo configurado. O codigo nao implementa leitura posterior do objeto S3, porque o processamento usa o arquivo local temporario.

## ECS/Fargate

Existe Dockerfile para o backend:

```text
python:3.12-slim
instala libsndfile1, ffmpeg, libgomp1
instala requirements
executa uvicorn em 0.0.0.0:8090
```

O README descreve uso em ECS/Fargate com storage efemero em `/tmp/voice-risk-screening-api`.

## CloudFront

Nao foi localizada configuracao de CloudFront, distribuicao, invalidacao, bucket website, origem ou IaC no codigo analisado.

O frontend gera build estatico com Vite em `dist/`, que tecnicamente poderia ser hospedado atras de CloudFront, mas essa integracao nao esta implementada no repositorio.

# Modelos de IA Utilizados

## Whisper local

Repositorio: `fiap-tech-challenge-phase-4-api`.

Implementacao:

- `app/services/transcription.py`;
- provider `WhisperLocalSpeechToTextProvider`;
- biblioteca `faster-whisper`;
- modelo configuravel por `WHISPER_MODEL_NAME`, default `base`;
- device configuravel por `WHISPER_DEVICE`, default `cpu`;
- compute type configuravel por `WHISPER_COMPUTE_TYPE`, default `int8`;
- VAD configuravel por `WHISPER_VAD_FILTER`.

Entrada:

- caminho de arquivo de audio local.

Saida:

- texto transcrito;
- idioma;
- duracao;
- confianca aproximada por `avg_logprob` dos segmentos;
- provider.

Finalidade:

- gerar texto a partir do audio para NLP heuristico e embedding MPNet.

## WavLM large

Repositorios:

- treinamento: `fiap-tech-challenge-phase-4-model`;
- inferencia: `fiap-tech-challenge-phase-4`.

Implementacao offline:

- `extract_audio_embeddings.py`;
- `microsoft/wavlm-large`;
- `AutoFeatureExtractor`;
- `WavLMModel`;
- audio 16 kHz mono;
- chunks por `max_chunk_seconds`;
- pooling por media temporal do `last_hidden_state`;
- media entre chunks.

Implementacao online:

- `app/services/mental_health.py`;
- carrega audio com `librosa.load(sr=16000, mono=True)`;
- aplica mesmo modelo e media temporal;
- vetor ajustado para a dimensao esperada pelo checkpoint.

Entrada:

- waveform 16 kHz mono.

Saida:

- embedding de audio, dimensao observada no artefato atual: 1024.

## Sentence Transformers MPNet

Repositorios:

- treinamento: `fiap-tech-challenge-phase-4-model`;
- inferencia: `fiap-tech-challenge-phase-4-api`.

Modelo:

- `sentence-transformers/all-mpnet-base-v2`.

Entrada:

- transcricao textual.

Saida:

- embedding textual, dimensao observada no artefato atual: 768.

## Transformer multimodal

Repositorios:

- treinamento: `fiap-tech-challenge-phase-4-model`;
- inferencia: `fiap-tech-challenge-phase-4-api`.

Arquitetura:

- `MultimodalTransformerClassifier`;
- projecao linear por modalidade;
- `LayerNorm`;
- `GELU`;
- dropout;
- token `[CLS]` treinavel;
- embeddings de modalidade treinaveis;
- `nn.TransformerEncoder`;
- cabeca binaria linear.

Configuracao observada no artefato atual:

```text
modalities: audio,text
target: depression_label
input_dims:
  audio: 1024
  text: 768
epochs configurados: 30
batch_size: 32
learning_rate: 0.0001
weight_decay: 0.0001
threshold: 0.5
```

Estrategia de treinamento:

- `BCEWithLogitsLoss`;
- `pos_weight` quando ha desbalanceamento;
- otimizador `AdamW`;
- gradient clipping;
- early stopping por ROC-AUC de validacao ou F1 quando ROC-AUC nao esta disponivel;
- metricas: accuracy, F1, ROC-AUC e loss.

## Detector nao supervisionado de anomalias

Repositorios:

- treinamento: `fiap-tech-challenge-phase-4-model`;
- inferencia: `fiap-tech-challenge-phase-4`.

Implementacao de treinamento:

- `train_anomaly_detector.py`;
- `IsolationForest`;
- `StandardScaler`;
- PCA opcional por `--pca-components`;
- modalidades configuraveis por `--modalities audio,text,metadata`;
- labels nao sao usadas como alvo de treinamento;
- `depression_label` pode ser usado apenas como avaliacao auxiliar via `--target-eval-column`.

Entrada offline:

- `unified/train.parquet`;
- `unified/validation.parquet`;
- `unified/test.parquet`;
- embeddings `.npy` de audio e texto;
- metadados seguros opcionais, como `age` e `duration_seconds`.

Saida offline:

- `processed/models/anomaly_detector/anomaly_detector.joblib`;
- `processed/models/anomaly_detector/anomaly_metrics.json`;
- `processed/models/anomaly_detector/anomaly_config.json`.

Formato do artefato:

```text
pipeline scikit-learn
modalities
embedding_dims
metadata_columns
metadata_fill_values
feature_blocks
feature_count
threshold
calibration.raw_min/raw_max
contamination
```

Implementacao online:

- `app/models/anomaly.py`;
- `AnomalyModelService`;
- gera embeddings WavLM/MPNet com a mesma logica do servico multimodal;
- monta vetor na ordem dos `feature_blocks`;
- calcula `raw_score = -decision_function`;
- normaliza score para `0..1` usando calibracao do treino;
- retorna `anomaly_prediction`.

## Modelos scikit-learn de voice risk

Repositorio de treinamento: `fiap-tech-challenge-phase-4-model`.

Repositorio de inferencia: `fiap-tech-challenge-phase-4-api`.

Modelo binario:

- `DictVectorizer`;
- `StandardScaler(with_mean=False)`;
- `LogisticRegression(max_iter=1000, class_weight="balanced")`;
- alvo `binary_risk`;
- saida `risco` ou `nao_risco`.

Modelo multilabel:

- `DictVectorizer`;
- `StandardScaler(with_mean=False)`;
- `OneVsRestClassifier(LogisticRegression(...))`;
- alvos:
  - `anxiety`;
  - `postpartum_depression`;
  - `hormonal_fatigue`;
  - `domestic_violence`.

Features:

- acusticas com prefixo `acoustic_`;
- score textual global;
- contagens por categoria textual;
- scores por categoria textual.

## NLP heuristico

Nao e um modelo estatistico treinado, mas e uma camada de IA simbolica/regras usada tanto no treino de voice risk quanto na inferencia.

Categorias:

- `fear`;
- `sadness`;
- `exhaustion`;
- `anxiety`;
- `isolation`;
- `coercion`;
- `violence`;
- `insecurity`;
- `postpartum`;
- `hormonal`.

Saidas:

- score global normalizado;
- contagem por categoria;
- score por categoria;
- evidencias textuais.

# Estrategias de Deteccao de Anomalias

A estrategia implementada possui duas camadas complementares:

1. sinais supervisionados ou baseados em regra, usados para risco geral, categorias clinicas e justificativas;
2. detector dedicado nao supervisionado, treinado com `IsolationForest` sobre embeddings multimodais e calibrado pelo conjunto de treino do Unified.

O detector nao supervisionado nao aprende uma classe clinica especifica. Ele aprende o perfil estatistico das representacoes de audio/texto/metadados e sinaliza amostras distantes desse padrao. Portanto, `anomaly_prediction.score` deve ser interpretado como sinal de revisao humana, nao como diagnostico.

Anomalias/sinais detectados no codigo:

| Sinal | Fonte | Estrategia |
| --- | --- | --- |
| Risco geral | features acusticas + NLP | `LogisticRegression` binaria e threshold |
| Ansiedade | texto/acustica | multilabel + termos de ansiedade/medo/inseguranca |
| Depressao pos-parto | texto/acustica | multilabel + tristeza, isolamento e pos-parto |
| Fadiga hormonal | texto/acustica | multilabel + exaustao, hormonal, baixa energia |
| Violencia domestica | texto/acustica | multilabel + violencia, coercao, medo, inseguranca |
| Sinal de depressao/saude mental | WavLM + MPNet | transformer multimodal e threshold |
| Anomalia multimodal | WavLM + MPNet + metadados opcionais | `IsolationForest` nao supervisionado e score calibrado |
| Hesitacao | audio | pausas, silencio e pausa longa |
| Baixa energia vocal | audio | `energy_mean <= LOW_ENERGY_THRESHOLD` |
| Silencio elevado | audio | `silence_ratio >= 0.35` |
| Pausas longas | audio | `max_pause_duration_seconds >= LONG_PAUSE_THRESHOLD_SECONDS` |

Geracao de alertas:

```text
binary_probability >= BINARY_RISK_THRESHOLD
  -> alerta binary_risk

multilabel_score >= MULTILABEL_RISK_THRESHOLD
  -> alerta por categoria

mental_health_probability >= MENTAL_HEALTH_THRESHOLD
  -> alerta depression_label ou target configurado

anomaly_score >= ANOMALY_THRESHOLD ou threshold do artefato
  -> alerta anomaly
```

Severidade:

```text
score >= HIGH_SEVERITY_THRESHOLD    -> alto
score >= MEDIUM_SEVERITY_THRESHOLD  -> medio
caso contrario                      -> baixo
```

# Estrategias de Monitoramento

Monitoramento implementado:

- logs estruturados JSON em stdout;
- middleware HTTP com `request_id`, metodo, path, status, duracao e host;
- logs por etapa da inferencia:
  - upload validado;
  - transcricao concluida;
  - features extraidas;
  - NLP concluido;
  - inferencia concluida;
  - alertas gerados;
  - cleanup concluido;
- limpeza de arquivos temporarios antigos no startup do FastAPI.

Dados sensiveis evitados nos logs:

- corpo da requisicao;
- transcricao completa;
- audio;
- dados clinicos livres.

Monitoramento nao implementado no codigo:

- metricas Prometheus;
- tracing distribuido;
- dashboard CloudWatch;
- alarmes versionados;
- logs estruturados com correlacao entre frontend e backend alem de `x-request-id`;
- monitoramento de drift de modelos;
- monitoramento de performance por classe.

# APIs e Servicos

## Backend FastAPI

`GET /health`

```text
Resposta:
  200 OK
  Content-Type: text/plain
  Body: OK
```

`POST /audio/analyze`

```text
Entrada:
  multipart/form-data
    file: UploadFile
    run_multilabel: boolean

Saida:
  AnalyzeResponse
```

Campos principais de resposta:

- `file_id`;
- `filename`;
- `duration_seconds`;
- `transcription`;
- `acoustic_features`;
- `nlp_analysis`;
- `binary_prediction`;
- `multilabel_prediction`;
- `mental_health_prediction`;
- `anomaly_prediction`;
- `alerts`.

Swagger/OpenAPI:

- `/docs`;
- `/redoc`;
- `/openapi.json`.

## Frontend

Servico de API:

- `src/services/api.ts`;
- base URL por `VITE_API_BASE_URL`;
- envio para `/audio/analyze`;
- `run_multilabel=true`;
- fallback mock quando `VITE_USE_MOCK=true`.

## Servicos externos

| Servico | Implementacao |
| --- | --- |
| AWS S3 | `boto3` no backend |
| Hugging Face Hub | download de datasets e modelos |
| faster-whisper | transcricao local |
| Vite dev proxy | proxy local `/api` para backend |

# Frontend

O frontend e uma SPA React/Vite com foco em fluxo de upload e visualizacao de triagem.

Componentes principais:

| Componente | Responsabilidade |
| --- | --- |
| `Dashboard.tsx` | Orquestra upload, estado de loading, erro e resultado |
| `AudioUpload.tsx` | Seleciona/arrasta arquivo e valida extensao local |
| `AnalysisResult.tsx` | Compoe o resultado completo |
| `BinaryRiskCard.tsx` | Exibe risco binario |
| `AnomalyCard.tsx` | Exibe score do detector nao supervisionado |
| `MultilabelScores.tsx` | Exibe categorias multilabel |
| `AlertsPanel.tsx` | Exibe alertas e justificativas |
| `ExplainabilityPanel.tsx` | Exibe evidencias textuais e acusticas |
| `AcousticCharts.tsx` | Exibe graficos Recharts |
| `Disclaimer.tsx` | Exibe aviso etico/legal |

Entrada:

- arquivo `.wav`, `.mp3` ou `.m4a`.

Saida visual:

- resumo do processamento;
- transcricao;
- probabilidade binaria;
- score de anomalia;
- scores por categoria;
- alertas;
- evidencias;
- graficos acusticos.

Limitacoes:

- nao ha autenticacao;
- nao ha controle de sessao;
- nao ha historico de analises;
- nao ha persistencia local de resultados;
- nao ha streaming em tempo real.

# Backend

O backend centraliza inferencia operacional. Ele nao inclui treinamento.

Fluxo do endpoint:

```text
analyze_audio()
  settings = get_settings()
  storage = AudioStorageService(settings)
  saved_audio = storage.save_upload(file)
  transcription = transcriber.transcribe(saved_audio.path)
  acoustic_features = extractor.extract(saved_audio.path, transcription.text)
  nlp_analysis = TextRiskAnalyzer().analyze(transcription.text)
  binary_prediction = BinaryRiskModelService.predict(...)
  multilabel_prediction = MultilabelRiskModelService.predict(...)
  mental_health_prediction = MentalHealthModelService.predict(...)
  anomaly_prediction = AnomalyModelService.predict(...)
  alerts = AlertGenerator.generate(...)
  storage.cleanup_processed_audio(saved_audio)
```

Configuracoes relevantes:

| Grupo | Variaveis |
| --- | --- |
| Aplicacao | `APP_NAME`, `ENVIRONMENT`, `LOG_LEVEL`, `LOG_FORMAT` |
| Storage | `STORAGE_BACKEND`, `TEMP_AUDIO_DIR`, `STORAGE_DIR`, `TEMP_AUDIO_RETENTION_MINUTES` |
| S3 | `S3_BUCKET_NAME`, `S3_PREFIX`, `AWS_REGION`, `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `S3_SERVER_SIDE_ENCRYPTION` |
| Audio | `MAX_AUDIO_SIZE_MB`, `MIN_AUDIO_DURATION_SECONDS`, `MAX_AUDIO_DURATION_SECONDS`, `AUDIO_SAMPLE_RATE`, `SILENCE_TOP_DB`, `MIN_PAUSE_DURATION_SECONDS` |
| Whisper | `TRANSCRIPTION_PROVIDER`, `WHISPER_MODEL_NAME`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_LANGUAGE`, `WHISPER_BEAM_SIZE`, `WHISPER_VAD_FILTER`, `WHISPER_CPU_THREADS`, `WHISPER_NUM_WORKERS` |
| Modelos | `MODEL_DIR`, `REQUIRE_TRAINED_MODELS`, `ALLOW_HEURISTIC_MODEL_FALLBACK`, `BINARY_MODEL_FILENAME`, `MULTILABEL_MODEL_FILENAME` |
| Thresholds | `BINARY_RISK_THRESHOLD`, `MULTILABEL_RISK_THRESHOLD`, `MEDIUM_SEVERITY_THRESHOLD`, `HIGH_SEVERITY_THRESHOLD`, `LOW_ENERGY_THRESHOLD`, `LONG_PAUSE_THRESHOLD_SECONDS` |
| Saude mental | `MENTAL_HEALTH_MODEL_DIR`, `MENTAL_HEALTH_TARGET`, `MENTAL_HEALTH_THRESHOLD`, `MENTAL_HEALTH_DEVICE`, `MENTAL_HEALTH_AUDIO_EMBEDDING_MODEL`, `MENTAL_HEALTH_TEXT_EMBEDDING_MODEL`, `MENTAL_HEALTH_MAX_AUDIO_CHUNK_SECONDS` |
| Anomalia | `ANOMALY_MODEL_DIR`, `ANOMALY_MODEL_FILENAME`, `ANOMALY_THRESHOLD`, `ANOMALY_REQUIRE_MODEL`, `ANOMALY_FEATURES` |

Observacoes operacionais:

- CORS esta configurado como `allow_origins=["*"]`;
- nao ha autenticacao ou autorizacao no backend;
- os modelos sao carregados a partir de arquivos locais;
- o detector de anomalias e opcional por padrao e retorna `model_available=false` quando o artefato nao existe;
- os modelos de embedding e Whisper usam cache local;
- as rotas de treino nao existem, e os testes garantem que `/audio/train/*` nao aparece no OpenAPI.

# Camada de Modelos

## Artefatos gerados pelo projeto de modelos

| Artefato | Origem | Destino na API |
| --- | --- | --- |
| `binary_risk_model.joblib` | `processed/models/voice_risk/` | `artifacts/models/` |
| `multilabel_risk_model.joblib` | `processed/models/voice_risk/` | `artifacts/models/` |
| `model.pt` | `processed/models/multimodal_transformer/` | `artifacts/models/mental_health/multimodal_transformer/` |
| `training_config.json` | `processed/models/multimodal_transformer/` | `artifacts/models/mental_health/multimodal_transformer/` |
| `metrics.json` | `processed/models/multimodal_transformer/` | `artifacts/models/mental_health/multimodal_transformer/` |
| `anomaly_detector.joblib` | `processed/models/anomaly_detector/` | `artifacts/models/anomaly_detector/` |
| `anomaly_metrics.json` | `processed/models/anomaly_detector/` | `artifacts/models/anomaly_detector/` |
| `anomaly_config.json` | `processed/models/anomaly_detector/` | `artifacts/models/anomaly_detector/` |

## Formato de inferencia

Modelos joblib:

```text
features acusticas + features NLP
  -> DictVectorizer
  -> StandardScaler
  -> LogisticRegression / OneVsRestClassifier
  -> probabilidades
```

Transformer multimodal:

```text
audio_path + transcription + duration
  -> WavLM embedding
  -> MPNet embedding
  -> metadata opcional
  -> MultimodalTransformerClassifier
  -> sigmoid(logits)
  -> label por threshold
```

Detector de anomalias:

```text
audio_path + transcription + duration
  -> WavLM embedding
  -> MPNet embedding
  -> metadata opcional
  -> IsolationForest.decision_function
  -> score normalizado por calibracao do treino
  -> label normal/anomalia por threshold
```

# Seguranca e Privacidade

Controles implementados:

- audio nao e persistido no backend apos processamento;
- objeto S3 e removido apos inferencia;
- logs evitam corpo da requisicao, transcricao completa e audio;
- credenciais AWS sao lidas por variaveis de ambiente;
- S3 usa criptografia server-side configuravel, default `AES256`;
- `.dockerignore` documentado para impedir entrada de caches e testes no build;
- o frontend exibe aviso de que o sistema nao substitui avaliacao medica.

Riscos e lacunas:

- CORS aberto para qualquer origem;
- ausencia de autenticacao;
- ausencia de rate limiting;
- ausencia de trilha formal de auditoria de acesso;
- ausencia de criptografia ou mascaramento adicional para transcricoes no frontend;
- ausencia de politica versionada de retencao S3;
- ausencia de IAM/IaC versionado;
- ausencia de consentimento, anonimizacao e governanca documentados em codigo para dados reais.

# Escalabilidade

Caracteristicas atuais:

- processamento por requisicao, sincrono;
- cada chamada pode carregar/processar Whisper, WavLM, MPNet e transformer;
- caches `lru_cache` reduzem custo de recarregamento de modelos;
- S3 e usado apenas como armazenamento temporario;
- Fargate/ECS e mencionado como ambiente-alvo, mas sem IaC.

Limitacoes para alta escala:

- sem fila para absorver picos;
- sem worker assincrono para processamento pesado;
- sem stream parcial de resposta;
- sem WebSocket;
- sem autoscaling policy versionada;
- sem cache compartilhado de modelos entre replicas;
- sem controle de concorrencia por tamanho de audio.

# Resultados Obtidos

Resultados locais encontrados no projeto de modelos:

| Artefato | Resultado |
| --- | --- |
| `raw_validation_report.json` | 1008 audios brutos, 0 erros |
| `metadata.parquet` | 4068 registros normalizados |
| `audio_manifest.parquet` | 1008 audios processados |
| `audio_embeddings.parquet` | 1007 embeddings de audio |
| `text_embeddings.parquet` | 993 embeddings de texto |
| `splits.parquet` | train 2789, validation 690, test 589 |

Distribuicao por dataset em `metadata.parquet`:

| Dataset | Registros |
| --- | ---: |
| EMU | 3061 |
| EATD-Corpus | 971 |
| WomanHealthFIAP | 14 |
| VoiceRiskTraining | 14 |
| VoiceRiskSynthetic | 8 |

Distribuicao do Unified:

| Split | Linhas | Observacao |
| --- | ---: | --- |
| train | 2789 | inclui 14 linhas com `voice_risk_label` |
| validation | 690 | inclui 6 linhas com `voice_risk_label` |
| test | 589 | inclui 2 linhas com `voice_risk_label` |

Metricas do baseline scikit-learn para `depression_label` com audio:

| Split | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| validation | 0.8188 | 0.1379 | 0.4253 |
| test | 0.6597 | 0.3288 | 0.6646 |

Metricas do transformer multimodal para `depression_label`:

| Split | Accuracy | F1 | ROC-AUC | Loss |
| --- | ---: | ---: | ---: | ---: |
| validation | 0.8406 | 0.2667 | 0.7546 | 2.1216 |
| test | 0.6875 | 0.2623 | 0.6524 | 7.4349 |

Configuracao observada do transformer:

- melhor epoca: 12;
- monitor de validacao: ROC-AUC 0.7546;
- modalidades: audio e texto;
- alvo: `depression_label`.

Metricas dos modelos voice risk:

| Modelo | Linhas | Metricas |
| --- | ---: | --- |
| binario | 14 | accuracy 1.0, ROC-AUC 1.0 |
| multilabel | 14 | subset_accuracy 1.0, label_accuracy 1.0 |

Importante: as metricas de voice risk foram calculadas sobre um dataset pequeno e supervisionado localmente, sem evidencia de holdout externo nesse script. Elas devem ser interpretadas como validacao tecnica do pipeline e nao como desempenho clinico.

Metricas do detector de anomalias:

- o codigo gera `processed/models/anomaly_detector/anomaly_metrics.json`;
- as metricas incluem taxa de anomalia por split, quantis de score, linhas ignoradas e avaliacao auxiliar contra `depression_label` quando disponivel;
- neste relatorio nao foi identificado um resultado real versionado desse treinamento, portanto os valores numericos devem ser gerados localmente com `python src/train_anomaly_detector.py`.

# Exemplos de Anomalias Detectadas

Os exemplos abaixo sao resultados esperados da implementacao com base nas regras, thresholds e modelos existentes. Eles nao sao diagnosticos.

## Exemplo 1: ansiedade

Entrada textual esperada:

```text
"Tenho ansiedade, fico ansiosa e meu coracao acelerado aparece em crise."
```

Sinais detectaveis:

- categorias NLP: `anxiety`, possivelmente `fear`;
- score textual elevado;
- se o audio tiver pausas e hesitacao, `hesitation_score` aumenta;
- modelo multilabel pode acionar `anxiety`;
- frontend exibe card de ansiedade como detectado.

Alerta esperado:

```text
alert_type: anxiety
severity: baixo/medio/alto conforme score
justificativas:
  - Score do modelo
  - Termos encontrados
  - Pausas longas, se houver
  - Padrao de hesitacao, se houver
```

## Exemplo 2: depressao pos-parto

Entrada textual esperada:

```text
"Depois do parto me sinto triste, sozinha e sem esperanca com o bebe."
```

Sinais detectaveis:

- categorias NLP: `postpartum`, `sadness`, `isolation`;
- risco multilabel `postpartum_depression`;
- possivel alerta se score ultrapassar threshold.

## Exemplo 3: fadiga hormonal

Entrada textual esperada:

```text
"Estou exausta, com fadiga hormonal, sem energia e nao durmo."
```

Sinais detectaveis:

- categorias NLP: `exhaustion`, `hormonal`;
- baixa energia vocal aumenta justificativa acustica;
- risco multilabel `hormonal_fatigue`.

## Exemplo 4: violencia domestica

Entrada textual esperada:

```text
"Tenho medo de voltar para casa, ele me controla e ja me empurrou."
```

Sinais detectaveis:

- categorias NLP: `fear`, `coercion`, `violence`, `insecurity`;
- risco multilabel `domestic_violence`;
- alerta com mensagem "Sinal associado a violencia domestica identificado."

## Exemplo 5: alteracao vocal

Sinais acusticos esperados:

```text
max_pause_duration_seconds >= 1.2
energy_mean <= 0.015
hesitation_score >= 0.45
silence_ratio >= 0.35
```

Justificativas esperadas:

- pausas longas detectadas;
- baixa energia vocal media detectada;
- padrao de hesitacao elevado;
- proporcao de silencio elevada.

## Exemplo 6: anomalia multimodal nao supervisionada

Entrada esperada:

- audio e texto validos;
- embedding de audio e/ou texto distante do padrao aprendido no Unified;
- duracao ou metadados fora da faixa comum, quando a modalidade `metadata` foi usada no treino.

Sinais detectaveis:

- `anomaly_prediction.score` elevado;
- `anomaly_prediction.label = "anomalia"` quando o score ultrapassa o limiar;
- alerta `anomaly` gerado pela API;
- frontend exibe `AnomalyCard` com score e disponibilidade do artefato.

Alerta esperado:

```text
alert_type: anomaly
severity: baixo/medio/alto conforme score
message: Anomalia multimodal identificada para revisao humana.
```

# Conclusao

Os tres repositorios formam uma arquitetura coerente para o escopo de triagem vocal multimodal:

- o projeto de modelos entrega pipeline reproduzivel de dados, embeddings e treinamento;
- o backend entrega inferencia multimodal operacional com FastAPI, S3 temporario, Whisper, features acusticas, NLP, modelos joblib, transformer e detector de anomalias;
- o frontend entrega uma experiencia funcional de upload, visualizacao, score de anomalia, alertas e explicabilidade.

A arquitetura ja contempla os principais blocos do desafio: analise de voz, processamento multimodal, geracao de alertas, suporte a decisao e integracao com AWS S3. Entretanto, alguns itens do contexto do desafio estao documentados ou seriam evolucoes arquiteturais, mas nao aparecem implementados no codigo analisado: CloudFront, IaC de ECS e monitoramento operacional completo.

Para uso corporativo ou clinico real, os proximos passos recomendados sao:

- restringir CORS e adicionar autenticacao;
- versionar infraestrutura AWS;
- criar validacao clinica com dados anonimizados e aprovados;
- separar pipeline assincrono para inferencias pesadas;
- adicionar governanca LGPD, consentimento, retencao e auditoria;
- avaliar desempenho por subgrupo, classe e qualidade de audio.
