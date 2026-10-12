# ArkIve — Motor de Inteligência Clínica Veterinária

> **FIAP Challenge 2026 — 2º Ano ADS | Turmas de Fevereiro**
> Parceria: **Clyvo Vet** · Disciplina: *Disruptive Architectures: IoT, IoB & Generative AI*

---

## Equipe

| Nome | RM |
|------|----|
| Gustavo Crevelari | RM561408 |
| Lucca Gomes | RM561996 |
| Rafaela Ferreira | RM561671 |
| Victor Sabelli | RM566224 |

---

## Repositório
 
[![GitHub](https://img.shields.io/badge/GitHub-Acessar%20Repositório-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/2TDSPO-ArkIve/arkive_clinical_engine)

---

## Demonstração em Vídeo

[![YouTube: Assistir Sprint 1 e 2](https://img.shields.io/badge/YouTube-Assistir%20Sprint%201%20e%202-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=zWqLgywXfv4)  
[![YouTube: Assistir Sprint 3](https://img.shields.io/badge/YouTube-Assistir%20Sprint%203-red?style=for-the-badge&logo=youtube)](https://youtu.be/WguMNFGqEAs)

---

## Problema Abordado

A jornada de saúde do pet é **fragmentada e reativa**. Responsáveis e veterinários interagem apenas em momentos pontuais — vacinas, emergências, retornos — sem continuidade inteligente entre as consultas.

Do ponto de vista clínico, isso significa que o veterinário frequentemente:

- Não tem acesso rápido ao histórico consolidado do animal no momento da consulta;
- Precisa cruzar manualmente sintomas, raça, espécie e predisposições genéticas;
- Toma decisões sem suporte de evidências clínicas atualizadas.

**Impacto direto:** agravamento evitável de quadros, baixa adesão a tratamentos e perda de recorrência para as clínicas.

---

## Solução Proposta

O **Motor de Inteligência Clínica Veterinária ArkIve** é um microsserviço Python que, a partir do ID de uma consulta veterinária, extrai automaticamente os dados clínicos do banco Oracle (histórico do animal, sintomas, bem-estar, predisposições genéticas da raça e diagnósticos anteriores), e aciona um modelo de linguagem (LLM) via API para gerar uma **hipótese diagnóstica estruturada** — pronta para ser consumida pelo serviço Java que persiste o resultado no banco.

O sistema opera em **modo estritamente read-only** no banco, nunca escrevendo ou alterando dados. Atende qualquer espécie animal — doméstica, silvestre, zoológica ou de produção. Disponível tanto via CLI (`main.py`) quanto via API REST (`api.py`).

### Como a solução melhora a jornada

| Para quem | Benefício |
|-----------|-----------|
| **Pet** | Hipótese diagnóstica mais fundamentada, considerando predisposição genética e histórico clínico |
| **Responsável** | Continuidade do cuidado — cada consulta alimenta a inteligência do sistema |
| **Veterinário** | Apoio à decisão clínica em segundos, com raciocínio explicado e grau de confiança |
| **Clínica** | Diferencial competitivo com IA integrada ao prontuário existente |

---

## Componente de Inteligência Artificial

O motor de IA é o núcleo do ArkIve. Abaixo: como ele raciocina, o que usa como base clínica e
como se conecta ao resto da plataforma.

Retomando o [problema abordado](#problema-abordado): sem inteligência ligando uma consulta à
seguinte, o veterinário decide sem o histórico consolidado à mão. A cada consulta registrada, o
motor consolida automaticamente o que se sabe daquele animal e devolve uma **hipótese diagnóstica
estruturada, explicada e com grau de confiança**, para o veterinário revisar e validar — e essa
validação já entra como contexto da próxima consulta. É isso que torna a jornada *contínua*.

### Um motor híbrido: regras + LLM

A abordagem é **híbrida — motor de regras determinístico + LLM (IA Generativa) com saída
estruturada validada**. Como cada parte é usada:

- **Motor de regras determinístico (Python puro, sem LLM):**
  - `_calculate_confidence()` — rubrica de pontos fixa sobre os dados reais do Oracle, produz
    `pc_confianca` (0–100). Auditável, reproduzível, custo zero.
  - `_decide_web_search()` — usa o **mesmo** `pc_confianca` para decidir se aciona a busca web
    (`pc_confianca < AMBIGUITY_THRESHOLD`). Fonte única de verdade, sem heurística paralela.
  - match de predisposição × sintomas por palavras-chave catalogadas (`TB_ARKIVE_DOENCA.DS_SINTOMAS`).
- **LLM (Groq API, `ChatGroq.with_structured_output`):** faz a **síntese clínica em português**,
  explicável, sobre contexto heterogêneo (perfil + sintomas + transcrição + bem-estar + histórico
  + medicações + status preventivo + literatura web opcional). Saída **validada por Pydantic v2**
  (`DiagnosticoOutputDetalhado`), `temperature=0.10`, com fallback automático entre modelos.
- **RAG local (ONNX + FAISS):** sempre que há relato clínico, busca semântica numa base
  curada de 782 fichas veterinárias (WOAH, AAZV, CFSPH, CAPC, ABCD, USGS, ARWH, WHA),
  filtrada pela espécie do animal. Os `RAG_TOP_K` documentos mais relevantes entram no prompt e
  seus nomes voltam em `fontes_rag`. Roda localmente, sem custo de API e independe do
  `pc_confianca` (ver [RAG — Base de Conhecimento Local](#rag--base-de-conhecimento-local)).
- **Busca web condicional (PubMed, fallback DuckDuckGo via `ddgs`):** complemento ao RAG quando a confiança é
  baixa, restrito a uma lista curada de fontes veterinárias confiáveis (ver
  [Fluxo de Decisão — Busca Web](#fluxo-de-decisão--busca-web)).
- **Prompt engineering:** system prompt com travas anti-alucinação explícitas
  (`prompts/diagnostic.py`).

Em resumo: determinismo onde existe regra clara (confiança, gatilho de busca, match de
predisposição); IA Generativa onde é preciso julgamento e redação — sempre com validação de schema
e o `pc_confianca` calculado fora do modelo.

### Personalização por animal

A personalização é **por animal**, não por um perfil genérico de espécie/raça. A cada execução, o
motor monta o contexto daquele animal específico a partir de:

| Sinal de personalização | Origem | Como personaliza |
|---|---|---|
| Perfil do paciente | `TB_ARKIVE_ANIMAL` + `TB_ARKIVE_ESPECIE` + `TB_ARKIVE_RACA` | espécie, raça exata, porte, sexo, status reprodutivo, idade precisa (`DT_NASCIMENTO`, com fallback para `NR_IDADE`) |
| Narrativa da consulta atual | `TB_ARKIVE_CONSULTA` (`DS_MOTIVO`, `DS_SINTOMAS`, `DS_TRANSCRICAO`) | relato bruto do veterinário daquela consulta |
| Bem-estar longitudinal | `TB_ARKIVE_AVALIACAO_BEM_ESTAR` | apetite, atividade, comportamento, peso e idade ao longo do tempo |
| Predisposição genética | `TB_ARKIVE_PREDISPOSICAO` + `TB_ARKIVE_DOENCA` | herança nível espécie (sempre) + raça exata do animal |
| Histórico de diagnósticos | `TB_ARKIVE_DIAGNOSTICO` (outras consultas) | recorrência, progressão de severidade, diagnósticos não validados pelo vet |
| Medicações e adesão | `TB_ARKIVE_PRESCRICAO` + `TB_ARKIVE_ADESAO_PRESCRICAO` | tratamento em curso, falha de resposta, baixa adesão como risco |
| Cuidado preventivo | `TB_ARKIVE_PROTOCOLO_PREVENTIVO` + `TB_ARKIVE_EVENTO_PREVENTIVO` | vacina/vermífugo/check-up atrasado como fator de risco de fundo |

O `pc_confianca` também é personalizado: ele é calculado sobre os dados reais **daquele** animal e
decide, para aquele caso, se vale enriquecer o raciocínio com literatura externa.

### Dados clínicos que a IA utiliza

Todos os dados vêm do banco Oracle da FIAP, em **modo estritamente read-only** (apenas `SELECT`).

| Fonte (origem) | Estrutura (colunas-chave) | Utilização pela IA |
|---|---|---|
| `TB_ARKIVE_ANIMAL` | `NM_ANIMAL`, `DS_SEXO`, `DS_CASTRADO`, `DT_NASCIMENTO`, `ID_ESPECIE`, `ID_RACA` | perfil-base do paciente |
| `TB_ARKIVE_ESPECIE` / `TB_ARKIVE_RACA` | `NM_ESPECIE`; `NM_RACA`, `TP_PORTE` | correlação de sintomas com espécie/raça/porte |
| `TB_ARKIVE_CONSULTA` | `DT_HORA`, `TP_MODALIDADE`, `DS_MOTIVO`, `DS_SINTOMAS`, `DS_OBSERVACAO`, `DS_TRANSCRICAO`, `KG_PESO` | quadro clínico atual; `DS_TRANSCRICAO` = relato bruto (voz) do veterinário |
| `TB_ARKIVE_AVALIACAO_BEM_ESTAR` | `NR_IDADE`, `KG_PESO`, `DS_APETITE`, `DS_ATIVIDADE`, `DS_COMPORTAMENTO` | indicadores sistêmicos; avaliação mais recente do animal |
| `TB_ARKIVE_DOENCA` + `TB_ARKIVE_PREDISPOSICAO` | `NM_DOENCA`, `DS_DOENCA`, `DS_SINTOMAS`; `ID_ESPECIE`, `ID_RACA`, `ID_DOENCA` | predisposições genéticas da espécie (sempre) e da raça exata |
| `TB_ARKIVE_DIAGNOSTICO` (outras consultas) | `DS_DIAGNOSTICO`, `TP_SEVERIDADE`, `PC_CONFIANCA`, `ST_CONFIRMADO`, `ST_VALIDACAO_VET` | continuidade de cuidado: recorrência, progressão, ceticismo com o não validado |
| `TB_ARKIVE_PRESCRICAO` + `TB_ARKIVE_ADESAO_PRESCRICAO` | `NM_MEDICAMENTO`, `DS_DOSAGEM`, `DS_FREQUENCIA`, `TP_VIA_ADMINISTRACAO`, `DT_INICIO`/`DT_FIM`; `ST_TOMOU` | medicações vigentes, resposta ao tratamento, adesão terapêutica |
| `TB_ARKIVE_PROTOCOLO_PREVENTIVO` + `TB_ARKIVE_EVENTO_PREVENTIVO` | `NM_PROTOCOLO`, `TP_PROTOCOLO` (`VACINA`/`VERMIFUGO`/`CHECK-UP`/`ANTIPARASITARIO`), `ST_STATUS`, `DT_PROXIMO` | imunização/vermifugação em atraso como fator de risco e recomendação |

**Saída da IA** — `DiagnosticoOutputDetalhado` (Pydantic v2), consumida pela API Java e persistida
em `TB_ARKIVE_DIAGNOSTICO`: ver [Schema de Saída](#schema-de-saída-pydantic-v2).

### Arquitetura e fluxo de dados

1. **Veterinário** usa a aplicação (app/web) para registrar animal, consulta, bem-estar,
   prescrições e eventos preventivos.
2. A **API Java** faz o CRUD e persiste tudo no **Oracle** (`TB_ARKIVE_*`).
3. Ao concluir uma consulta, a API Java aciona o **Motor ArkIve** por `ID_CONSULTA` — via CLI
   (`python main.py <id>`) ou REST (`GET /diagnostico/{id_consulta}`).
4. O motor **lê o Oracle em modo read-only** (5 consultas: dados clínicos, predisposições,
   histórico de diagnósticos, prescrições, cuidado preventivo), calcula o `pc_confianca`,
   consulta a **base local RAG** (fichas veterinárias indexadas), opcionalmente consulta o
   **DuckDuckGo**, e chama a **Groq API**.
5. O motor devolve um **JSON estruturado** (`DiagnosticoOutputDetalhado` + `ds_insight_ia`).
6. A **API Java grava** o resultado em `TB_ARKIVE_DIAGNOSTICO`.
7. A aplicação **exibe a hipótese** ao veterinário, que a valida (`ST_VALIDACAO_VET`) — e essa
   validação entra no histórico da próxima consulta.

```mermaid
graph TD
    subgraph App["Aplicação CLYVO VET"]
        U["Veterinário<br/>(app / web)"]
        J["API Java<br/>(CRUD + persistência)"]
    end

    ORA[("Oracle FIAP<br/>TB_ARKIVE_*")]

    subgraph Motor["Motor de Inteligência Clínica ArkIve — Python (READ-ONLY no banco)"]
        direction TB
        E1["1. Extração clínica<br/>5 SELECTs parametrizados"]
        E2["2. pc_confianca<br/>rubrica determinística (Python puro)"]
        E2b["2b. RAG local<br/>busca semântica filtrada por espécie"]
        E3["3. Decisão de busca web<br/>pc_confianca &lt; AMBIGUITY_THRESHOLD?"]
        E4["4. Busca web (condicional)"]
        E5["5. Síntese clínica (LLM)<br/>resumo + RAG + web<br/>saída validada por Pydantic v2"]
        E1 --> E2 --> E2b --> E3 --> E4 --> E5
        IDX[("rag/index<br/>FAISS + chunks<br/>WOAH, AAZV, CFSPH, CAPC, ABCD…")]
        IDX -.->|"carregado no boot · e5-small ONNX"| E2b
    end

    DDG["DuckDuckGo<br/>fontes veterinárias confiáveis<br/>(PubMed, Merck/MSD, WSAVA, SciELO…)"]
    GROQ["Groq API<br/>LLM + fallback entre modelos"]

    U -->|"HTTPS / REST"| J
    J -->|"SQL (escrita)"| ORA
    J -->|"aciona: ID_CONSULTA<br/>(CLI ou GET /diagnostico/{id})"| E1
    ORA -.->|"oracledb Thin · SELECT apenas"| E1
    E4 <-->|"HTTPS"| DDG
    E5 <-->|"HTTPS"| GROQ
    E5 -->|"JSON: DiagnosticoOutputDetalhado"| J
    J -->|"INSERT em TB_ARKIVE_DIAGNOSTICO"| ORA
    J -->|"exibe hipótese + grau de confiança"| U
```

A fronteira do motor com o banco é **somente leitura** em três camadas (privilégio `GRANT SELECT`,
`autocommit = False`, `rollback()` no `finally`) — ver [Garantias de Segurança](#garantias-de-segurança-read-only).

---

## Tecnologias Utilizadas

| Camada | Tecnologia | Papel no sistema |
|--------|-----------|-----------------|
| Linguagem | Python 3.11+ | Orquestra todas as etapas |
| LLM / IA Generativa | [Groq API](https://console.groq.com) · modelo configurável via `.env`, com fallback automático entre modelos | Gera o raciocínio clínico e o diagnóstico |
| Integração LLM | `langchain-groq` + `langchain-core` · `ChatGroq.with_structured_output()` | Conecta ao Groq e garante saída JSON validada pelo Pydantic; sem chains ou pipelines LCEL |
| Banco de Dados | Oracle via `oracledb` (modo Thin) | Fonte de dados clínicos — somente leitura |
| Validação de Schema | Pydantic v2 | Valida e tipifica a saída da IA |
| RAG local | `onnxruntime` + `tokenizers` (`Sabelli/arkive-e5-vet`: e5-small fine-tunado, int8) + `faiss-cpu` | Busca semântica na base de fichas veterinárias (`rag/index/`), sem chamada externa |
| Busca Web (fallback) | `ddgs` (DuckDuckGo Search) | Literatura veterinária complementar — só resultados de uma lista curada de fontes confiáveis são aproveitados |
| API REST | FastAPI + Uvicorn | Endpoint HTTP alternativo ao CLI (`GET /diagnostico/{id_consulta}`) + `GET /health` (liveness) |
| Variáveis de Ambiente | `python-dotenv` | Isola credenciais do código-fonte |

---

## Arquitetura do Sistema

O diagrama arquitetural completo (aplicação, banco, APIs e componentes de IA) está na seção
[Arquitetura e fluxo de dados](#arquitetura-e-fluxo-de-dados).
Abaixo, o detalhamento do pipeline interno do motor:

```
Entrada: main.py ──► python main.py <ID_CONSULTA>   (ou api.py ──► GET /diagnostico/{id_consulta})
Etapa 1 ──► Oracle (READ-ONLY): 5 SELECTs parametrizados extraem animal (incl. DT_NASCIMENTO), espécie, raça,
             consulta (incl. DS_TRANSCRICAO — relato bruto do veterinário), bem-estar, predisposições genéticas (nível espécie +
             raça exata), os últimos DIAGNOSTIC_HISTORY_LIMIT diagnósticos anteriores do animal, as últimas
             HISTORICO_CUIDADO_LIMIT prescrições (com adesão) e os eventos de cuidado preventivo
             (vacina/vermífugo/check-up, priorizando ATRASADO/PENDENTE).
Etapa 2 ──► Python puro (sem LLM): _calculate_confidence() calcula pc_confianca com rubrica fixa baseada na
             narrativa clínica (DS_TRANSCRICAO + DS_SINTOMAS combinados) e nos demais dados reais do Oracle
             (inclui match de predisposição via nome da doença e DS_SINTOMAS catalogado).
Etapa 2b ─► RAG local (sempre que há relato): DS_MOTIVO + narrativa clínica viram consulta semântica (expandida
             com glossário PT→EN) no índice FAISS; resultados filtrados pela espécie e agregados por documento;
             top RAG_TOP_K vão ao prompt (bloco de até RAG_MAX_CHARS caracteres).
Etapa 3 ──► Python puro (sem LLM): _decide_web_search() aciona busca web se pc_confianca < AMBIGUITY_THRESHOLD —
             mesma métrica usada em toda a decisão, sem heurística paralela.
Etapa 4 ──► DuckDuckGo (condicional): busca literatura veterinária; um pós-filtro descarta todo resultado
             que não seja de uma lista curada de fontes confiáveis (PubMed/PMC, Merck & MSD Vet Manual,
             WSAVA, AVMA, periódicos peer-reviewed, SciELO, CFMV…). PubMed (E-utilities) roda primeiro,
             com os títulos do RAG (ou os termos em inglês do glossário) como busca, filtrada pela espécie
             (MeSH, ex. Felidae[mh]; sem resultado, repete sem o filtro). Vão ao LLM 3 artigos com o
             abstract (até 700 caracteres). O ddgs só entra se o
             PubMed voltar vazio (Google e DuckDuckGo bloqueiam IP de datacenter, como o do Render).
Etapa 5 ──► Groq API (com fallback + retry entre modelos — ver seção dedicada): recebe resumo clínico
             (incl. medicações vigentes e status de cuidado preventivo) + pc_confianca pronto + bloco RAG +
             contexto web (se houver) e gera DiagnosticoOutputDetalhado (4 campos de raciocínio clínico),
             validado pelo Pydantic v2.
Saída: JSON ──► {ds_diagnostico, tp_severidade, ds_insight_ia, pc_confianca, fontes_rag, fontes_pesquisadas, tokens, + campos de
             insight individuais (insight_perfil, insight_correlacao, insight_predisposicao, insight_limitacoes)}

A resposta é consumida pela API Java para persistência em TB_ARKIVE_DIAGNOSTICO.
```

### Estrutura de Arquivos

```
arkive_clinical_engine/
├── .env                            # Variáveis de ambiente (versionado; segredos ficam como <API_KEY>)
├── requirements.txt                # Dependências com versões fixas
├── config.py                       # Configuração centralizada + validação fail-fast
├── main.py                         # Ponto de entrada CLI
├── api.py                          # Ponto de entrada API REST (FastAPI) — inclui GET /health (liveness)
├── agents/
│   └── clinical_agent.py           # Motor principal (rubrica determinística + RAG + busca web + Groq)
├── database/
│   ├── connection.py               # Conexão Oracle Thin mode, READ-ONLY
│   └── queries.py                  # 5 SQLs parametrizados + dataclass ClinicalContext
├── prompts/
│   └── diagnostic.py               # System prompt do Groq (histórico de versões fica no Git)
├── rag/
│   ├── build_index.ipynb           # Gera o índice (Colab): chunking, filtros de seção, embeddings
│   ├── finetune_e5.ipynb           # Fine-tuning do e5 (Colab T4), exporta ONNX int8 e publica no HF privado
│   ├── ajuste_rag.ipynb            # Varredura de RAG_MIN_SCORE e custo de RAG_MAX_CHARS (Colab CPU, minutos)
│   ├── retriever.py                # RagRetriever: busca semântica + bloco de contexto para o prompt
│   ├── eval/                       # avaliar.py + casos.jsonl (108 casos rotulados), pares.jsonl, baseline.json
│   ├── index/                      # faiss.index + chunks.jsonl + manifest.json (versionado)
│   └── model_cache/                # Modelo de embedding baixado (ignorado no Git)
└── schemas/
    └── diagnostic_detalhado.py     # Pydantic v2: DiagnosticoOutputDetalhado
```

---

## Como Executar (How To)

### Pré-requisitos

- Python 3.11 ou superior
- Acesso ao banco Oracle da FIAP (`oracle.fiap.com.br`)
- Conta gratuita no [Groq Console](https://console.groq.com) para obter a API Key

### 1. Clonar o repositório

```bash
git clone https://github.com/2TDSPO-ArkIve/arkive_clinical_engine.git
cd arkive_clinical_engine
```

### 2. Criar e ativar o ambiente virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

> **Atenção — Windows:** se ocorrer erro de compilação C++ ao instalar `oracledb`, instale o [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) ou tente:
> ```bash
> pip install oracledb==2.3.0 --only-binary=:all:
> ```

**RAG:** o índice já vem pronto em `rag/index/` (para regerar, rode `rag/build_index.ipynb` no Colab). Falta baixar o modelo de embedding (~113 MB) e gerar o tokenizer podado, ambos em `rag/model_cache/`. O modelo fica num repo privado do HF Hub: defina `HF_TOKEN` (token **Read**) no `.env` ou no ambiente:

```bash
python -m rag.retriever --download
python -m rag.retriever            # teste com relato de exemplo
```

Se o RAG não carregar, a API sobe normalmente e só registra `RAG indisponível` no log. No Render, use como Build Command: `pip install -r requirements.txt && python -m rag.retriever --download`, com `HF_TOKEN` nas variáveis de ambiente.

> **Windows com Smart App Control:** o Windows pode bloquear a DLL do `faiss` (`DLL load failed while importing _swigfaiss: Uma política de Controle de Aplicativo bloqueou este arquivo`). O motor segue funcionando sem RAG; para testar o RAG, use Linux/WSL ou o deploy no Render.

### 4. Configurar as variáveis de ambiente

Edite o `.env`:

```env
# Banco Oracle FIAP
ORACLE_DSN=oracle.fiap.com.br:1521/ORCL
ORACLE_USER=seu_usuario_fiap
ORACLE_PASSWORD=sua_senha_fiap

# Groq (obtenha em https://console.groq.com → API Keys)
GROQ_API_KEY=gsk_sua_chave_aqui
GROQ_MODEL_PRIMARY=openai/gpt-oss-120b
GROQ_MODEL_FALLBACKS=qwen/qwen3.8-27b,openai/gpt-oss-20b
GROQ_REASONING_EFFORT=low   # só nos gpt-oss: low, medium ou high
GROQ_MAX_RETRIES_PER_MODEL=2
GROQ_RETRY_BACKOFF_SECONDS=2

# Configurações opcionais
LOG_LEVEL=INFO
AMBIGUITY_THRESHOLD=60
DIAGNOSTIC_HISTORY_LIMIT=5
HISTORICO_CUIDADO_LIMIT=5
MAX_TRANSCRICAO_CHARS=6000
RAG_TOP_K=3          # documentos do RAG enviados ao LLM
RAG_MAX_CHARS=4500   # tamanho do bloco RAG (dividido entre os documentos)
RAG_MIN_SCORE=0.45   # similaridade mínima para um documento do RAG ir ao LLM
HF_TOKEN=<token Read do HF>  # modelo do RAG em repo privado
```

### 5. Executar

```bash
python main.py <ID_CONSULTA>
```

**Exemplo:**

```bash
python main.py 1
```

**Alternativa via API REST:**

```bash
uvicorn api:app --reload
# GET http://localhost:8000/diagnostico/1
# GET http://localhost:8000/health      → {"status": "ok", "rag": true} (liveness + RAG carregado)
```

### 6. Saída esperada

```json
{
  "ds_diagnostico": "Suspeita de Gastroenterite Infecciosa Canina",
  "tp_severidade": "MODERADA",
  "insight_perfil": "Paciente Rex, canino, macho...",
  "insight_correlacao": "Vômito frequente e fezes moles correlacionam-se com...",
  "insight_predisposicao": "Nenhuma predisposição mapeada localmente; limitação de dado...",
  "insight_limitacoes": "Recomenda-se exame de fezes e hemograma completo...",
  "ds_insight_ia": "Paciente Rex, canino, macho...\n\nVômito frequente...\n\nNenhuma predisposição...\n\nRecomenda-se...",
  "pc_confianca": 55,
  "fontes_pesquisadas": [],
  "fontes_rag": [{"fonte": "CAPC", "titulo": "Giardia"}],
  "tokens": {"entrada": 4810, "saida": 1320, "total": 6130}
}
```

`tokens` traz o consumo da chamada ao Groq (entrada, saída e total); não faz parte do schema e não é persistido. `ds_insight_ia` é a composição textual dos 4 campos de insight (perfil + correlação + predisposição + limitações), mantida para compatibilidade com a coluna única `DS_INSIGHT_IA` (CLOB) esperada pelo serviço Java.

Logs de execução são exibidos no terminal. Em caso de erro, o JSON de saída conterá o campo `"error"` com a causa.

---

## Fluxo de Decisão — Busca Web

A busca web é acionada com base no **mesmo `pc_confianca`** calculado deterministicamente (ver seção seguinte) — não existe uma métrica de "ambiguidade" separada. Se `pc_confianca < AMBIGUITY_THRESHOLD` (padrão: 60%), o DuckDuckGo é acionado para buscar literatura veterinária atualizada, enriquecendo o contexto antes da chamada à LLM.

Só é aproveitado o resultado que vier de uma **lista curada de ~108 domínios de fontes veterinárias confiáveis**, cobrindo:

- bases indexadas (PubMed/PMC, Europe PMC, CAB Direct, SciELO, BVS-Vet);
- publishers peer-reviewed (Wiley, ScienceDirect, Springer, Cambridge, SAGE, Taylor & Francis, PLOS, Frontiers, MDPI, BMC, BMJ/Vet Record);
- manuais e referência clínica (Merck & MSD Vet Manual, VIN, IVIS, Plumb's, IRIS, Vetstream, Clinician's Brief);
- diretrizes de associações (WSAVA, AVMA, AAHA, BSAVA, FECAVA, AAEP, ABCD, Anclivepa, ABROVET);
- colleges de especialidade (ACVIM/ACVS/ACVP/ACVECC/ACVO/ACVD/ACVR/ACVB/ACVAA/ACVN, ECVIM-CA, ECVS);
- conselhos e reguladores (CFMV, CRMV-SP, RCVS, FVE, CVMA);
- autoridades sanitárias / One Health (WOAH, FAO, WHO, CDC, EFSA, USDA, Embrapa);
- toxicologia e parasitologia (ASPCA Poison Control, Pet Poison Helpline, CAPC, ESCCAP);
- periódicos e escolas de veterinária do Brasil e do exterior.

Qualquer outro domínio é descartado (inclusive `scholar.google` e landing pages de bases pagas). Se nada confiável for encontrado, o motor segue sem contexto externo (`fontes_pesquisadas: []`). A lista completa fica em `TRUSTED_VET_DOMAINS` (`agents/clinical_agent.py`).

**Resultado: normalmente 1 chamada à API do Groq por execução.** Em caso de erro transitório ou cota esgotada, o sistema pode tentar novamente ou trocar de modelo automaticamente — ver "Modelos Groq — Fallback e Retry" abaixo.

> **Dica para testes:** defina `AMBIGUITY_THRESHOLD=101` no `.env` para forçar a busca web em todas as execuções, já que `pc_confianca` nunca ultrapassa 100.

---

## RAG — Base de Conhecimento Local

Diferente da busca web, o RAG roda **em toda consulta com relato clínico**. É local, barato e não depende do `pc_confianca`.

**Base indexada** (`rag/index/`, gerada por `rag/build_index.ipynb`): 782 fichas em inglês (811 arquivos), 13.257 trechos.

| Fonte | Conteúdo | Trechos |
|---|---|---|
| WOAH | Manuais terrestre e aquático: doenças de notificação obrigatória, diagnóstico e vacinas | 4.501 |
| WHA | Wildlife Health Australia: fichas de doenças da fauna | 2.121 |
| CFSPH | Fichas de doenças infecciosas e zoonoses (Iowa State) | 1.756 |
| ABCD | Doenças infecciosas felinas | 1.560 |
| AAZV | Fichas de doenças de animais de zoológico e silvestres | 1.499 |
| CAPC | Parasitas de animais de companhia | 865 |
| USGS | Doenças de aves silvestres | 613 |
| ARWH | Australian Registry of Wildlife Health: fauna silvestre | 342 |

**Como a busca funciona** (`rag/retriever.py`):

1. A consulta é `DS_MOTIVO` + narrativa clínica, fatiada em janelas e expandida com um glossário PT→EN (a base está em inglês).
2. Embedding com `Sabelli/arkive-e5-vet` (ONNX int8 via `onnxruntime`) e busca por similaridade de cosseno no FAISS.
   É o `multilingual-e5-small` fine-tunado em relatos sintéticos PT → fichas EN (`rag/finetune_e5.ipynb`, pares em `rag/eval/pares.jsonl`).
   O tokenizer multilíngue completo (250k peças) ocupa ~250 MB e não cabe nos 512 MB do Render; o build
   (`--download`) grava uma versão podada às peças com caracteres latinos/gregos/símbolos e do corpus (~80 MB em memória),
   com tokenização idêntica à original em todos os 13.257 trechos.
3. Resultados filtrados pela espécie do animal (`MAPA_ESPECIE`). Fichas multiespécie sempre entram, mas perdem 0.01 de score (`PENALIDADE_GENERICA`) para não lotar o top-k; espécie não mapeada = sem filtro.
4. Trechos agrupados por documento (fonte + título). Documentos abaixo de `RAG_MIN_SCORE` são descartados; os `RAG_TOP_K` melhores restantes vão ao prompt com resumo + trechos, limitados a `RAG_MAX_CHARS` caracteres. Se nada passar do corte, o RAG fica vazio.
5. O system prompt avisa que relevância é similaridade de texto, não probabilidade da doença: o LLM só usa um documento se os sinais clínicos e a epidemiologia (idade, espécie, exposição) forem compatíveis, e nunca o põe no `ds_diagnostico` com base só em sintoma genérico. Os documentos enviados voltam em `fontes_rag`.

> **Fine-tuning:** no conjunto rotulado (`rag/eval/casos.jsonl`, 108 casos), o modelo próprio int8 contra o e5 genérico:
>
> | Métrica | e5 genérico (corte 0.84) | fine-tunado int8 (corte 0.48) |
> |---|---|---|
> | recall@3 | 0.376 | 0.634 |
> | MRR@10 | 0.327 | 0.510 |
> | recall@3 após o corte | 0.194 | 0.194 |
> | ruído@3 | 0.330 | 0.093 |
>
> Corte atual 0.45, escolhido em `rag/ajuste_rag.ipynb` (menor corte com ruído@3 ≤ 0.15): recall@3 após o corte 0.247, ruído@3 0.142.
>
> Para medir de novo: `python -m rag.eval.avaliar` (compara com `rag/eval/baseline.json`; `--salvar` grava um novo baseline).

> **Limite conhecido:** ruído e documento relevante ainda se sobrepõem perto do corte. O corte reduz o ruído, mas a trava principal é o prompt. A base cobre doenças infecciosas e parasitárias: quadros como pica, corpo estranho ou doença renal não têm ficha compatível.

Falha no RAG (índice ausente, modelo não baixado, erro na busca) **não derruba a análise**: o motor registra o aviso no log e segue com Oracle + web.

| Variável | Padrão | Papel |
|---|---|---|
| `RAG_TOP_K` | `3` | Quantos documentos vão ao LLM |
| `RAG_MAX_CHARS` | `4500` | Tamanho máximo do bloco RAG no prompt |
| `RAG_MIN_SCORE` | `0.45` | Similaridade mínima (0–1) para um documento ir ao LLM (calibrado para o modelo atual) |
| `HF_TOKEN` | — | Token Read do HF para baixar o modelo privado (`--download`) |
| `RAG_INDEX_DIR` | `rag/index` | Pasta do índice |
| `RAG_MODEL_DIR` | `rag/model_cache` | Cache do modelo de embedding |

---

## Cálculo de Confiança (pc_confianca)

O grau de confiança é calculado **deterministicamente em Python** com base nos dados reais do Oracle, antes de chamar a LLM. O modelo recebe o valor pronto e apenas o utiliza — nunca recalcula. Esse mesmo valor decide se a busca web é acionada (ver seção anterior).

A rubrica de "sintomas" avalia a **narrativa clínica** — `DS_TRANSCRICAO` (relato bruto do veterinário) concatenado a `DS_SINTOMAS` (campo estruturado), quando ambos existirem — nunca só o campo estruturado isoladamente. `DS_TRANSCRICAO` nunca é copiado para `DS_SINTOMAS`/`DS_OBSERVACAO`: a combinação existe apenas como variável de execução usada pela rubrica, pela decisão de busca web e pela query de busca.

| Critério | Pontuação |
|----------|-----------|
| BASE (sempre) | +30 pts |
| Narrativa clínica específica e detalhada (> 3 características) | +25 pts |
| Narrativa clínica moderadamente descritiva (1–3 características) | +10 pts |
| Predisposição genética diretamente relacionada aos sintomas (nome da doença ou palavras-chave `DS_SINTOMAS`) | +20 pts |
| Predisposição genética presente mas indiretamente relacionada | +10 pts |
| Avaliação de bem-estar completa e coerente | +10 pts |
| Peso registrado e compatível | +5 pts |
| Dados clínicos relevantes ausentes (peso, idade — `DT_NASCIMENTO` ou `NR_IDADE` — ou bem-estar) | -10 pts |
| Narrativa clínica vaga ou genérica demais | -15 pts |

> **Predisposição racial ≠ evidência principal.** O system prompt (`prompts/diagnostic.py`) só permite tratar uma predisposição genética mapeada como diferencial prioritário quando há sinal clínico compatível na narrativa (transcrição e/ou sintomas). Sem sinal clínico compatível, a predisposição entra apenas como fator de risco de fundo em `insight_predisposicao` — nunca como base de `ds_diagnostico`.

> **Medicações e cuidado preventivo não entram na rubrica.** As seções de prescrições e de status preventivo são injetadas no resumo clínico e alimentam o **raciocínio da LLM** (`insight_correlacao`, `insight_limitacoes`), mas **não** alteram o `pc_confianca` — a rubrica mede qualidade de sintoma, predisposição e bem-estar, e é mantida estável para permanecer auditável.

---

## Modelos Groq — Fallback e Retry

O motor tenta os modelos na ordem `GROQ_MODEL_PRIMARY` → `GROQ_MODEL_FALLBACKS`. Erros de cota/indisponibilidade (HTTP 429, modelo descontinuado) pulam direto para o próximo modelo; erros transitórios (timeout, conexão, 5xx) são retentados no mesmo modelo antes de trocar.

| Variável | Padrão | Papel |
|---|---|---|
| `GROQ_MODEL_PRIMARY` | `openai/gpt-oss-120b` | Modelo tentado primeiro |
| `GROQ_MODEL_FALLBACKS` | `qwen/qwen3.8-27b,openai/gpt-oss-20b` | Modelos seguintes, em ordem |
| `GROQ_REASONING_EFFORT` | `low` | Esforço de raciocínio dos gpt-oss (`low`/`medium`/`high`); ignorado nos outros modelos |
| `GROQ_MAX_RETRIES_PER_MODEL` | `2` | Tentativas no mesmo modelo para erro transitório |
| `GROQ_RETRY_BACKOFF_SECONDS` | `2` | Espera entre tentativas, crescente (2s, 4s, ...) |

Cada execução recomeça pelo modelo primário — não há memória de qual modelo foi usado na execução anterior.

---

## Custo por Análise (Groq)

Cada análise faz uma chamada ao LLM com cerca de **6,3 mil tokens** (medido no painel do Groq com raciocínio `medium`: system prompt, resumo clínico, bloco RAG e resposta). O consumo real de cada análise volta no campo `tokens` do JSON e aparece no log; `GROQ_REASONING_EFFORT=low` reduz a parte do raciocínio. No plano gratuito, o `openai/gpt-oss-120b` tem 8 mil tokens por minuto e 200 mil por dia, o que dá cerca de **1 análise por minuto e 30 por dia**. Acima disso, o fallback troca de modelo.

`RAG_MAX_CHARS` e `RAG_TOP_K` mexem direto nesse custo: cada 1.000 caracteres a mais no bloco RAG somam cerca de 250 tokens.

---

## Schema de Saída (Pydantic v2)

`DiagnosticoOutputDetalhado`, mapeado para gravação futura na tabela `TB_ARKIVE_DIAGNOSTICO` pelo serviço Java:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `ds_diagnostico` | `str` (5–500 chars) | Título conciso da hipótese diagnóstica |
| `tp_severidade` | `Literal["LEVE", "MODERADA", "GRAVE"]` | Classificação de severidade |
| `insight_perfil` | `str` (mín. 20 chars) | Perfil do paciente e apresentação clínica |
| `insight_correlacao` | `str` (mín. 20 chars) | Correlação entre sintomas, bem-estar, histórico e hipótese |
| `insight_predisposicao` | `str` (mín. 20 chars) | Papel das predisposições genéticas no raciocínio |
| `insight_limitacoes` | `str` (mín. 20 chars) | Limitações do diagnóstico e exames complementares sugeridos |
| `pc_confianca` | `int` (0–100) | Grau de certeza calculado deterministicamente em Python |
| `fontes_rag` | `list[{fonte, titulo}]` | Documentos do RAG enviados ao LLM (organização + documento). Fora do schema do LLM: o sistema adiciona ao JSON |
| `fontes_pesquisadas` | `list[str]` | URLs consultadas (lista vazia se busca web não foi acionada) |

O payload final também inclui `ds_insight_ia`, composto a partir dos 4 campos de insight acima, para compatibilidade com a coluna única `DS_INSIGHT_IA` (CLOB) do banco. Inclui também `tokens` (`entrada`, `saida`, `total`), o consumo da chamada ao Groq, preenchido pelo sistema e não persistido.

---

## Garantias de Segurança (READ-ONLY)

O microsserviço garante a imutabilidade do banco em três camadas:

1. **Privilégios DB:** o usuário Oracle deve ter apenas `GRANT SELECT` nas tabelas ArkIve (enforçado pelo DBA);
2. **`autocommit = False`:** configurado explicitamente na conexão;
3. **`rollback()` no finally:** desfaz qualquer transação pendente acidental antes de fechar a conexão.

Nenhum `INSERT`, `UPDATE`, `DELETE` ou `MERGE` existe em qualquer arquivo do projeto.

---

## Solução de Problemas Comuns

| Erro | Causa | Solução |
|------|-------|---------|
| `ORA-12505` | SID não reconhecido | Usar o DSN no formato longo: `(DESCRIPTION=(ADDRESS=...)(CONNECT_DATA=(SID=ORCL)))` |
| `ORA-01017` | Usuário/senha incorretos | Verificar `ORACLE_USER` e `ORACLE_PASSWORD` no `.env` |
| `429 quota exceeded` | Cota diária de um modelo atingida | O sistema troca automaticamente para o próximo modelo em `GROQ_MODEL_FALLBACKS`; se todos falharem, aguardar reset ou criar nova API key |
| `Erro de configuração` ao subir `api.py`/`main.py` | Variável obrigatória ausente ou malformada no `.env` | Ler a mensagem impressa (lista exatamente o que falhou) e corrigir o `.env` |
| `ModuleNotFoundError` | Dependência não instalada | Rodar `pip install -r requirements.txt` com o venv ativo |
| `Nenhuma consulta encontrada` | ID inexistente no banco | Verificar se o ID existe em `TB_ARKIVE_CONSULTA` |
| `RAG indisponível (...)` no log | Índice ausente em `rag/index/`, modelo não baixado ou `faiss` bloqueado pelo Windows | Rodar `python -m rag.retriever --download`; no Windows com Smart App Control, testar no Linux/WSL ou Render |
| `Ratelimit` no DuckDuckGo | Muitas buscas em sequência | O sistema continua sem contexto web; aguarde alguns segundos entre execuções |
| `ddgs sem resultado` no log do Render | PubMed voltou vazio e Google (403) / DuckDuckGo (202) bloqueiam IP de datacenter | O sistema continua sem contexto web; o RAG segue valendo |

---

## Dependências

```
oracledb>=2.3.0,<3.0.0
langchain-core>=0.3.0,<0.4.0
langchain-groq>=0.2.0,<1.0.0
ddgs>=0.1.0
onnxruntime==1.30.0
tokenizers==0.23.2
huggingface_hub==1.33.0
faiss-cpu==1.15.1
pydantic>=2.7.0,<3.0.0
python-dotenv>=1.0.0,<2.0.0
fastapi>=0.110.0
uvicorn>=0.27.0
```

---

## Licença

Projeto acadêmico desenvolvido para o Challenge FIAP 2026 em parceria com a Clyvo Vet.
Uso restrito ao contexto educacional.