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

> Assista à apresentação do projeto, explicação da arquitetura e testes (Sprint 1-2):

[![Assistir no YouTube](https://img.shields.io/badge/YouTube-Assistir%20Apresentação-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=zWqLgywXfv4)

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

O **Motor de Inteligência Clínica Veterinária ArkIve** é um microsserviço Python que, a partir do ID de uma consulta veterinária, extrai automaticamente os dados clínicos do banco Oracle (histórico do animal, sintomas, bem-estar, predisposições genéticas da raça), e aciona um modelo de linguagem (LLM) via API para gerar uma **hipótese diagnóstica estruturada** — pronta para ser consumida pelo serviço Java que persiste o resultado no banco.

O sistema opera em **modo estritamente read-only** no banco, nunca escrevendo ou alterando dados. Atende qualquer espécie animal — doméstica, silvestre, zoológica ou de produção.

### Como a solução melhora a jornada

| Para quem | Benefício |
|-----------|-----------|
| **Pet** | Hipótese diagnóstica mais fundamentada, considerando predisposição genética e histórico clínico |
| **Responsável** | Continuidade do cuidado — cada consulta alimenta a inteligência do sistema |
| **Veterinário** | Apoio à decisão clínica em segundos, com raciocínio explicado e grau de confiança |
| **Clínica** | Diferencial competitivo com IA integrada ao prontuário existente |

---

## Tecnologias Utilizadas

| Camada | Tecnologia | Papel no sistema |
|--------|-----------|-----------------|
| Linguagem | Python 3.11+ | Orquestra todas as etapas |
| LLM / IA Generativa | [Groq API](https://console.groq.com) · tentativa em ordem: `openai/gpt-oss-120b` → `qwen/qwen3.6-27b` → `openai/gpt-oss-20b` (`GROQ_MODEL_FALLBACKS`) | Gera o raciocínio clínico e o diagnóstico |
| Integração LLM | `langchain-groq` + `langchain-core` · `ChatGroq.with_structured_output()` | Conecta ao Groq e garante saída JSON validada pelo Pydantic; sem chains ou pipelines LCEL |
| Banco de Dados | Oracle via `oracledb` (modo Thin) | Fonte de dados clínicos — somente leitura |
| Validação de Schema | Pydantic v2 | Valida e tipifica a saída da IA |
| Busca Web (fallback) | `ddgs` (DuckDuckGo Search) | Literatura veterinária complementar |
| API HTTP | `fastapi` + `uvicorn` | Expõe o motor via endpoint REST (`api.py`), como alternativa à CLI (`main.py`) |
| Variáveis de Ambiente | `python-dotenv` | Isola credenciais do código-fonte |

---

## Arquitetura do Sistema

```
Entrada: main.py ──► python main.py <ID_CONSULTA>
Etapa 1 ──► Oracle (READ-ONLY): Extrai animal, espécie, raça, consulta, bem-estar, predisposições genéticas e histórico de diagnósticos anteriores do mesmo animal.
Etapa 2 ──► Python puro (sem LLM): _calculate_confidence() calcula pc_confianca com rubrica fixa baseada nos dados reais do Oracle.
Etapa 3 ──► Python puro (sem LLM): _evaluate_ambiguity_locally() pontua a qualidade dos dados clínicos (0–100).
Se score < AMBIGUITY_THRESHOLD OU pc_confianca < CONFIDENCE_THRESHOLD → busca web geral ativada. Decisão 100% determinística, sem custo de API.
Etapa 4 ──► DuckDuckGo (condicional, geral): Busca literatura veterinária se ambiguidade detectada. Prioriza NCBI/PubMed e Merck Veterinary Manual.
Etapa 4b ─► DuckDuckGo (condicional, dedicada): Se o catálogo local não tem predisposições mapeadas para a espécie/raça, dispara uma busca dedicada a predisposição genética — independente do resultado da Etapa 3.
Etapa 4c ─► Python puro (sem LLM): se qualquer busca (geral ou dedicada) retornou fontes, aplica bônus determinístico (WEB_ENRICHMENT_CONFIDENCE_BONUS) ao pc_confianca, capado em 100.
Etapa 5 ──► Groq API (1 chamada): tenta os modelos em ordem — openai/gpt-oss-120b → qwen/qwen3.6-27b → openai/gpt-oss-20b (GROQ_MODEL_FALLBACKS) — recebe resumo clínico (incluindo histórico anterior) + pc_confianca pronto e gera DiagnosticoOutput validado pelo Pydantic v2.
Saída: JSON ──► {ds_diagnostico, tp_severidade, ds_insight_ia, pc_confianca, fontes_pesquisadas}

A resposta é consumida pela API Java para persistência em TB_ARKIVE_DIAGNOSTICO.
```

### Estrutura de Arquivos

```
arkive_clinical_engine/
├── .env # Variáveis de ambiente
├── requirements.txt # Dependências com versões fixas
├── config.py # Configuração centralizada + validação fail-fast
├── main.py # Ponto de entrada CLI
├── api.py # API HTTP (FastAPI) — expõe o motor via endpoint REST
├── agents/
│ └── clinical_agent.py # Motor principal (LangChain + Groq + heurística + cache)
├── cache/
│ └── ttl_cache.py # Cache TTL+LRU em memória para buscas web (DuckDuckGo)
├── database/
│ ├── connection.py # Conexão Oracle Thin mode, READ-ONLY
│ └── queries.py # SQLs parametrizados + dataclass ClinicalContext
├── prompts/
│ └── diagnostic_v3.py # System prompt enviado ao Groq
├── schemas/
│ └── diagnostic.py # Pydantic v2: DiagnosticoOutput + AmbiguityAssessment
└── tests/
  ├── test_clinical_agent.py # Rubrica de confiança, ambiguidade, cache, integração
  ├── test_diagnostic_history_summary.py # Renderização do histórico de diagnósticos
  └── test_ttl_cache.py # Cache TTL+LRU
```

---

## Como Executar (How To)

### Pré-requisitos

- Python 3.11 ou superior
- Acesso ao banco Oracle da FIAP (`oracle.fiap.com.br`)
- Conta gratuita no [Groq Console](https://console.groq.com) para obter a API Key

### 1. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/arkive_clinical_engine.git
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

### 4. Configurar as variáveis de ambiente

Copie o exemplo e preencha com suas credenciais:

Edite o `.env`:

```env
# Banco Oracle FIAP
ORACLE_DSN=oracle.fiap.com.br:1521/ORCL
ORACLE_USER=seu_usuario_fiap
ORACLE_PASSWORD=sua_senha_fiap

# Groq (obtenha em https://console.groq.com → API Keys)
GROQ_API_KEY=gsk_sua_chave_aqui

# Configurações opcionais
LOG_LEVEL=INFO
AMBIGUITY_THRESHOLD=60
CONFIDENCE_THRESHOLD=70
WEB_SEARCH_CACHE_TTL_SECONDS=3600
WEB_SEARCH_CACHE_MAX_SIZE=200
WEB_ENRICHMENT_CONFIDENCE_BONUS=12
DIAGNOSTIC_HISTORY_LIMIT=5
```

> **Nota:** `GROQ_TEMPERATURE` (usado nas chamadas ao modelo) é fixado em `0.10` diretamente em `config.py` e **não** é configurável via `.env`.

### 5. Executar

```bash
python main.py <ID_CONSULTA>
```

**Exemplo:**

```bash
python main.py 1
```

### 6. Saída esperada

O JSON de saída reflete o schema DiagnosticoOutput (Pydantic v2): quatro campos de raciocínio clínico estruturado (insight_perfil, insight_correlacao, insight_predisposicao, insight_limitacoes) e um campo calculado (ds_insight_ia) que os concatena automaticamente para persistência no Oracle.

```json
{
  "ds_diagnostico": "Suspeita de traqueobronquite infecciosa (tosse dos canis)",
  "tp_severidade": "MODERADA",

  "insight_perfil": "Luna é uma fêmea de Poodle médio, 1 ano de idade e castrada, apresentando tosse seca há cinco dias que evoluiu para tosse produtiva noturna...",
  "insight_correlacao": "A tosse noturna e produtiva é típica de irritação das vias aéreas superiores, como ocorre na traqueobronquite infecciosa...",
  "insight_predisposicao": "Não foram identificadas predisposições genéticas específicas para doenças respiratórias em Poodles nas bases locais ou nas buscas dedicadas...",
  "insight_limitacoes": "A avaliação remota impede a ausculta pulmonar e a inspeção física detalhada, limitando a confirmação clínica...",

  "pc_confianca": 77,

  "fontes_pesquisadas": [
    "https://pubmed.ncbi.nlm.nih.gov/37798816/",
    "https://www.merckvetmanual.com/dog-owners/lung-and-airway-disorders-of-dogs/kennel-cough-infectious-tracheobronchitis-in-dogs"
  ],

  "ds_insight_ia": "Luna é uma fêmea de Poodle médio... [4 parágrafos concatenados automaticamente pelo computed_field]"
}
```

> **Nota:** ds_insight_ia é um computed_field — não é definido pela IA diretamente, mas gerado a partir da concatenação dos 4 campos insight_* acima, separados por linha em branco. Esse é o único campo que o serviço Java efetivamente grava em TB_ARKIVE_DIAGNOSTICO.DS_INSIGHT_IA; os campos insight_* individuais existem apenas para facilitar exibição segmentada (ex: abas/accordion) por consumidores futuros.

Logs de execução são exibidos no terminal. Em caso de erro, o JSON de saída conterá o campo `"error"` com a causa.

---

## Executando via API (FastAPI)

Além da CLI, o motor pode ser exposto como um serviço HTTP através do `api.py`.

### Subir o servidor

```bash
uvicorn api:app --reload
```

Por padrão, a API sobe em `http://127.0.0.1:8000`.

### Endpoint disponível

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/diagnostico/{id_consulta}` | Executa o pipeline completo de análise clínica para o `ID_CONSULTA` informado e retorna o `DiagnosticoOutput` em JSON |

**Exemplo:**

```bash
curl http://127.0.0.1:8000/diagnostico/1
```

**Códigos de resposta:**

| Status | Situação |
|--------|----------|
| `200` | Diagnóstico gerado com sucesso |
| `400` | `id_consulta` inválido (não positivo) |
| `404` | Consulta não encontrada no Oracle |
| `502` | Falha no motor de IA (Groq) |
| `500` | Erro inesperado |

> **CORS:** a API está configurada com `allow_origins=["*"]`, permitindo chamadas de qualquer origem (útil para testes com front-ends web/HTML/JS). Ajuste essa configuração antes de um deploy em produção.

> **Atenção:** se as variáveis de ambiente obrigatórias (`ORACLE_USER`, `ORACLE_PASSWORD`, `GROQ_API_KEY`) estiverem ausentes, a mensagem de erro é apenas impressa no console durante a inicialização — a aplicação só falhará de fato na primeira requisição, quando o `ClinicalIntelligenceEngine` tentar inicializar.

---

## Fluxo de Decisão — Busca Web

O sistema avalia a qualidade dos dados clínicos localmente **antes** de acionar a LLM, usando uma heurística sem custo de API. A decisão de buscar literatura veterinária é baseada exclusivamente nos dados reais do Oracle — sem regras hardcoded por espécie:

| Critério | Penalidade no Score |
|----------|-------------------|
| Sintomas ausentes ou < 20 caracteres | -35 pts |
| Sintomas genéricos (1 palavra) | -15 pts |
| Motivo da consulta ausente ou < 10 caracteres | -20 pts |
| Sem predisposições genéticas mapeadas no banco para a espécie/raça | -20 pts |
| Avaliação de bem-estar ausente | -10 pts |

Se o score ficar abaixo do `AMBIGUITY_THRESHOLD` (padrão: 60%), o DuckDuckGo é acionado para buscar literatura veterinária atualizada no NCBI/PubMed e Merck Veterinary Manual, enriquecendo o contexto antes da chamada à LLM.

**Resultado: NO MÁXIMO 1 chamada à API do Groq por execução.**

> **Dica para testes:** defina `AMBIGUITY_THRESHOLD=101` no `.env` para forçar a busca web em todas as execuções, independente da qualidade dos dados locais.

> **Nota — exclusão lógica no catálogo de predisposições:** a query de predisposições genéticas (`PREDISPOSITION_QUERY`) só considera doenças com `TB_ARKIVE_DOENCA.ST_ATIVO = 'S'`. Como `TB_ARKIVE_PREDISPOSICAO` não possui coluna própria de `ST_ATIVO`, uma doença excluída logicamente (`ST_ATIVO = 'N'`) nunca é retornada como predisposição, mesmo que o vínculo N:N ainda exista na tabela de predisposição.

---

## Cálculo de Confiança (pc_confianca)

O grau de confiança é calculado **deterministicamente em Python** com base nos dados reais do Oracle, antes de chamar a LLM. O modelo recebe o valor pronto e apenas o utiliza — nunca recalcula.

| Critério | Pontuação |
|----------|-----------|
| BASE (sempre) | +30 pts |
| Sintomas específicos e detalhados (> 3 características) | +25 pts |
| Sintomas moderadamente descritivos (1–3 características) | +10 pts |
| Predisposição genética diretamente relacionada aos sintomas | +20 pts |
| Predisposição genética presente mas indiretamente relacionada | +10 pts |
| Avaliação de bem-estar completa e coerente | +10 pts |
| Peso registrado e compatível | +5 pts |
| Dados clínicos relevantes ausentes (peso, idade ou bem-estar) | -10 pts |
| Sintomas vagos ou genéricos demais | -15 pts |

**Correlação sintomas × predisposição:** a decisão entre "diretamente relacionada" (+20) e "indiretamente relacionada" (+10) compara os sintomas relatados na consulta atual contra **duas fontes** do catálogo de doenças (`TB_ARKIVE_DOENCA`): o nome da doença (`NM_DOENCA`) **e** as palavras-chave clínicas cadastradas em `DS_SINTOMAS`. Isso torna a correlação mais precisa em casos onde o nome da doença não compartilha vocabulário com os sintomas relatados, mas as palavras-chave cadastradas sim (ex: doença "Displasia Coxofemoral" com `DS_SINTOMAS = 'claudicação dor articular dificuldade locomoção'`, correlacionando corretamente com sintomas de claudicação). Depende do preenchimento de `DS_SINTOMAS` no catálogo — enquanto essa coluna estiver vazia para uma doença, a correlação cai para o comportamento anterior (apenas via `NM_DOENCA`).

**`CONFIDENCE_THRESHOLD`** (padrão: `70`) — limiar mínimo aceitável para o `pc_confianca` calculado acima. Se o valor ficar abaixo desse limiar, a busca web geral é acionada **mesmo que os dados locais estejam completos** (ex: sintomas específicos e bem-estar registrado, mas ainda assim uma hipótese incerta). Funciona em conjunto com `AMBIGUITY_THRESHOLD`: a busca web dispara se **qualquer um** dos dois critérios indicar necessidade — completude dos dados **ou** confiança na hipótese.

---

## Cache de Busca Web (DuckDuckGo)

Para reduzir latência e o risco de rate limit do DuckDuckGo, `_perform_web_search()` mantém um cache TTL+LRU em memória (`cache/ttl_cache.py`), compartilhado entre chamadas do mesmo processo:

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `WEB_SEARCH_CACHE_TTL_SECONDS` | `3600` (1 hora) | Tempo de vida de uma entrada em cache. Buscas com a mesma query normalizada (`normalize_cache_key()`) dentro dessa janela reaproveitam o resultado anterior, sem nova chamada ao DuckDuckGo. |
| `WEB_SEARCH_CACHE_MAX_SIZE` | `200` | Número máximo de queries distintas mantidas simultaneamente em cache. Acima disso, a entrada menos recentemente usada (LRU) é descartada para abrir espaço. |

> **Limitação conhecida:** o cache é por processo. Em deploy com múltiplos workers (`uvicorn --workers N`), cada worker mantém seu próprio cache — reduz buscas repetidas dentro de cada worker, mas não é compartilhado entre eles. Falhas ou resultados vazios do DuckDuckGo **não** são cacheados, para não perpetuar um erro transitório (rate limit, timeout) em buscas futuras.

---

## Bônus de Confiança por Enriquecimento Web

**`WEB_ENRICHMENT_CONFIDENCE_BONUS`** (padrão: `12`) — pontos percentuais somados ao `pc_confianca` calculado localmente quando **qualquer** busca web (geral e/ou dedicada a predisposição) retorna ao menos uma fonte. A lógica: se o diagnóstico final incorporou evidência de literatura veterinária externa, ele se apoia em mais informação do que os dados do Oracle isoladamente sustentavam — o bônus reflete isso de forma determinística. Sem fontes encontradas, o `pc_confianca` permanece inalterado. O resultado é sempre capado em 100 (`_apply_web_enrichment_bonus()`).

---

## Histórico de Diagnósticos Anteriores

Para continuidade do cuidado, o resumo clínico enviado à LLM inclui os últimos diagnósticos de **outras** consultas do mesmo animal (`DIAGNOSTIC_HISTORY_QUERY`).

**`DIAGNOSTIC_HISTORY_LIMIT`** (padrão: `5`) — número máximo de diagnósticos anteriores incluídos no resumo. Limita o histórico aos mais recentes (`ORDER BY DT_HORA DESC`) para não estourar a janela de contexto do LLM com um histórico muito longo.

Cada entrada do histórico expõe dois sinais de confiabilidade distintos, ambos vindos de `TB_ARKIVE_DIAGNOSTICO`:

| Campo Oracle | Significado | Exibido no resumo como |
|--------------|-------------|-------------------------|
| `ST_CONFIRMADO` | Se o diagnóstico em si foi confirmado (default `'S'`) | "confirmado" / "não confirmado" |
| `ST_VALIDACAO_VET` | Se o veterinário validou o **insight gerado pela IA** para aquele diagnóstico (nullable, sem default) | "validado pelo veterinário" / "não validado pelo veterinário" / "insight de IA não avaliado pelo veterinário" (quando `NULL`) |

O system prompt (`prompts/diagnostic_v3.py`) instrui a LLM a dar **ceticismo redobrado** a diagnósticos anteriores cujo insight de IA não foi validado pelo veterinário — por isso `ST_VALIDACAO_VET` precisa necessariamente chegar ao texto do resumo, e não apenas ser extraído do banco.

---

## Schema de Saída (Pydantic v2)

Mapeado para gravação futura na tabela `TB_ARKIVE_DIAGNOSTICO` pelo serviço Java:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `ds_diagnostico` | `str` (5–500 chars) | Título conciso da hipótese diagnóstica |
| `tp_severidade` | `Literal["LEVE", "MODERADA", "GRAVE"]` | Classificação de severidade |
| `insight_perfil` | `str` (mín. 20 chars) | Perfil do paciente e apresentação clínica principal |
| `insight_correlacao` | `str` (mín. 20 chars) | Correlação entre sintomas, bem-estar e hipótese diagnóstica |
| `insight_predisposicao` | `str` (mín. 20 chars) | Papel das predisposições genéticas no raciocínio clínico |
| `insight_limitacoes` | `str` (mín. 20 chars) | Limitações do diagnóstico e exames complementares sugeridos |
| `ds_insight_ia` | `str` (`computed_field`) | Concatenação automática dos 4 campos `insight_*` acima, separados por linha em branco — é este campo, não os individuais, que é efetivamente gravado em `TB_ARKIVE_DIAGNOSTICO.DS_INSIGHT_IA` |
| `pc_confianca` | `int` (0–100) | Grau de certeza calculado deterministicamente em Python; a IA nunca recalcula |
| `fontes_pesquisadas` | `list[str]` | URLs consultadas (lista vazia se nenhuma busca web foi acionada) |

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
| `429 quota exceeded` | Cota diária da API atingida | A cota do Groq é de ~14.400 req/dia; aguardar reset à meia-noite ou criar nova API key |
| `ModuleNotFoundError` | Dependência não instalada | Rodar `pip install -r requirements.txt` com o venv ativo |
| `Nenhuma consulta encontrada` | ID inexistente no banco | Verificar se o ID existe em `TB_ARKIVE_CONSULTA` |
| `Ratelimit` no DuckDuckGo | Muitas buscas em sequência | O sistema continua sem contexto web; aguarde alguns segundos entre execuções |

---

## Dependências

```
oracledb>=2.3.0,<3.0.0
langchain-core>=0.3.0,<0.4.0
langchain-groq>=0.2.0,<1.0.0
ddgs>=0.1.0
pydantic>=2.7.0,<3.0.0
python-dotenv>=1.0.0,<2.0.0
fastapi>=0.110.0
uvicorn>=0.27.0
```

---

## Licença

Projeto acadêmico desenvolvido para o Challenge FIAP 2026 em parceria com a Clyvo Vet.
Uso restrito ao contexto educacional.