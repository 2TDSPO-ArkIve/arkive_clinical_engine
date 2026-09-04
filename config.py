"""
config.py
=========
Centraliza o carregamento e a validação de todas as configurações do
Motor de Inteligência Clínica Veterinária ArkIve.

Lê variáveis do arquivo .env via python-dotenv; falha rapidamente
(fail-fast) se credenciais críticas estiverem ausentes ou malformadas.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Carrega o .env a partir do diretório do projeto (sobe um nível se necessário)
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)

# ── Logging ──────────────────────────────────────────────────────────────────

_LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL: int = getattr(logging, _LOG_LEVEL_STR, logging.INFO)

logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

# ── Oracle Database ───────────────────────────────────────────────────────────

ORACLE_DSN: str = os.getenv("ORACLE_DSN", "localhost:1521/XEPDB1")
ORACLE_USER: str = os.getenv("ORACLE_USER", "")
ORACLE_PASSWORD: str = os.getenv("ORACLE_PASSWORD", "")

# ── GROQ (Free Tier) ─────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_TEMPERATURE: float = 0.10

#: Modelo principal. Mantido como variável de ambiente (e não mais constante
#: fixa) para permitir troca sem alterar código — ver GROQ_MODEL_FALLBACKS
#: para a cadeia de fallback usada quando o modelo primário falha ou estoura
#: a cota de requisições.
GROQ_MODEL_PRIMARY: str = os.getenv("GROQ_MODEL_PRIMARY", "openai/gpt-oss-120b")

#: Lista de modelos de fallback, em ordem de tentativa, separados por vírgula.
#: Usados quando GROQ_MODEL_PRIMARY falha (erro transitório após esgotar as
#: tentativas) ou retorna erro de cota/indisponibilidade (429 e afins).
_GROQ_MODEL_FALLBACKS_RAW: str = os.getenv(
    "GROQ_MODEL_FALLBACKS", "qwen/qwen3.6-27b,openai/gpt-oss-20b"
)

#: Quantas tentativas fazer no MESMO modelo antes de desistir dele por erro
#: transitório (timeout, erro de conexão, 5xx). Erros de cota (429) NÃO usam
#: esse retry — pulam direto para o próximo modelo da lista.
_GROQ_MAX_RETRIES_PER_MODEL_RAW: str = os.getenv("GROQ_MAX_RETRIES_PER_MODEL", "2")

#: Backoff base em segundos entre tentativas no mesmo modelo (exponencial:
#: 1ª espera = valor, 2ª espera = valor*2, ...).
_GROQ_RETRY_BACKOFF_SECONDS_RAW: str = os.getenv("GROQ_RETRY_BACKOFF_SECONDS", "2")

# ── Limiar de Ambiguidade / Confiança ─────────────────────────────────────────

#: Se o pc_confianca calculado deterministicamente for menor que este valor,
#: a busca web é acionada (ver agents/clinical_agent.py).
_AMBIGUITY_THRESHOLD_RAW: str = os.getenv("AMBIGUITY_THRESHOLD", "60")

# ── Histórico Clínico ──────────────────────────────────────────────────────────

#: Quantos diagnósticos anteriores do mesmo animal (de outras consultas)
#: são buscados no Oracle e injetados no resumo clínico enviado à IA.
_DIAGNOSTIC_HISTORY_LIMIT_RAW: str = os.getenv("DIAGNOSTIC_HISTORY_LIMIT", "5")

# ── Transcrição de Voz ──────────────────────────────────────────────────────

#: Tamanho máximo (em caracteres) da transcrição de voz (DS_TRANSCRICAO)
#: exibida no resumo clínico — evita estourar a janela de contexto do LLM
#: com uma consulta muito longa.
_MAX_TRANSCRICAO_CHARS_RAW: str = os.getenv("MAX_TRANSCRICAO_CHARS", "6000")

# ── Parsing seguro de inteiros vindos do .env ─────────────────────────────────
#
# int(os.getenv(...)) direto no nível de módulo derruba o processo com um
# traceback cru se alguém colocar um valor não-numérico no .env. Centra-
# lizamos o parsing aqui para que qualquer erro vire uma mensagem clara em
# validate_config() (fail-fast), em vez de uma exceção não tratada na
# importação do módulo.


def _parse_positive_int(raw: str, var_name: str, default_on_error: int) -> tuple[int, str | None]:
    """
    Tenta converter `raw` para int positivo.

    Retorna (valor, None) em sucesso, ou (default_on_error, mensagem_de_erro)
    em falha — o valor default só existe para permitir que o módulo termine
    de carregar; validate_config() é quem efetivamente barra a execução.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default_on_error, f"{var_name} deve ser um número inteiro (valor recebido: '{raw}')."
    if value <= 0:
        return default_on_error, f"{var_name} deve ser um número inteiro positivo (valor recebido: '{raw}')."
    return value, None


_config_errors: list[str] = []

AMBIGUITY_THRESHOLD, _err = _parse_positive_int(_AMBIGUITY_THRESHOLD_RAW, "AMBIGUITY_THRESHOLD", 60)
if _err:
    _config_errors.append(_err)

DIAGNOSTIC_HISTORY_LIMIT, _err = _parse_positive_int(_DIAGNOSTIC_HISTORY_LIMIT_RAW, "DIAGNOSTIC_HISTORY_LIMIT", 5)
if _err:
    _config_errors.append(_err)

MAX_TRANSCRICAO_CHARS, _err = _parse_positive_int(_MAX_TRANSCRICAO_CHARS_RAW, "MAX_TRANSCRICAO_CHARS", 6000)
if _err:
    _config_errors.append(_err)

GROQ_MAX_RETRIES_PER_MODEL, _err = _parse_positive_int(
    _GROQ_MAX_RETRIES_PER_MODEL_RAW, "GROQ_MAX_RETRIES_PER_MODEL", 2
)
if _err:
    _config_errors.append(_err)

GROQ_RETRY_BACKOFF_SECONDS, _err = _parse_positive_int(
    _GROQ_RETRY_BACKOFF_SECONDS_RAW, "GROQ_RETRY_BACKOFF_SECONDS", 2
)
if _err:
    _config_errors.append(_err)

# ── Montagem da cadeia de modelos Groq ────────────────────────────────────────

_GROQ_MODEL_FALLBACKS: list[str] = [
    m.strip() for m in _GROQ_MODEL_FALLBACKS_RAW.split(",") if m.strip()
]

if not GROQ_MODEL_PRIMARY.strip():
    _config_errors.append("GROQ_MODEL_PRIMARY não pode ser vazio.")

#: Lista final de modelos, na ordem em que devem ser tentados: primário
#: primeiro, depois os fallbacks na ordem declarada em GROQ_MODEL_FALLBACKS.
#: Usada por agents/clinical_agent.py para orquestrar retry + fallback.
GROQ_MODELS: list[str] = (
    [GROQ_MODEL_PRIMARY.strip()] + _GROQ_MODEL_FALLBACKS if GROQ_MODEL_PRIMARY.strip() else _GROQ_MODEL_FALLBACKS
)

# ── Validação Fail-Fast ───────────────────────────────────────────────────────


def validate_config() -> None:
    """
    Valida credenciais obrigatórias e formatos de variáveis numéricas.

    Raises:
        RuntimeError: Se alguma credencial estiver ausente ou algum valor
            numérico do .env estiver malformado.
    """
    missing: list[str] = []
    if not ORACLE_USER:
        missing.append("ORACLE_USER")
    if not ORACLE_PASSWORD:
        missing.append("ORACLE_PASSWORD")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    problems: list[str] = list(_config_errors)
    if missing:
        problems.append(f"Defina as variáveis no .env: {', '.join(missing)}")

    if problems:
        raise RuntimeError(" | ".join(problems))

    logger.info(
        "Configuração validada | Oracle DSN: %s | Modelos Groq (ordem de tentativa): %s | "
        "AMBIGUITY_THRESHOLD=%d | DIAGNOSTIC_HISTORY_LIMIT=%d",
        ORACLE_DSN,
        GROQ_MODELS,
        AMBIGUITY_THRESHOLD,
        DIAGNOSTIC_HISTORY_LIMIT,
    )