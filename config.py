"""Configuração do motor ArkIve lida do .env, com validação fail-fast em validate_config()."""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)

_LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL: int = getattr(logging, _LOG_LEVEL_STR, logging.INFO)

logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

ORACLE_DSN: str = os.getenv("ORACLE_DSN", "localhost:1521/XEPDB1")
ORACLE_USER: str = os.getenv("ORACLE_USER", "")
ORACLE_PASSWORD: str = os.getenv("ORACLE_PASSWORD", "")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_TEMPERATURE: float = 0.10
GROQ_MODEL_PRIMARY: str = os.getenv("GROQ_MODEL_PRIMARY", "openai/gpt-oss-120b")
_GROQ_MODEL_FALLBACKS_RAW: str = os.getenv(
    "GROQ_MODEL_FALLBACKS", "qwen/qwen3.8-27b,openai/gpt-oss-20b"
)
#: Só nos gpt-oss (low/medium/high). Raciocínio conta no limite de tokens por minuto da Groq.
GROQ_REASONING_EFFORT: str = os.getenv("GROQ_REASONING_EFFORT", "low")
#: Retry no mesmo modelo só para erro transitório; 429 pula direto para o próximo.
_GROQ_MAX_RETRIES_PER_MODEL_RAW: str = os.getenv("GROQ_MAX_RETRIES_PER_MODEL", "2")
#: Backoff linear: valor, valor*2, ...
_GROQ_RETRY_BACKOFF_SECONDS_RAW: str = os.getenv("GROQ_RETRY_BACKOFF_SECONDS", "2")

#: pc_confianca abaixo disto aciona a busca web.
_AMBIGUITY_THRESHOLD_RAW: str = os.getenv("AMBIGUITY_THRESHOLD", "60")

#: Limites de histórico, prescrições/preventivos e transcrição no resumo enviado ao LLM.
_DIAGNOSTIC_HISTORY_LIMIT_RAW: str = os.getenv("DIAGNOSTIC_HISTORY_LIMIT", "5")
_HISTORICO_CUIDADO_LIMIT_RAW: str = os.getenv("HISTORICO_CUIDADO_LIMIT", "5")
_MAX_TRANSCRICAO_CHARS_RAW: str = os.getenv("MAX_TRANSCRICAO_CHARS", "6000")
#: Corte do abstract de cada artigo PubMed (3 artigos) no contexto da busca web.
_PUBMED_ABSTRACT_CHARS_RAW: str = os.getenv("PUBMED_ABSTRACT_CHARS", "700")
#: Predisposições (TB_ARKIVE_DOENCA) no prompt: cada uma custa ~150 tokens, e a tabela só cresce.
_PREDISPOSICAO_LIMIT_RAW: str = os.getenv("PREDISPOSICAO_LIMIT", "5")
_PREDISPOSICAO_DESC_CHARS_RAW: str = os.getenv("PREDISPOSICAO_DESC_CHARS", "400")
#: Teto do ranking semântico das predisposições (embed das doenças novas); estourou, cai no lexical.
_PREDISPOSICAO_RANK_TIMEOUT_S_RAW: str = os.getenv("PREDISPOSICAO_RANK_TIMEOUT_S", "10")

#: Índice de rag/build_index.ipynb. Opcional: sem ele o motor segue sem RAG.
RAG_INDEX_DIR: Path = Path(os.getenv("RAG_INDEX_DIR", str(Path(__file__).parent / "rag" / "index")))
#: Modelo de embedding (~113 MB), baixado no build por `python -m rag.retriever --download`.
RAG_MODEL_DIR: Path = Path(os.getenv("RAG_MODEL_DIR", str(Path(__file__).parent / "rag" / "model_cache")))
#: Orçamento do bloco RAG, dividido igualmente entre os documentos.
_RAG_MAX_CHARS_RAW: str = os.getenv("RAG_MAX_CHARS", "4500")
_RAG_TOP_K_RAW: str = os.getenv("RAG_TOP_K", "3")
#: Calibrado no e5 fine-tunado int8 (rag/finetune_e5.ipynb, célula 12); trocar o modelo exige recalibrar.
_RAG_MIN_SCORE_RAW: str = os.getenv("RAG_MIN_SCORE", "0.45")


def _parse_positive_int(raw: str, var_name: str, default_on_error: int) -> tuple[int, str | None]:
    """(valor, None) ou (default, erro); o erro vira mensagem em validate_config(), não traceback no import."""
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

HISTORICO_CUIDADO_LIMIT, _err = _parse_positive_int(_HISTORICO_CUIDADO_LIMIT_RAW, "HISTORICO_CUIDADO_LIMIT", 5)
if _err:
    _config_errors.append(_err)

MAX_TRANSCRICAO_CHARS, _err = _parse_positive_int(_MAX_TRANSCRICAO_CHARS_RAW, "MAX_TRANSCRICAO_CHARS", 6000)
if _err:
    _config_errors.append(_err)

PUBMED_ABSTRACT_CHARS, _err = _parse_positive_int(_PUBMED_ABSTRACT_CHARS_RAW, "PUBMED_ABSTRACT_CHARS", 700)
if _err:
    _config_errors.append(_err)

PREDISPOSICAO_LIMIT, _err = _parse_positive_int(_PREDISPOSICAO_LIMIT_RAW, "PREDISPOSICAO_LIMIT", 5)
if _err:
    _config_errors.append(_err)

PREDISPOSICAO_DESC_CHARS, _err = _parse_positive_int(_PREDISPOSICAO_DESC_CHARS_RAW, "PREDISPOSICAO_DESC_CHARS", 400)
if _err:
    _config_errors.append(_err)

PREDISPOSICAO_RANK_TIMEOUT_S, _err = _parse_positive_int(
    _PREDISPOSICAO_RANK_TIMEOUT_S_RAW, "PREDISPOSICAO_RANK_TIMEOUT_S", 10
)
if _err:
    _config_errors.append(_err)

RAG_MAX_CHARS, _err = _parse_positive_int(_RAG_MAX_CHARS_RAW, "RAG_MAX_CHARS", 4500)
if _err:
    _config_errors.append(_err)

RAG_TOP_K, _err = _parse_positive_int(_RAG_TOP_K_RAW, "RAG_TOP_K", 3)
if _err:
    _config_errors.append(_err)

try:
    RAG_MIN_SCORE: float = float(_RAG_MIN_SCORE_RAW)
    if not 0 <= RAG_MIN_SCORE <= 1:
        raise ValueError
except ValueError:
    RAG_MIN_SCORE = 0.45
    _config_errors.append(f"RAG_MIN_SCORE deve ser um número entre 0 e 1 (valor recebido: '{_RAG_MIN_SCORE_RAW}').")

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

_GROQ_MODEL_FALLBACKS: list[str] = [
    m.strip() for m in _GROQ_MODEL_FALLBACKS_RAW.split(",") if m.strip()
]

if not GROQ_MODEL_PRIMARY.strip():
    _config_errors.append("GROQ_MODEL_PRIMARY não pode ser vazio.")

#: Ordem de tentativa: primário, depois os fallbacks.
GROQ_MODELS: list[str] = (
    [GROQ_MODEL_PRIMARY.strip()] + _GROQ_MODEL_FALLBACKS if GROQ_MODEL_PRIMARY.strip() else _GROQ_MODEL_FALLBACKS
)


def validate_config() -> None:
    """RuntimeError se faltar credencial ou algum valor do .env estiver malformado."""
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
