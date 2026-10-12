"""
Motor de Inteligência Clínica Veterinária ArkIve.
Pipeline: Oracle (READ-ONLY) → pc_confianca determinístico → RAG local →
busca web se pc_confianca < AMBIGUITY_THRESHOLD → Groq com retry e fallback.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import (
    AMBIGUITY_THRESHOLD,
    GROQ_API_KEY,
    GROQ_MAX_RETRIES_PER_MODEL,
    GROQ_MODELS,
    GROQ_REASONING_EFFORT,
    GROQ_RETRY_BACKOFF_SECONDS,
    GROQ_TEMPERATURE,
    RAG_INDEX_DIR,
    RAG_MAX_CHARS,
    RAG_MIN_SCORE,
    RAG_TOP_K,
)
from database.connection import get_connection
from database.queries import ClinicalContext, fetch_clinical_data
from prompts.diagnostic import DIAGNOSTIC_SYSTEM_PROMPT
from schemas.diagnostic_detalhado import DiagnosticoOutputDetalhado

logger = logging.getLogger(__name__)

#: Única lista de fontes aceitas na busca web; o resto é descartado. Sem meta-buscadores,
#: landing pages de bases pagas nem domínio genérico de universidade/governo.
#: Match por sufixo de host: um domínio-raiz cobre os subdomínios.
TRUSTED_VET_DOMAINS: tuple[str, ...] = (
    # Bases indexadas / bibliográficas (nih.gov cobre pubmed/pmc/ncbi)
    "nih.gov", "europepmc.org", "cabdirect.org",
    "doaj.org", "scielo.br", "scielo.org", "bvsalud.org", "vet.bvs.br",
    # Publishers acadêmicos / periódicos peer-reviewed
    "onlinelibrary.wiley.com", "sciencedirect.com", "link.springer.com",
    "springeropen.com", "tandfonline.com", "journals.sagepub.com",
    "cambridge.org", "journals.plos.org", "frontiersin.org", "mdpi.com",
    "nature.com", "karger.com", "biomedcentral.com", "bmj.com",
    # Periódicos / portais científicos brasileiros
    "journals.usp.br", "revistas.usp.br", "seer.ufu.br", "seer.ufrgs.br",
    "periodicos.uff.br", "portalseer.ufba.br", "pubvet.com.br", "rbmv.org",
    # Manuais, diretrizes e referência clínica
    "merckvetmanual.com", "msdvetmanual.com", "vin.com", "ivis.org",
    "vetstream.com", "plumbs.com", "iris-kidney.com", "cliniciansbrief.com",
    "todaysveterinarypractice.com", "todaysveterinarynurse.com",
    "vetfolio.com", "dvm360.com", "vetsmart.com.br",
    # Toxicologia e parasitologia
    "aspca.org", "aspcapro.org", "petpoisonhelpline.com",
    "capcvet.org", "esccap.org", "troccap.com",
    # Associações e federações
    "wsava.org", "avma.org", "aaha.org", "fecava.org", "bsava.com",
    "bva.co.uk", "catvets.com", "icatcare.org", "abcd-vets.org",
    "aaep.org", "aabp.org", "anclivepabrasil.com.br", "abrovet.org.br",
    "abraveq.com.br", "sbcve.org.br",
    # Colleges de especialidade (board-certified)
    "acvim.org", "acvs.org", "acvp.org", "acvecc.org", "acvo.org",
    "acvd.org", "acvr.org", "acvb.org", "acvaa.org", "acvn.org",
    "ecvim-ca.org", "ecvs.org",
    # Conselhos e reguladores
    "cfmv.gov.br", "crmvsp.gov.br", "rcvs.org.uk", "fve.org",
    "canadianveterinarians.net",
    # Autoridades sanitárias / One Health (usda.gov cobre aphis)
    "woah.org", "fao.org", "who.int", "cdc.gov", "efsa.europa.eu",
    "usda.gov", "embrapa.br",
    # Escolas de veterinária (recursos clínicos / pesquisa)
    "vet.cornell.edu", "vetmed.ucdavis.edu", "vetmed.tamu.edu",
    "vet.upenn.edu", "vetmed.wsu.edu", "vetmed.iastate.edu", "vet.osu.edu",
    "cvm.ncsu.edu", "vetmed.ufl.edu", "vet.tufts.edu", "vetmed.wisc.edu",
    "vetmedbiosci.colostate.edu", "rvc.ac.uk", "ovc.uoguelph.ca",
    "vetmeduni.ac.at", "fmv.ulisboa.pt", "fmvz.usp.br", "fmvz.unesp.br",
    "vet.ufmg.br", "dvt.ufv.br",
)

#: Onde procurar primeiro (via `site:`); o filtro continua sendo TRUSTED_VET_DOMAINS.
PRIORITY_VET_DOMAINS: tuple[str, ...] = (
    "merckvetmanual.com", "msdvetmanual.com", "nih.gov", "avma.org",
    "cfmv.gov.br", "scielo.br", "vin.com", "cliniciansbrief.com",
)


def _is_trusted_source(url: str) -> bool:
    """True se o host é um domínio de TRUSTED_VET_DOMAINS ou subdomínio dele.
    Barra spoofing por sufixo ('nih.gov.exemplo.com')."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if host.startswith("www."):
        host = host[4:]
    return any(host == d or host.endswith("." + d) for d in TRUSTED_VET_DOMAINS)


@dataclass
class _WebSearchDecision:
    needs_web_search: bool
    reason: str  # só para log
    search_query: str


class ClinicalIntelligenceEngine:
    """Uso: ClinicalIntelligenceEngine().analyze(id_consulta=42) -> dict"""

    def __init__(self) -> None:
        logger.info(
            "Inicializando ClinicalIntelligenceEngine | Modelo primário: %s | "
            "Fallbacks (em ordem): %s",
            GROQ_MODELS[0],
            GROQ_MODELS[1:],
        )
        self._chains: dict[str, Any] = {}  # um ChatGroq por modelo, criado no primeiro uso

        # Opcional: sem índice, modelo ou faiss, o motor segue só com Oracle + web.
        try:
            from rag.retriever import RagRetriever
            self._rag = RagRetriever(RAG_INDEX_DIR)
            logger.info("RAG carregado | %d chunks | %s", len(self._rag.chunks), RAG_INDEX_DIR)
        except Exception as exc:
            logger.warning("RAG indisponível (%s) — seguindo sem base de conhecimento local.", exc)
            self._rag = None

    def _get_chain(self, model: str) -> Any:
        """method="json_schema": no modo function_calling os gpt-oss da Groq emitem uma
        pseudo-tool "json" e dão 400 "tool 'json' which was not in request.tools"."""
        if model not in self._chains:
            logger.debug("Criando client ChatGroq para o modelo '%s' (primeiro uso).", model)
            extra = {"reasoning_effort": GROQ_REASONING_EFFORT} if "gpt-oss" in model else {}
            llm = ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=GROQ_TEMPERATURE, **extra)
            # include_raw: usage_metadata para o log de tokens; o erro de parse vem no dict, não levanta
            self._chains[model] = llm.with_structured_output(
                DiagnosticoOutputDetalhado,
                method="json_schema",
                include_raw=True,
            )
        return self._chains[model]

    def analyze(self, id_consulta: int) -> dict[str, Any]:
        """
        Raises:
            ValueError: Consulta não encontrada no banco.
            RuntimeError: Falha ao gerar diagnóstico no Groq (todos os
                modelos da lista falharam).
            oracledb.DatabaseError: Falha de acesso ao Oracle.
        """
        logger.info("── Iniciando análise clínica | ID_CONSULTA=%d ──", id_consulta)

        ctx: ClinicalContext = self._fetch_oracle_data(id_consulta)
        clinical_summary: str = ctx.to_clinical_summary()

        # transcrição + sintomas, só em memória: nunca gravada em ctx.ds_sintomas
        narrativa_clinica = " ".join(
            p for p in ((ctx.ds_transcricao or "").strip(), (ctx.ds_sintomas or "").strip()) if p
        )
        confianca_calculada = _calculate_confidence(ctx, narrativa_clinica)
        logger.info("Confiança calculada deterministicamente: %d%%", confianca_calculada)

        # RAG roda sempre que há relato; falha não derruba a análise
        rag_context, rag_docs = "", []
        texto_rag = f"{(ctx.ds_motivo or '').strip()} {narrativa_clinica}".strip()
        if self._rag and texto_rag:
            try:
                rag_context, rag_docs = self._rag.contexto(
                    texto_rag, ctx.nm_especie, RAG_TOP_K, RAG_MAX_CHARS, RAG_MIN_SCORE)
                if rag_docs:
                    logger.info("RAG | documentos: %s", rag_docs)
                else:
                    logger.info("RAG | nenhum documento acima de %.2f", RAG_MIN_SCORE)
            except Exception as exc:
                logger.warning("Busca RAG falhou: %s. Prosseguindo sem base de conhecimento local.", exc)

        web_decision = _decide_web_search(ctx, confianca_calculada, narrativa_clinica)
        logger.info(
            "Decisão de busca web | pc_confianca=%d%% | limiar=%d%% | busca_web=%s | motivo: %s",
            confianca_calculada,
            AMBIGUITY_THRESHOLD,
            web_decision.needs_web_search,
            web_decision.reason,
        )

        web_context: str = ""
        sources: list[str] = []

        if web_decision.needs_web_search:
            logger.info("Acionando busca web | Query: '%s'", web_decision.search_query)
            web_context, sources = self._perform_web_search(web_decision.search_query)
            logger.info("%d fonte(s) recuperada(s) do DuckDuckGo.", len(sources))
        else:
            logger.info("Dados locais suficientes — busca web não acionada.")

        logger.info("Enviando requisição ao Groq...")
        diagnostic, tokens = self._generate_diagnostic(
            clinical_summary=clinical_summary,
            web_context=web_context,
            sources=sources,
            ctx=ctx,
            confianca_calculada=confianca_calculada,
            rag_context=rag_context,
        )
        # Determinístico, como pc_confianca: o sistema sabe o que foi enviado ao LLM.
        diagnostic.fontes_rag = rag_docs

        logger.info(
            "Diagnóstico gerado | '%s' | Severidade: %s | Confiança: %d%%",
            diagnostic.ds_diagnostico,
            diagnostic.tp_severidade,
            diagnostic.pc_confianca,
        )

        # ds_insight_ia: os 4 insights juntos, para a coluna única DS_INSIGHT_IA do serviço Java
        result = diagnostic.model_dump()
        result["ds_insight_ia"] = diagnostic.to_ds_insight_ia()
        # fora do schema: o LLM não preenche, e o serviço Java não persiste
        result["tokens"] = tokens
        return result

    def _fetch_oracle_data(self, id_consulta: int) -> ClinicalContext:
        logger.info("Conectando ao Oracle (Thin mode)...")
        with get_connection() as conn:
            ctx = fetch_clinical_data(conn, id_consulta)
        logger.info(
            "Dados extraídos | Animal: %s | Espécie: %s | Raça: %s | Predisposições: %d",
            ctx.nm_animal,
            ctx.nm_especie,
            ctx.nm_raca or "SRD/Não informada",
            len(ctx.predisposicoes),
        )
        return ctx

    def _perform_web_search(self, query: str) -> tuple[str, list[str]]:
        """(texto_contexto, urls) só de fontes confiáveis."""
        try:
            from ddgs import DDGS
        except ImportError:
            logger.error("ddgs não instalado: pip install ddgs")
            return "", []

        site_clause = " OR ".join(f"site:{d}" for d in PRIORITY_VET_DOMAINS)
        query_restrita = f"({site_clause}) {query}"

        try:
            time.sleep(2)  # rate limit do ddgs em execuções seguidas
            with DDGS() as ddgs:
                # Restrita a PRIORITY_VET_DOMAINS primeiro; site: no ddgs é frouxo, então cai na ampla.
                # Só google/duckduckgo: os demais backends não trazem domínios da whitelist ou dão 429.
                results = list(ddgs.text(
                    query_restrita,
                    max_results=20,
                    safesearch="moderate",
                    backend="google,duckduckgo",
                ))
                if not any(_is_trusted_source(r.get("href", "")) for r in results):
                    logger.info("Busca restrita a PRIORITY_VET_DOMAINS não retornou fonte confiável — tentando busca ampla.")
                    time.sleep(2)
                    results = list(ddgs.text(
                        query,
                        max_results=20,
                        safesearch="moderate",
                        backend="google,duckduckgo",
                    ))

            snippets: list[str] = []
            urls: list[str] = []
            descartados = 0
            for result in results:
                href = result.get("href", "")
                if not href:
                    continue
                if not _is_trusted_source(href):
                    descartados += 1
                    continue
                snippets.append(
                    f"📄 {result.get('title', '')}\n"
                    f"{result.get('body', '')}\n"
                    f"Fonte: {href}"
                )
                urls.append(href)
                if len(urls) >= 5:
                    break

            logger.info(
                "%d de %d resultado(s) descartado(s) por não estar em fonte veterinária confiável.",
                descartados,
                len(results),
            )

            if not urls:
                logger.info("Nenhuma fonte veterinária confiável na busca — seguindo sem contexto web.")
                return "", []

            web_context = "\n\n" + ("─" * 60) + "\n\n".join(snippets)
        except Exception as exc:
            logger.warning("Busca web falhou: %s. Prosseguindo sem contexto externo.", exc)
            web_context = ""
            urls = []

        return web_context, urls

    def _generate_diagnostic(
        self,
        clinical_summary: str,
        web_context: str,
        sources: list[str],
        ctx: ClinicalContext,
        confianca_calculada: int,
        rag_context: str = "",
    ) -> tuple[DiagnosticoOutputDetalhado, dict[str, Any]]:
        parts = [
            "Analise os seguintes dados clínicos veterinários e gere o "
            "diagnóstico estruturado:\n\n",
            clinical_summary,
            f"\n\n>>> VALOR OBRIGATÓRIO: pc_confianca = {confianca_calculada} <<<\n"
            "Este valor foi calculado deterministicamente pelo sistema com base "
            "nos dados clínicos reais. Use EXATAMENTE este número no campo "
            "pc_confianca — não recalcule, não ajuste, não arredonde.\n",
        ]

        if rag_context:
            parts.extend([
                "\n\n" + "═" * 60 + "\n📚 ",
                rag_context,
                "\n" + "═" * 60,
            ])

        if web_context:
            parts.extend([
                "\n\n" + "═" * 60,
                "\n🌐 CONTEXTO ADICIONAL — LITERATURA VETERINÁRIA (BUSCA WEB):\n",
                web_context,
                "\n" + "═" * 60,
                "\nIMPORTANTE: integre as evidências ao raciocínio clínico nos "
                "campos de insight. Não cole URLs em nenhum deles — elas já "
                "estão em fontes_pesquisadas.",
            ])

        if sources:
            parts.append(
                "\nURLs consultadas (para fontes_pesquisadas):\n"
                + "\n".join(f"  {i + 1}. {url}" for i, url in enumerate(sources))
            )

        messages = [
            SystemMessage(content=DIAGNOSTIC_SYSTEM_PROMPT),
            HumanMessage(content="".join(parts)),
        ]

        diagnostic, tokens = self._invoke_with_fallback(messages)

        diagnostic.pc_confianca = confianca_calculada
        if sources and not diagnostic.fontes_pesquisadas:
            diagnostic.fontes_pesquisadas = sources

        return diagnostic, tokens

    def _invoke_with_fallback(self, messages: list[Any]) -> tuple[DiagnosticoOutputDetalhado, dict[str, Any]]:
        """Percorre GROQ_MODELS a partir do primário em toda chamada. Erro transitório: retry
        com backoff no mesmo modelo. Cota/indisponível: próximo modelo direto.
        RuntimeError se todos falharem."""
        ultima_excecao: Exception | None = None

        for model in GROQ_MODELS:
            chain = self._get_chain(model)

            for tentativa in range(1, GROQ_MAX_RETRIES_PER_MODEL + 1):
                try:
                    saida = chain.invoke(messages)
                    if saida["parsing_error"]:
                        raise saida["parsing_error"]
                    uso = getattr(saida["raw"], "usage_metadata", None) or {}
                    tokens = {"entrada": uso.get("input_tokens"), "saida": uso.get("output_tokens"),
                              "total": uso.get("total_tokens")}
                    logger.info(
                        "Diagnóstico gerado usando modelo: %s (tentativa %d/%d) | tokens: %s",
                        model,
                        tentativa,
                        GROQ_MAX_RETRIES_PER_MODEL,
                        tokens,
                    )
                    return saida["parsed"], tokens

                except Exception as exc:
                    ultima_excecao = exc

                    if _is_quota_or_unavailable_error(exc):
                        logger.warning(
                            "Modelo '%s' indisponível ou com cota esgotada (%s) — "
                            "pulando para o próximo modelo da lista.",
                            model,
                            exc,
                        )
                        break

                    if _is_retryable_error(exc) and tentativa < GROQ_MAX_RETRIES_PER_MODEL:
                        backoff = GROQ_RETRY_BACKOFF_SECONDS * tentativa
                        logger.warning(
                            "Erro transitório no modelo '%s' (tentativa %d/%d): %s — "
                            "nova tentativa em %ds.",
                            model,
                            tentativa,
                            GROQ_MAX_RETRIES_PER_MODEL,
                            exc,
                            backoff,
                        )
                        time.sleep(backoff)
                        continue

                    logger.warning(
                        "Modelo '%s' falhou (tentativa %d/%d, sem mais retries neste "
                        "modelo): %s — pulando para o próximo modelo.",
                        model,
                        tentativa,
                        GROQ_MAX_RETRIES_PER_MODEL,
                        exc,
                    )
                    break

        logger.error(
            "Todos os modelos Groq falharam (%s). Última exceção: %s",
            GROQ_MODELS,
            ultima_excecao,
        )
        raise RuntimeError(
            f"O modelo não gerou um diagnóstico estruturado válido após tentar "
            f"{GROQ_MODELS}: {ultima_excecao}"
        ) from ultima_excecao


def _termos_doenca(doenca: dict[str, str]) -> set[str]:
    """Palavras de 4+ letras do nome da doença e de DS_SINTOMAS (TB_ARKIVE_DOENCA)."""
    texto = f"{doenca.get('nm_doenca', '')} {doenca.get('ds_sintomas', '')}"
    return set(re.findall(r"\b\w{4,}\b", texto.lower()))


def _calculate_confidence(ctx: ClinicalContext, sintomas: str) -> int:
    """
    pc_confianca determinístico (0–100); o LLM recebe o valor pronto. Rubrica (base 30):
    +25 Sintomas específicos e detalhados (> 3 palavras relevantes)
    +10 Sintomas moderadamente descritivos (1–3 palavras relevantes)
    +20 Predisposição genética diretamente relacionada aos sintomas
    +10 Predisposição genética presente mas indiretamente relacionada
    +10 Bem-estar completo (apetite + atividade + comportamento)
    +5  Peso registrado
    -10 Dados relevantes ausentes (idade — DT_NASCIMENTO ou NR_IDADE —, peso ou bem-estar)
    -15 Sintomas vagos ou genéricos
    """
    score = 30

    palavras_relevantes = re.findall(r"\b\w{4,}\b", sintomas)
    if len(palavras_relevantes) > 3:
        score += 25
    elif len(palavras_relevantes) >= 1:
        score += 10
    else:
        score -= 15

    if ctx.predisposicoes:
        sintomas_lower = sintomas.lower()
        diretamente_relacionada = any(
            any(termo in sintomas_lower for termo in _termos_doenca(doenca))
            for doenca in ctx.predisposicoes
        )
        score += 20 if diretamente_relacionada else 10

    if ctx.ds_apetite and ctx.ds_atividade and ctx.ds_comportamento:
        score += 10

    if ctx.peso_efetivo_kg:
        score += 5

    dados_ausentes = (
        not ctx.idade_anos
        or not ctx.peso_efetivo_kg
        or (not ctx.ds_apetite and not ctx.ds_atividade and not ctx.ds_comportamento)
    )
    if dados_ausentes:
        score -= 10

    resultado = max(0, min(100, score))

    logger.debug(
        "Rubrica pc_confianca | sintomas=%d palavras | predisposições=%d | "
        "bem-estar=%s | peso=%s | resultado=%d",
        len(palavras_relevantes),
        len(ctx.predisposicoes),
        "completo" if ctx.ds_apetite and ctx.ds_atividade and ctx.ds_comportamento else "parcial/ausente",
        f"{ctx.peso_efetivo_kg}kg" if ctx.peso_efetivo_kg else "ausente",
        resultado,
    )

    return resultado


def _decide_web_search(ctx: ClinicalContext, pc_confianca: int, sintomas: str) -> _WebSearchDecision:
    """Busca web se pc_confianca < AMBIGUITY_THRESHOLD. As razões servem só para o log."""
    motivo = (ctx.ds_motivo or "").strip()
    reasons: list[str] = []

    if len(sintomas) < 20:
        reasons.append("sintomas ausentes ou insuficientes")
    elif len(re.findall(r"\w+", sintomas)) <= 1:
        reasons.append("sintomas excessivamente genéricos")

    if len(motivo) < 10:
        reasons.append("motivo da consulta não informado")

    if not ctx.predisposicoes:
        especie_str = ctx.nm_especie or "não identificada"
        raca_str = f" / raça '{ctx.nm_raca}'" if ctx.nm_raca else ""
        reasons.append(
            f"nenhuma predisposição genética mapeada no banco para a espécie "
            f"'{especie_str}'{raca_str} — busca web pode enriquecer o diagnóstico"
        )

    if not ctx.ds_apetite and not ctx.ds_atividade and not ctx.ds_comportamento:
        reasons.append("avaliação de bem-estar ausente")

    return _WebSearchDecision(
        needs_web_search=pc_confianca < AMBIGUITY_THRESHOLD,
        reason="; ".join(reasons) if reasons else "dados clínicos suficientes",
        search_query=_build_search_query(ctx, sintomas, motivo),
    )


def _is_quota_or_unavailable_error(exc: Exception) -> bool:
    """Cota (429) ou modelo indisponível/descontinuado: retry no mesmo modelo não ajuda.
    Olha a mensagem também, porque o tipo da exceção muda entre versões das libs."""
    status_code = getattr(exc, "status_code", None)
    if status_code in (404, 429):
        return True

    # sem "unavailable": 503 "service unavailable" é transitório e merece retry
    mensagem = str(exc).lower()
    marcadores = (
        "rate limit",
        "rate_limit",
        "quota",
        "429",
        "decommissioned",
        "does not exist",
        "model_not_found",
    )
    return any(marcador in mensagem for marcador in marcadores)


def _is_retryable_error(exc: Exception) -> bool:
    """Timeout, conexão ou 5xx: vale nova tentativa no mesmo modelo."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return True

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    mensagem = str(exc).lower()
    marcadores = (
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "internal server error",
        "service unavailable",
    )
    return any(marcador in mensagem for marcador in marcadores)


#: Palavras longas de fala transcrita que passariam como "sinal clínico" em _build_search_query.
_FILLER_WORDS: frozenset[str] = frozenset({
    "necessariamente", "normalmente", "encontramos", "geralmente",
    "atualmente", "realmente", "basicamente", "praticamente",
})


def _build_search_query(ctx: ClinicalContext, sintomas: str, motivo: str) -> str:
    """Espécie + raça + 4 palavras-chave dos sintomas + termos de literatura."""
    parts: list[str] = []
    excluir: set[str] = set()

    if ctx.nm_especie:
        parts.append(ctx.nm_especie.lower())
        excluir.update(re.findall(r"\b\w{4,}\b", ctx.nm_especie.lower()))
    if ctx.nm_raca:
        parts.append(ctx.nm_raca.lower())
        excluir.update(re.findall(r"\b\w{4,}\b", ctx.nm_raca.lower()))
    if ctx.nm_animal:
        excluir.update(re.findall(r"\b\w{4,}\b", ctx.nm_animal.lower()))

    texto = sintomas or motivo
    if texto:
        brutos = re.findall(r"\b\w{4,}\b", texto.lower())
        candidatos = list(dict.fromkeys(
            p for p in brutos if p not in excluir and p not in _FILLER_WORDS
        ))
        # termo clínico tende a ser mais longo que preenchimento; heurística, não NER
        candidatos.sort(key=len, reverse=True)
        parts.extend(candidatos[:4])

    parts.append("veterinary clinical diagnosis peer-reviewed")

    return " ".join(parts)[:200]