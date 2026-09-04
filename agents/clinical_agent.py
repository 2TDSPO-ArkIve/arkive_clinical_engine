"""
agents/clinical_agent.py
========================
Motor de Inteligência Clínica Veterinária ArkIve.
Pipeline: Oracle (READ-ONLY) → pc_confianca determinístico → decisão de
          busca web (baseada no próprio pc_confianca vs AMBIGUITY_THRESHOLD)
          → DuckDuckGo opcional → chamada ao Groq com fallback automático
          entre modelos (GROQ_MODELS) e retry com backoff para erros
          transitórios.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import (
    AMBIGUITY_THRESHOLD,
    GROQ_API_KEY,
    GROQ_MAX_RETRIES_PER_MODEL,
    GROQ_MODELS,
    GROQ_RETRY_BACKOFF_SECONDS,
    GROQ_TEMPERATURE,
)
from database.connection import get_connection
from database.queries import ClinicalContext, fetch_clinical_data
from prompts.diagnostic import DIAGNOSTIC_SYSTEM_PROMPT
from schemas.diagnostic_detalhado import DiagnosticoOutputDetalhado

logger = logging.getLogger(__name__)

# Resultado da decisão de busca web (sem LLM)


@dataclass
class _WebSearchDecision:
    """
    Decisão sobre acionar busca web, derivada do pc_confianca já calculado
    deterministicamente por _calculate_confidence() — fonte única de
    verdade, comparada diretamente contra AMBIGUITY_THRESHOLD.
    """
    needs_web_search: bool
    reason: str         # Descrição legível dos motivos prováveis (para logs)
    search_query: str   # Query sugerida para o DuckDuckGo



class ClinicalIntelligenceEngine:
    """
    Motor de Inteligência Clínica Veterinária com RAG local (Oracle) e
    fallback automático de busca web (DuckDuckGo).

    Uso::

        engine = ClinicalIntelligenceEngine()
        result: dict = engine.analyze(id_consulta=42)
    """

    def __init__(self) -> None:
        """
        Inicializa o motor com fallback automático entre modelos Groq.

        Não constrói mais uma única chain fixa no __init__: os clients
        ChatGroq (um por modelo) são criados sob demanda e cacheados em
        self._chains na primeira vez que cada modelo é efetivamente
        necessário — evita conectar modelos de fallback que talvez nunca
        cheguem a ser usados numa execução sem erros.
        """
        logger.info(
            "Inicializando ClinicalIntelligenceEngine | Modelo primário: %s | "
            "Fallbacks (em ordem): %s",
            GROQ_MODELS[0],
            GROQ_MODELS[1:],
        )
        self._chains: dict[str, Any] = {}

    def _get_chain(self, model: str) -> Any:
        """
        Retorna a chain de diagnóstico estruturado para `model`, criando e
        cacheando o client ChatGroq correspondente na primeira chamada.

        Usa method="json_schema" (response_format nativo da Groq) em vez do
        padrão "function_calling" (tool-calling). Os modelos gpt-oss servidos
        pela Groq têm um bug conhecido no modo function_calling: emitem uma
        pseudo-tool chamada "json" que não bate com o nome da tool registrada
        pelo langchain-groq, causando erro 400 "tool 'json' which was not in
        request.tools" — mesmo quando o raciocínio do modelo está correto.
        json_schema evita esse problema por não depender de tool-calling.
        """
        if model not in self._chains:
            logger.debug("Criando client ChatGroq para o modelo '%s' (primeiro uso).", model)
            llm = ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=GROQ_TEMPERATURE)
            self._chains[model] = llm.with_structured_output(
                DiagnosticoOutputDetalhado,
                method="json_schema",
            )
        return self._chains[model]

    # Método Público Principal

    def analyze(self, id_consulta: int) -> dict[str, Any]:
        """
        Ponto de entrada principal. Faz 1 chamada bem-sucedida ao Groq — mas,
        em caso de erro transitório ou de cota, pode tentar novamente no
        mesmo modelo (retry com backoff) ou trocar de modelo (fallback)
        antes de desistir. Ver _invoke_with_fallback().

        Raises:
            ValueError: Consulta não encontrada no banco.
            RuntimeError: Falha ao gerar diagnóstico no Groq (todos os
                modelos da lista falharam).
            oracledb.DatabaseError: Falha de acesso ao Oracle.
        """
        logger.info("── Iniciando análise clínica | ID_CONSULTA=%d ──", id_consulta)

        # Etapa 1: Extração Oracle
        ctx: ClinicalContext = self._fetch_oracle_data(id_consulta)
        clinical_summary: str = ctx.to_clinical_summary()

        # Etapa 2: Cálculo determinístico do pc_confianca — feito ANTES da
        # decisão de busca web, pois agora é ele a fonte única de verdade
        # usada para decidir se a busca web é necessária (ver Etapa 3).
        sintomas = (ctx.ds_sintomas or "").strip()
        confianca_calculada = _calculate_confidence(ctx, sintomas)
        logger.info("Confiança calculada deterministicamente: %d%%", confianca_calculada)

        # Etapa 3: Decisão de busca web — compara o MESMO pc_confianca
        # calculado acima contra AMBIGUITY_THRESHOLD. Evita manter duas
        # rubricas de score independentes (uma para "ambiguidade" e outra
        # para "confiança") que podiam divergir entre si.
        web_decision = _decide_web_search(ctx, confianca_calculada, sintomas)
        logger.info(
            "Decisão de busca web | pc_confianca=%d%% | limiar=%d%% | busca_web=%s | motivo: %s",
            confianca_calculada,
            AMBIGUITY_THRESHOLD,
            web_decision.needs_web_search,
            web_decision.reason,
        )

        # Etapa 4: Busca web condicional
        web_context: str = ""
        sources: list[str] = []

        if web_decision.needs_web_search:
            logger.info("Acionando busca web | Query: '%s'", web_decision.search_query)
            web_context, sources = self._perform_web_search(web_decision.search_query)
            logger.info("%d fonte(s) recuperada(s) do DuckDuckGo.", len(sources))
        else:
            logger.info("Dados locais suficientes — busca web não acionada.")

        # Etapa 5: Chamada ao Groq (com fallback/retry entre modelos)
        logger.info("Enviando requisição ao Groq...")
        diagnostic: DiagnosticoOutputDetalhado = self._generate_diagnostic(
            clinical_summary=clinical_summary,
            web_context=web_context,
            sources=sources,
            ctx=ctx,
            confianca_calculada=confianca_calculada,
        )

        logger.info(
            "Diagnóstico gerado | '%s' | Severidade: %s | Confiança: %d%%",
            diagnostic.ds_diagnostico,
            diagnostic.tp_severidade,
            diagnostic.pc_confianca,
        )

        # Compõe o payload final: os 4 campos de insight continuam
        # disponíveis individualmente (para quem quiser consumi-los
        # granularmente), e ds_insight_ia é adicionado como a junção deles
        # em um único texto — mantendo compatibilidade com o contrato de
        # coluna única (DS_INSIGHT_IA CLOB) esperado pelo serviço Java.
        result = diagnostic.model_dump()
        result["ds_insight_ia"] = diagnostic.to_ds_insight_ia()
        return result

    # Métodos Privados

    def _fetch_oracle_data(self, id_consulta: int) -> ClinicalContext:
        """Extrai dados clínicos do Oracle usando conexão READ-ONLY em Thin mode."""
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
        """Busca no DuckDuckGo. Retorna (texto_contexto, lista_de_urls)."""
        try:
            from ddgs import DDGS
        except ImportError:
            logger.error("ddgs não instalado: pip install ddgs")
            return "", []

        snippets: list[str] = []
        urls: list[str] = []

        try:
            time.sleep(2)  # Reduz chance de rate limit em execuções consecutivas
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5, safesearch="moderate"))
            for result in results:
                href = result.get("href", "")
                if href:
                    snippets.append(
                        f"📄 {result.get('title', '')}\n"
                        f"{result.get('body', '')}\n"
                        f"Fonte: {href}"
                    )
                    urls.append(href)
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
    ) -> DiagnosticoOutputDetalhado:
        """Monta o prompt completo e faz a única chamada ao Groq. Retorna DiagnosticoOutputDetalhado validado."""
        parts = [
            "Analise os seguintes dados clínicos veterinários e gere o "
            "diagnóstico estruturado:\n\n",
            clinical_summary,
            f"\n\n>>> VALOR OBRIGATÓRIO: pc_confianca = {confianca_calculada} <<<\n"
            "Este valor foi calculado deterministicamente pelo sistema com base "
            "nos dados clínicos reais. Use EXATAMENTE este número no campo "
            "pc_confianca — não recalcule, não ajuste, não arredonde.\n",
        ]

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

        diagnostic: DiagnosticoOutputDetalhado = self._invoke_with_fallback(messages)

        # Garante que pc_confianca é sempre o valor calculado pelo sistema
        diagnostic.pc_confianca = confianca_calculada

        # Preenche fontes caso a LLM não o tenha feito
        if sources and not diagnostic.fontes_pesquisadas:
            diagnostic.fontes_pesquisadas = sources

        return diagnostic

    def _invoke_with_fallback(self, messages: list[Any]) -> DiagnosticoOutputDetalhado:
        """
        Invoca o Groq percorrendo GROQ_MODELS na ordem configurada (modelo
        primário primeiro, depois os fallbacks).

        Para cada modelo, tenta até GROQ_MAX_RETRIES_PER_MODEL vezes — mas
        SOMENTE quando o erro é classificado como transitório (timeout,
        erro de conexão, 5xx), com backoff exponencial baseado em
        GROQ_RETRY_BACKOFF_SECONDS (1ª espera = valor, 2ª = valor*2, ...).
        Erros de cota/indisponibilidade (429, modelo descontinuado etc.)
        pulam DIRETO para o próximo modelo, sem retry — repetir no mesmo
        modelo não ajuda nesses casos.

        "Stateless" por chamada: toda invocação de analyze() reinicia esta
        função do zero, sempre tentando o modelo primário primeiro — assim
        o sistema não fica "preso" num modelo de fallback depois que a
        cota do modelo primário já foi resetada pela Groq.

        Raises:
            RuntimeError: se TODOS os modelos da lista falharem.
        """
        ultima_excecao: Exception | None = None

        for model in GROQ_MODELS:
            chain = self._get_chain(model)

            for tentativa in range(1, GROQ_MAX_RETRIES_PER_MODEL + 1):
                try:
                    diagnostic: DiagnosticoOutputDetalhado = chain.invoke(messages)
                    logger.info(
                        "Diagnóstico gerado usando modelo: %s (tentativa %d/%d)",
                        model,
                        tentativa,
                        GROQ_MAX_RETRIES_PER_MODEL,
                    )
                    return diagnostic

                except Exception as exc:
                    ultima_excecao = exc

                    if _is_quota_or_unavailable_error(exc):
                        logger.warning(
                            "Modelo '%s' indisponível ou com cota esgotada (%s) — "
                            "pulando para o próximo modelo da lista.",
                            model,
                            exc,
                        )
                        break  # não adianta retry no mesmo modelo; próximo modelo

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
                    break  # esgotou retries (ou erro não-retryable) -> próximo modelo

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
    """
    Extrai o conjunto de palavras-chave (≥4 letras) de uma predisposição,
    combinando o nome da doença com DS_SINTOMAS (palavras-chave clínicas
    catalogadas em TB_ARKIVE_DOENCA, quando presentes).
    """
    texto = f"{doenca.get('nm_doenca', '')} {doenca.get('ds_sintomas', '')}"
    return set(re.findall(r"\b\w{4,}\b", texto.lower()))


def _calculate_confidence(ctx: ClinicalContext, sintomas: str) -> int:
    """
    Calcula pc_confianca deterministicamente. O modelo recebe o valor pronto.

    Rubrica (base = 30):
    +25 Sintomas específicos e detalhados (> 3 palavras relevantes)
    +10 Sintomas moderadamente descritivos (1–3 palavras relevantes)
    +20 Predisposição genética diretamente relacionada aos sintomas
    +10 Predisposição genética presente mas indiretamente relacionada
    +10 Bem-estar completo (apetite + atividade + comportamento)
    +5  Peso registrado
    -10 Dados relevantes ausentes (idade, peso ou bem-estar)
    -15 Sintomas vagos ou genéricos

    Retorna inteiro entre 0 e 100.
    """
    score = 30  # BASE sempre

    # Sintomas
    palavras_relevantes = re.findall(r"\b\w{4,}\b", sintomas)
    if len(palavras_relevantes) > 3:
        score += 25
    elif len(palavras_relevantes) >= 1:
        score += 10
    else:
        score -= 15

    # Predisposições genéticas
    if ctx.predisposicoes:
        sintomas_lower = sintomas.lower()
        # Verifica se alguma doença mapeada tem termos presentes nos sintomas
        # relatados. Compara contra o NOME da doença E contra as palavras-
        # chave clínicas catalogadas em DS_SINTOMAS (TB_ARKIVE_DOENCA) —
        # esta última é mais precisa, já que é o campo pensado justamente
        # para o motor de regras, em vez de depender só do nome da doença.
        diretamente_relacionada = any(
            any(termo in sintomas_lower for termo in _termos_doenca(doenca))
            for doenca in ctx.predisposicoes
        )
        score += 20 if diretamente_relacionada else 10

    # Bem-estar
    if ctx.ds_apetite and ctx.ds_atividade and ctx.ds_comportamento:
        score += 10

    # Peso
    if ctx.peso_efetivo_kg:
        score += 5

    # Penalidade: dados relevantes ausentes
    dados_ausentes = (
        not ctx.nr_idade
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
    """
    Decide se a busca web deve ser acionada.

    IMPORTANTE: a decisão usa o MESMO pc_confianca já calculado
    deterministicamente por _calculate_confidence() — não recalcula uma
    métrica própria. Antes desta versão, existiam duas rubricas
    independentes (uma para "ambiguidade" com score próprio, outra para
    "confiança"), que podiam divergir entre si e não respeitavam o mesmo
    AMBIGUITY_THRESHOLD. Agora há uma única fonte de verdade.

    As "razões" abaixo continuam sendo calculadas — mas apenas para fins
    de log e para montar a query de busca web; elas NÃO determinam mais o
    booleano needs_web_search (quem determina é só a comparação de
    pc_confianca contra AMBIGUITY_THRESHOLD).
    """
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
    """
    Identifica erros de cota (rate limit / HTTP 429) ou de modelo
    indisponível/descontinuado. Nesses casos, repetir no MESMO modelo não
    ajuda — o fallback deve pular direto para o próximo modelo da lista,
    sem gastar tentativas de retry.

    A inspeção usa tanto `status_code` (quando a exceção da lib groq/
    langchain-groq o expõe) quanto o texto da mensagem como fallback, já
    que o tipo exato de exceção pode variar entre versões das bibliotecas.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    mensagem = str(exc).lower()
    marcadores = (
        "rate limit",
        "rate_limit",
        "quota",
        "429",
        "decommissioned",
        "does not exist",
        "model_not_found",
        "not found",
        "unavailable",
    )
    return any(marcador in mensagem for marcador in marcadores)


def _is_retryable_error(exc: Exception) -> bool:
    """
    Identifica erros transitórios (timeout, erro de conexão, 5xx) que
    justificam uma nova tentativa NO MESMO modelo antes de desistir dele.

    Erros de cota/indisponibilidade são tratados separadamente por
    _is_quota_or_unavailable_error() e NÃO passam por aqui — devem pular
    direto para o próximo modelo, sem retry.
    """
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


def _build_search_query(ctx: ClinicalContext, sintomas: str, motivo: str) -> str:
    """Monta query veterinária para o DuckDuckGo priorizando NCBI e Merck."""
    parts: list[str] = []

    if ctx.nm_especie:
        parts.append(ctx.nm_especie.lower())
    if ctx.nm_raca:
        parts.append(ctx.nm_raca.lower())

    texto = sintomas or motivo
    if texto:
        parts.extend(re.findall(r"\b\w{4,}\b", texto)[:4])

    parts.append(
        "veterinary clinical diagnosis "
        "site:ncbi.nlm.nih.gov OR site:merckvetmanual.com"
    )

    return " ".join(parts)[:200]