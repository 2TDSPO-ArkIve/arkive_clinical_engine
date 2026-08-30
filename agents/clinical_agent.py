"""
agents/clinical_agent.py
========================
Motor de Inteligência Clínica Veterinária ArkIve.

Pipeline:
  Etapa 1 — Oracle READ-ONLY: extração de dados clínicos
  Etapa 2 — Heurística local: decide se busca web é necessária (sem LLM)
  Etapa 3 — Cálculo determinístico do pc_confianca em Python (sem LLM)
  Etapa 4 — DuckDuckGo: busca literatura veterinária se necessário (sem LLM)
  Etapa 5 — Groq: UMA única chamada ao LLM com contexto completo

Fallback de modelo:
  Se o modelo principal falhar no function calling, tenta os próximos da
  lista GROQ_MODEL_FALLBACKS definida em config.py.
  Se todos falharem no function calling, aciona parsing manual de JSON puro.

Resultado: NO MÁXIMO 1 chamada à API do Groq por execução.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import (
    AMBIGUITY_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    GROQ_API_KEY,
    GROQ_MODEL_FALLBACKS,
    GROQ_TEMPERATURE,
)
from database.connection import get_connection
from database.queries import ClinicalContext, fetch_clinical_data
from schemas.diagnostic import DiagnosticoOutput

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  System Prompt
# ─────────────────────────────────────────────────────────────────────────────

_DIAGNOSTIC_SYSTEM_PROMPT = (
    "Você é o Motor de Inteligência Clínica Veterinária do sistema ArkIve, "
    "desenvolvido para auxiliar médicos veterinários brasileiros na formulação "
    "de hipóteses diagnósticas fundamentadas. O sistema atende qualquer espécie "
    "animal — doméstica, silvestre, zoológica ou de produção.\n\n"

    "PAPEL E RESPONSABILIDADE:\n"
    "Analise os dados clínicos fornecidos e gere uma suspeita diagnóstica "
    "estruturada, priorizando a segurança do paciente e a precisão clínica. "
    "Você está produzindo uma HIPÓTESE DIAGNÓSTICA para orientar o veterinário "
    "— não um diagnóstico definitivo.\n\n"

    "INSTRUÇÕES CRÍTICAS ANTI-ALUCINAÇÃO:\n"
    "1. Baseie-se EXCLUSIVAMENTE nos dados clínicos fornecidos e em evidências "
    "médico-veterinárias estabelecidas (ou nas fontes web incluídas no contexto).\n"
    "2. NUNCA invente sintomas, resultados laboratoriais ou informações ausentes.\n"
    "3. Predisposições genéticas são FATORES DE RISCO, não diagnósticos definitivos.\n"
    "4. Se fontes web foram consultadas, integre as evidências de forma crítica "
    "no raciocínio clínico. Não liste URLs dentro do ds_insight_ia — as fontes "
    "já serão registradas separadamente no campo fontes_pesquisadas.\n\n"

    "ESTRUTURA OBRIGATÓRIA DO ds_insight_ia:\n"
    "Escreva em português técnico e objetivo, seguindo exatamente esta ordem, "
    "sem títulos ou marcadores — apenas parágrafos fluidos:\n"
    "  1º parágrafo: perfil do paciente e apresentação clínica principal.\n"
    "  2º parágrafo: correlação entre sintomas, bem-estar e hipótese diagnóstica.\n"
    "  3º parágrafo: papel das predisposições genéticas no raciocínio clínico.\n"
    "  4º parágrafo: limitações do diagnóstico e exames complementares sugeridos. "
    "Se fontes web enriqueceram o diagnóstico, mencione apenas que evidências "
    "da literatura veterinária corroboram a hipótese — sem colar URLs.\n\n"

    "RACIOCÍNIO CLÍNICO ESPERADO:\n"
    "- Correlacione sintomas com espécie, raça, sexo e status reprodutivo.\n"
    "- Considere dados de bem-estar como indicadores sistêmicos relevantes.\n"
    "- Priorize predisposições genéticas mapeadas como diferenciais prioritários.\n\n"

    "GARANTIAS DE TIPO OBRIGATÓRIAS — VIOLAÇÕES CAUSAM FALHA NO SISTEMA:\n"
    "• `ds_diagnostico`     → string de texto, entre 5 e 500 caracteres.\n"
    "• `tp_severidade`      → exatamente uma destas strings: 'LEVE', 'MODERADA' "
    "ou 'GRAVE'. Nunca use outros valores.\n"
    "• `ds_insight_ia`      → string de texto, mínimo 50 caracteres. "
    "PROIBIDO incluir URLs, links ou endereços web neste campo. "
    "Se fontes web foram consultadas, mencione apenas que a literatura "
    "veterinária corrobora a hipótese — as URLs ficam exclusivamente "
    "em fontes_pesquisadas.\n"
    "• `pc_confianca`       → inteiro puro fornecido pelo sistema no campo "
    "'>>> VALOR OBRIGATÓRIO: pc_confianca <<<'. Use EXATAMENTE este número. "
    "NUNCA recalcule, NUNCA ajuste, NUNCA envie como string ou float.\n"
    "• `fontes_pesquisadas` → lista de strings com URLs. Lista vazia [] se "
    "busca web não foi realizada. NUNCA null ou omitido.\n\n"

    "Responda EXCLUSIVAMENTE no formato JSON estruturado conforme o schema "
    "fornecido. Não inclua texto adicional fora do JSON."
)

# ─────────────────────────────────────────────────────────────────────────────
#  Dataclass auxiliar
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _LocalAmbiguityResult:
    """Resultado da avaliação de ambiguidade feita em Python puro (sem LLM)."""
    needs_web_search: bool
    score: int
    reason: str
    search_query: str


# ─────────────────────────────────────────────────────────────────────────────
#  Classe Principal
# ─────────────────────────────────────────────────────────────────────────────


class ClinicalIntelligenceEngine:
    """
    Motor de Inteligência Clínica Veterinária com RAG local (Oracle),
    fallback automático de busca web (DuckDuckGo) e fallback de modelo Groq.

    Tenta cada modelo em GROQ_MODEL_FALLBACKS até encontrar um que suporte
    function calling. Se nenhum suportar, aciona parsing manual de JSON puro.
    """

    def __init__(self) -> None:
        logger.info("Inicializando ClinicalIntelligenceEngine...")

        self._llm = None
        self._diagnostic_chain = None
        self._modelo_ativo: str = ""

        for model in GROQ_MODEL_FALLBACKS:
            try:
                logger.info("Tentando modelo: %s", model)
                llm = ChatGroq(
                    model=model,
                    api_key=GROQ_API_KEY,
                    temperature=GROQ_TEMPERATURE,
                )
                chain = llm.with_structured_output(DiagnosticoOutput)
                self._llm = llm
                self._diagnostic_chain = chain
                self._modelo_ativo = model
                logger.info("Modelo ativo: %s", model)
                break
            except Exception as exc:
                logger.warning(
                    "Modelo %s indisponível: %s — tentando próximo.", model, exc
                )

        if self._llm is None:
            raise RuntimeError(
                "Nenhum modelo Groq disponível no momento. "
                "Verifique a GROQ_API_KEY e o status em https://console.groq.com"
            )

        logger.info("Chain de diagnóstico inicializada (1 chamada de API por execução).")

    # ──────────────────────────────────────────────────────────────────────────
    #  Método Público Principal
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self, id_consulta: int) -> dict[str, Any]:
        """
        Ponto de entrada principal. Faz exatamente 1 chamada ao Groq.

        Raises:
            ValueError: Consulta não encontrada no banco.
            RuntimeError: Todos os modelos e estratégias falharam.
            oracledb.DatabaseError: Falha de acesso ao Oracle.
        """
        logger.info("── Iniciando análise clínica | ID_CONSULTA=%d ──", id_consulta)

        # Etapa 1: Extração Oracle
        ctx: ClinicalContext = self._fetch_oracle_data(id_consulta)
        clinical_summary: str = ctx.to_clinical_summary()

        # Etapa 2: Heurística local de ambiguidade
        sintomas = (ctx.ds_sintomas or "").strip()
        confianca_calculada = _calculate_confidence(ctx, sintomas)
        logger.info("Confiança calculada deterministicamente: %d%%", confianca_calculada)

        ambiguity = _evaluate_ambiguity_locally(ctx, confianca_calculada)
        logger.info(
            "Heurística local | score=%d%% | busca_web=%s | motivo: %s",
            ambiguity.score,
            ambiguity.needs_web_search,
            ambiguity.reason,
        )

        # Etapa 4: Busca web condicional
        web_context: str = ""
        sources: list[str] = []

        if ambiguity.needs_web_search:
            logger.info("Acionando busca web | Query: '%s'", ambiguity.search_query)
            web_context, sources = self._perform_web_search(ambiguity.search_query)
            logger.info("%d fonte(s) recuperada(s) do DuckDuckGo.", len(sources))
        else:
            logger.info("Dados locais suficientes — busca web não acionada.")

        # Etapa 5: Chamada ao Groq
        logger.info("Enviando requisição ao Groq (chamada 1/1)...")
        diagnostic: DiagnosticoOutput = self._generate_diagnostic(
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

        return diagnostic.model_dump()

    # ──────────────────────────────────────────────────────────────────────────
    #  Métodos Privados
    # ──────────────────────────────────────────────────────────────────────────

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
    ) -> DiagnosticoOutput:
        """
        Monta o prompt completo e tenta gerar o diagnóstico.

        Estratégia 1: function calling via with_structured_output().
        Estratégia 2: fallback de parsing manual de JSON puro.
        """
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
                "\nIMPORTANTE: integre as evidências ao raciocínio clínico no "
                "ds_insight_ia. Não cole URLs no insight — elas já estão em "
                "fontes_pesquisadas.",
            ])

        if sources:
            parts.append(
                "\nURLs consultadas (para fontes_pesquisadas):\n"
                + "\n".join(f"  {i + 1}. {url}" for i, url in enumerate(sources))
            )

        messages = [
            SystemMessage(content=_DIAGNOSTIC_SYSTEM_PROMPT),
            HumanMessage(content="".join(parts)),
        ]

        diagnostic: DiagnosticoOutput | None = None

        # Estratégia 1: function calling (with_structured_output)
        try:
            diagnostic = self._diagnostic_chain.invoke(messages)
            logger.info("Structured output via function calling bem-sucedido.")
        except Exception as exc:
            logger.warning(
                "Function calling falhou (%s) — acionando fallback de JSON parsing.", exc
            )

        # Estratégia 2: fallback de JSON puro
        if diagnostic is None:
            diagnostic = self._generate_diagnostic_json_fallback(messages)

        # Garante que pc_confianca é sempre o valor calculado pelo sistema
        diagnostic.pc_confianca = confianca_calculada

        if sources and not diagnostic.fontes_pesquisadas:
            diagnostic.fontes_pesquisadas = sources

        return diagnostic

    def _generate_diagnostic_json_fallback(
        self, messages: list
    ) -> DiagnosticoOutput:
        """
        Fallback para modelos que não suportam function calling.
        Instrui o modelo a retornar JSON puro e faz o parsing manualmente.
        Compatível com qualquer modelo que suporte chat completion básico.
        """
        json_schema_instruction = (
            "\n\nRetorne APENAS o seguinte JSON, sem nenhum texto antes ou depois, "
            "sem blocos de código markdown:\n"
            "{\n"
            '  "ds_diagnostico": "string entre 5 e 500 caracteres",\n'
            '  "tp_severidade": "LEVE" ou "MODERADA" ou "GRAVE",\n'
            '  "ds_insight_ia": "string com mínimo 50 caracteres, sem URLs",\n'
            '  "pc_confianca": integer entre 0 e 100,\n'
            '  "fontes_pesquisadas": []\n'
            "}"
        )

        messages_with_json = list(messages)
        last_human = messages_with_json[-1]
        messages_with_json[-1] = HumanMessage(
            content=last_human.content + json_schema_instruction
        )

        try:
            response = self._llm.invoke(messages_with_json)
            raw_text = response.content.strip()

            # Remove blocos de código markdown se presentes
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text)

            data = json.loads(raw_text)
            diagnostic = DiagnosticoOutput(**data)
            logger.info("Fallback JSON parsing bem-sucedido.")
            return diagnostic

        except Exception as exc:
            logger.error("Fallback JSON parsing também falhou: %s", exc)
            raise RuntimeError(
                f"Nenhuma estratégia de geração funcionou para este modelo: {exc}"
            ) from exc


# ─────────────────────────────────────────────────────────────────────────────
#  Funções auxiliares (fora da classe)
# ─────────────────────────────────────────────────────────────────────────────


def _calculate_confidence(ctx: ClinicalContext, sintomas: str) -> int:
    """
    Calcula pc_confianca deterministicamente. O modelo recebe o valor pronto.

    Rubrica (base = 30):
    +25 Sintomas específicos e detalhados (> 3 palavras relevantes)
    +10 Sintomas moderadamente descritivos (1-3 palavras relevantes)
    +20 Predisposição genética diretamente relacionada aos sintomas
    +10 Predisposição genética presente mas indiretamente relacionada
    +10 Bem-estar completo (apetite + atividade + comportamento)
    +5  Peso registrado
    -10 Dados relevantes ausentes (idade, peso ou bem-estar)
    -15 Sintomas vagos ou genéricos

    Retorna inteiro entre 0 e 100.
    """
    score = 30  # BASE sempre

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
            any(
                termo in sintomas_lower
                for termo in re.findall(r"\b\w{4,}\b", doenca.get("nm_doenca", "").lower())
            )
            for doenca in ctx.predisposicoes
        )
        score += 20 if diretamente_relacionada else 10

    if ctx.ds_apetite and ctx.ds_atividade and ctx.ds_comportamento:
        score += 10

    if ctx.peso_efetivo_kg:
        score += 5

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
        "completo" if ctx.ds_apetite and ctx.ds_atividade and ctx.ds_comportamento
        else "parcial/ausente",
        f"{ctx.peso_efetivo_kg}kg" if ctx.peso_efetivo_kg else "ausente",
        resultado,
    )

    return resultado


def _evaluate_ambiguity_locally(ctx: ClinicalContext, confianca_calculada: int) -> _LocalAmbiguityResult:
    """
    Avalia qualidade dos dados clínicos com regras determinísticas (sem LLM).

    A busca web é acionada se QUALQUER um dos critérios for verdadeiro:
      - score de qualidade dos dados < AMBIGUITY_THRESHOLD
      - confianca_calculada < CONFIDENCE_THRESHOLD

    Penalidades no score de qualidade (base = 100):
    -35 DS_SINTOMAS ausente ou < 20 caracteres
    -15 DS_SINTOMAS genérico (1 palavra)
    -20 DS_MOTIVO ausente ou < 10 caracteres
    -20 Sem predisposições genéticas mapeadas para a espécie/raça
    -10 Avaliação de bem-estar ausente
    """
    score = 100
    reasons: list[str] = []

    sintomas = (ctx.ds_sintomas or "").strip()
    motivo = (ctx.ds_motivo or "").strip()

    if len(sintomas) < 20:
        score -= 35
        reasons.append("sintomas ausentes ou insuficientes")
    elif len(re.findall(r"\w+", sintomas)) <= 1:
        score -= 15
        reasons.append("sintomas excessivamente genéricos")

    if len(motivo) < 10:
        score -= 20
        reasons.append("motivo da consulta não informado")

    if not ctx.predisposicoes:
        especie_str = ctx.nm_especie or "não identificada"
        raca_str = f" / raça '{ctx.nm_raca}'" if ctx.nm_raca else ""
        score -= 20
        reasons.append(
            f"nenhuma predisposição genética mapeada no banco para a espécie "
            f"'{especie_str}'{raca_str} — busca web pode enriquecer o diagnóstico"
        )

    if not ctx.ds_apetite and not ctx.ds_atividade and not ctx.ds_comportamento:
        score -= 10
        reasons.append("avaliação de bem-estar ausente")

    score = max(0, score)

    score_baixo = score < AMBIGUITY_THRESHOLD
    confianca_baixa = confianca_calculada < CONFIDENCE_THRESHOLD

    if confianca_baixa and not score_baixo:
        reasons.append(
            f"confiança diagnóstica ({confianca_calculada}%) abaixo do limiar "
            f"({CONFIDENCE_THRESHOLD}%) — busca web acionada para enriquecer o diagnóstico"
        )

    return _LocalAmbiguityResult(
        needs_web_search=score_baixo or confianca_baixa,
        score=score,
        reason="; ".join(reasons) if reasons else "dados clínicos suficientes",
        search_query=_build_search_query(ctx, sintomas, motivo),
    )


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