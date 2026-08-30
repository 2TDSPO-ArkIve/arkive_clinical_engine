"""
tests/test_clinical_agent.py
=============================
Testes unitários das funções puras e determinísticas de agents/clinical_agent.py:

- _calculate_confidence: rubrica de cálculo do pc_confianca
- _evaluate_ambiguity_locally: heurística de decisão de busca web
- _build_search_query: montagem da query do DuckDuckGo

Essas funções não dependem de Oracle, Groq nem rede — são as peças mais
"auditáveis" do sistema (a rubrica que efetivamente pesa no diagnóstico
clínico) e por isso são as prioritárias para cobertura de testes.

Rodar com:
    pytest tests/test_clinical_agent.py -v
"""

from __future__ import annotations

import pytest

from agents.clinical_agent import (
    ClinicalIntelligenceEngine,
    _build_search_query,
    _calculate_confidence,
    _evaluate_ambiguity_locally,
    _web_search_cache,
)
from database.queries import ClinicalContext


def _make_ctx(**overrides) -> ClinicalContext:
    """Cria um ClinicalContext mínimo, sobrescrevendo campos conforme necessário."""
    base = dict(
        id_animal=1,
        nm_animal="Rex",
        ds_sexo="M",
        ds_castrado="S",
        id_especie=1,
        nm_especie="Canina",
        id_raca=1,
        nm_raca="Labrador",
        tp_porte="GRANDE",
        id_consulta=1,
        ds_motivo="Vômito e diarreia recorrentes",
        ds_sintomas="",
        kg_peso_consulta=25.0,
        nr_idade=5.0,
        ds_apetite="",
        ds_atividade="",
        ds_comportamento="",
        predisposicoes=[],
    )
    base.update(overrides)
    return ClinicalContext(**base)


# ─────────────────────────────────────────────────────────────────────────────
# _calculate_confidence
# ─────────────────────────────────────────────────────────────────────────────


class TestCalculateConfidence:
    def test_base_score_sem_dados_extras(self):
        ctx = _make_ctx()
        # sintomas vazio (-15), sem predisposição, sem bem-estar, com peso (+5)
        # 30 - 15 + 5 - 10 (dados ausentes: bem-estar) = 10
        score = _calculate_confidence(ctx, sintomas="")
        assert 0 <= score <= 100
        assert score < 50  # dados insuficientes devem gerar confiança baixa

    def test_sintomas_detalhados_aumentam_score(self):
        ctx = _make_ctx()
        sintomas_detalhados = "vomito diarreia letargia perda apetite febre"
        score_detalhado = _calculate_confidence(ctx, sintomas=sintomas_detalhados)
        score_vazio = _calculate_confidence(ctx, sintomas="")
        assert score_detalhado > score_vazio

    def test_predisposicao_diretamente_relacionada_aumenta_mais_que_indireta(self):
        sintomas = "hipotireoidismo suspeito com letargia e ganho de peso"
        ctx_direta = _make_ctx(
            predisposicoes=[{"nm_doenca": "Hipotireoidismo Canino", "ds_doenca": "..."}]
        )
        ctx_indireta = _make_ctx(
            predisposicoes=[{"nm_doenca": "Displasia Coxofemoral", "ds_doenca": "..."}]
        )
        score_direta = _calculate_confidence(ctx_direta, sintomas=sintomas)
        score_indireta = _calculate_confidence(ctx_indireta, sintomas=sintomas)
        assert score_direta > score_indireta

    def test_bem_estar_completo_aumenta_score(self):
        sintomas = "sintomas moderados"
        ctx_sem_bem_estar = _make_ctx()
        ctx_com_bem_estar = _make_ctx(
            ds_apetite="Reduzido", ds_atividade="Baixa", ds_comportamento="Apático"
        )
        score_sem = _calculate_confidence(ctx_sem_bem_estar, sintomas=sintomas)
        score_com = _calculate_confidence(ctx_com_bem_estar, sintomas=sintomas)
        assert score_com > score_sem

    def test_score_nunca_sai_do_intervalo_0_100(self):
        ctx = _make_ctx(kg_peso_consulta=None, nr_idade=None)
        score = _calculate_confidence(ctx, sintomas="")
        assert 0 <= score <= 100


# ─────────────────────────────────────────────────────────────────────────────
# _evaluate_ambiguity_locally
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluateAmbiguityLocally:
    def test_dados_completos_nao_aciona_busca_web(self):
        ctx = _make_ctx(
            ds_sintomas="Vômito recorrente há 3 dias, letargia e perda de apetite intensa",
            ds_motivo="Consulta de rotina com queixa gastrointestinal",
            predisposicoes=[{"nm_doenca": "Gastroenterite", "ds_doenca": "..."}],
            ds_apetite="Reduzido",
            ds_atividade="Baixa",
            ds_comportamento="Apático",
        )
        resultado = _evaluate_ambiguity_locally(ctx, confianca_calculada=85)
        assert resultado.needs_web_search is False
        assert resultado.score >= 60

    def test_sintomas_ausentes_aciona_busca_web(self):
        ctx = _make_ctx(ds_sintomas="", ds_motivo="")
        resultado = _evaluate_ambiguity_locally(ctx, confianca_calculada=80)
        assert resultado.needs_web_search is True
        assert "sintomas" in resultado.reason

    def test_confianca_baixa_aciona_busca_mesmo_com_dados_completos(self):
        ctx = _make_ctx(
            ds_sintomas="Vômito recorrente há 3 dias, letargia e perda de apetite intensa",
            ds_motivo="Consulta de rotina com queixa gastrointestinal",
            predisposicoes=[{"nm_doenca": "Gastroenterite", "ds_doenca": "..."}],
            ds_apetite="Reduzido",
            ds_atividade="Baixa",
            ds_comportamento="Apático",
        )
        resultado = _evaluate_ambiguity_locally(ctx, confianca_calculada=40)
        assert resultado.needs_web_search is True

    def test_sem_predisposicao_mapeada_penaliza_score(self):
        sintomas = "Vômito recorrente há 3 dias, letargia e perda de apetite intensa"
        ctx_com_pred = _make_ctx(
            ds_sintomas=sintomas,
            ds_motivo="Consulta de rotina",
            predisposicoes=[{"nm_doenca": "Gastroenterite", "ds_doenca": "..."}],
        )
        ctx_sem_pred = _make_ctx(
            ds_sintomas=sintomas, ds_motivo="Consulta de rotina", predisposicoes=[]
        )
        resultado_com = _evaluate_ambiguity_locally(ctx_com_pred, confianca_calculada=80)
        resultado_sem = _evaluate_ambiguity_locally(ctx_sem_pred, confianca_calculada=80)
        assert resultado_sem.score < resultado_com.score

    def test_score_nunca_negativo(self):
        ctx = _make_ctx(ds_sintomas="", ds_motivo="", predisposicoes=[])
        resultado = _evaluate_ambiguity_locally(ctx, confianca_calculada=0)
        assert resultado.score >= 0


# ─────────────────────────────────────────────────────────────────────────────
# _build_search_query
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildSearchQuery:
    def test_inclui_especie_e_raca(self):
        ctx = _make_ctx(nm_especie="Canina", nm_raca="Labrador")
        query = _build_search_query(ctx, sintomas="vomito diarreia", motivo="")
        assert "canina" in query.lower()
        assert "labrador" in query.lower()

    def test_prioriza_fontes_confiaveis(self):
        ctx = _make_ctx()
        query = _build_search_query(ctx, sintomas="letargia febre", motivo="")
        assert "ncbi.nlm.nih.gov" in query
        assert "merckvetmanual.com" in query

    def test_usa_motivo_quando_sintomas_vazio(self):
        ctx = _make_ctx(nm_especie="Felina", nm_raca="")
        query = _build_search_query(ctx, sintomas="", motivo="anorexia progressiva grave")
        assert "anorexia" in query.lower()

    def test_query_nunca_excede_200_caracteres(self):
        ctx = _make_ctx(nm_especie="Canina", nm_raca="Labrador Retriever")
        sintomas_longos = " ".join(["sintoma" + str(i) for i in range(50)])
        query = _build_search_query(ctx, sintomas=sintomas_longos, motivo="")
        assert len(query) <= 200


# ─────────────────────────────────────────────────────────────────────────────
# _perform_web_search — cache de buscas
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformWebSearchCache:
    """
    Testa a integração do cache TTL+LRU dentro de _perform_web_search, sem
    depender de rede real: o DDGS é mockado.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        # Garante isolamento entre testes, já que o cache é um singleton
        # de módulo compartilhado entre chamadas.
        _web_search_cache.clear()
        yield
        _web_search_cache.clear()

    def _make_engine_without_init(self) -> ClinicalIntelligenceEngine:
        """Cria a instância sem rodar __init__ (evita precisar de GROQ_API_KEY real)."""
        return ClinicalIntelligenceEngine.__new__(ClinicalIntelligenceEngine)

    def test_segunda_busca_identica_usa_cache_sem_chamar_ddgs(self, monkeypatch):
        import sys
        import types

        call_count = {"n": 0}

        class _FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def text(self, query, max_results, safesearch):
                call_count["n"] += 1
                return [
                    {"title": "Artigo Teste", "body": "Corpo do artigo", "href": "https://ncbi.nlm.nih.gov/artigo"}
                ]

        fake_module = types.ModuleType("ddgs")
        fake_module.DDGS = _FakeDDGS
        monkeypatch.setitem(sys.modules, "ddgs", fake_module)
        monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

        engine = self._make_engine_without_init()

        _ctx1, urls1 = engine._perform_web_search("canina labrador vomito diarreia")
        _ctx2, urls2 = engine._perform_web_search("canina labrador vomito diarreia")

        assert call_count["n"] == 1  # DDGS só foi chamado na primeira vez
        assert urls1 == urls2 == ["https://ncbi.nlm.nih.gov/artigo"]

    def test_falha_na_busca_nao_e_cacheada(self, monkeypatch):
        import sys
        import types

        class _FailingDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def text(self, *args, **kwargs):
                raise RuntimeError("rate limit simulado")

        fake_module = types.ModuleType("ddgs")
        fake_module.DDGS = _FailingDDGS
        monkeypatch.setitem(sys.modules, "ddgs", fake_module)
        monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

        engine = self._make_engine_without_init()
        web_context, urls = engine._perform_web_search("query que sempre falha")

        assert web_context == ""
        assert urls == []
        assert len(_web_search_cache) == 0  # nada foi cacheado após falha


if __name__ == "__main__":
    pytest.main([__file__, "-v"])