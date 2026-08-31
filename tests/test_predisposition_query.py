"""
tests/test_predisposition_query.py
====================================
Regressão: PREDISPOSITION_QUERY não filtrava doenças com ST_ATIVO = 'N',
permitindo que doenças excluídas logicamente do catálogo (TB_ARKIVE_DOENCA)
ainda fossem trazidas como predisposição genética e enviadas ao LLM.

Como TB_ARKIVE_PREDISPOSICAO não tem coluna ST_ATIVO própria, o filtro
correto é sempre em TB_ARKIVE_DOENCA.ST_ATIVO = 'S'.
"""

from __future__ import annotations

from database.queries import PREDISPOSITION_QUERY


def test_predisposition_query_filtra_doenca_ativa():
    sql_normalizado = " ".join(PREDISPOSITION_QUERY.split())
    assert "D.ST_ATIVO = 'S'" in sql_normalizado.upper()


def test_predisposition_query_traz_ds_sintomas_para_correlacao():
    """DS_SINTOMAS precisa ser selecionado para alimentar a correlação em
    agents.clinical_agent._calculate_confidence."""
    sql_normalizado = " ".join(PREDISPOSITION_QUERY.split())
    assert "D.DS_SINTOMAS" in sql_normalizado.upper()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
