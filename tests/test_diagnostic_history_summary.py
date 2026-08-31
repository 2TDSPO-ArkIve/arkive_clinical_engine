"""
tests/test_diagnostic_history_summary.py
==========================================
Testes da seção "HISTÓRICO DE DIAGNÓSTICOS ANTERIORES" dentro de
ClinicalContext.to_clinical_summary() (database/queries.py).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from database.queries import ClinicalContext


def _make_ctx(**overrides) -> ClinicalContext:
    base = dict(
        id_animal=1,
        nm_animal="Rex",
        ds_sexo="M",
        ds_castrado="S",
        nm_especie="Canina",
        nm_raca="Labrador",
        id_consulta=10,
        ds_motivo="Retorno",
        ds_sintomas="Letargia",
    )
    base.update(overrides)
    return ClinicalContext(**base)


class TestHistoricoDiagnosticosNoSumario:
    def test_sem_historico_mostra_mensagem_padrao(self):
        ctx = _make_ctx(diagnosticos_anteriores=[])
        resumo = ctx.to_clinical_summary()
        assert "HISTÓRICO DE DIAGNÓSTICOS ANTERIORES" in resumo
        assert "Nenhum diagnóstico anterior registrado" in resumo

    def test_com_historico_lista_diagnosticos(self):
        ctx = _make_ctx(
            diagnosticos_anteriores=[
                {
                    "dt_hora": datetime(2025, 1, 10, 14, 30),
                    "ds_diagnostico": "Suspeita de Gastroenterite",
                    "nm_doenca": None,
                    "tp_severidade": "MODERADA",
                    "pc_confianca": 60,
                    "st_confirmado": "S",
                },
                {
                    "dt_hora": datetime(2024, 11, 2, 9, 0),
                    "ds_diagnostico": "Suspeita de Hipotireoidismo",
                    "nm_doenca": "Hipotireoidismo Canino",
                    "tp_severidade": "LEVE",
                    "pc_confianca": 45,
                    "st_confirmado": "N",
                },
            ]
        )
        resumo = ctx.to_clinical_summary()

        assert "Suspeita de Gastroenterite" in resumo
        assert "Hipotireoidismo Canino" in resumo  # usa nm_doenca quando disponível
        assert "10/01/2025" in resumo
        assert "02/11/2024" in resumo
        assert "MODERADA" in resumo
        assert "confirmado" in resumo
        assert "não confirmado" in resumo

    def test_usa_ds_diagnostico_quando_nm_doenca_ausente(self):
        ctx = _make_ctx(
            diagnosticos_anteriores=[
                {
                    "dt_hora": datetime(2025, 3, 1),
                    "ds_diagnostico": "Suspeita de Otite Externa",
                    "nm_doenca": None,
                    "tp_severidade": "LEVE",
                    "pc_confianca": 50,
                    "st_confirmado": "S",
                }
            ]
        )
        resumo = ctx.to_clinical_summary()
        assert "Suspeita de Otite Externa" in resumo

    def test_exibe_status_de_validacao_pelo_veterinario(self):
        """
        Regressão: ST_VALIDACAO_VET era extraído do Oracle mas nunca
        aparecia no resumo textual enviado ao LLM, apesar do system prompt
        instruir 'ceticismo redobrado' para diagnósticos não validados.
        """
        ctx = _make_ctx(
            diagnosticos_anteriores=[
                {
                    "dt_hora": datetime(2025, 1, 10),
                    "ds_diagnostico": "Suspeita de Gastroenterite",
                    "nm_doenca": None,
                    "tp_severidade": "MODERADA",
                    "pc_confianca": 60,
                    "st_confirmado": "S",
                    "st_validacao_vet": "S",
                },
                {
                    "dt_hora": datetime(2024, 11, 2),
                    "ds_diagnostico": "Suspeita de Hipotireoidismo",
                    "nm_doenca": None,
                    "tp_severidade": "LEVE",
                    "pc_confianca": 45,
                    "st_confirmado": "N",
                    "st_validacao_vet": "N",
                },
                {
                    "dt_hora": datetime(2024, 9, 1),
                    "ds_diagnostico": "Suspeita de Dermatite",
                    "nm_doenca": None,
                    "tp_severidade": "LEVE",
                    "pc_confianca": 40,
                    "st_confirmado": "N",
                    "st_validacao_vet": None,
                },
            ]
        )
        resumo = ctx.to_clinical_summary()
        assert "validado pelo veterinário" in resumo
        assert "não validado pelo veterinário" in resumo
        assert "não avaliado pelo veterinário" in resumo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])