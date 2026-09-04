"""
schemas/diagnostic_detalhado.py
================================
Schema Pydantic v2 que define o contrato de saída estruturada da IA usado
em conjunto com prompts/diagnostic.py (raciocínio clínico dividido em 4
campos independentes: perfil, correlação, predisposição e limitações).

Mantido em arquivo/classe separados de schemas/diagnostic.py apenas por
organização — DiagnosticoOutput (schema com um único campo ds_insight_ia)
permanece intacto naquele arquivo. Esta separação não representa
versionamento (isso é responsabilidade do Git); é só uma divisão por
responsabilidade para facilitar a manutenção de cada schema.

DiagnosticoOutputDetalhado:
    Schema principal preenchido pela IA via ChatGroq.with_structured_output().
    Os 4 campos de insight são compostos em um único texto através de
    to_ds_insight_ia(), preservando o contrato de uma única coluna
    DS_INSIGHT_IA (CLOB) esperado pelo serviço Java em
    TB_ARKIVE_DIAGNOSTICO — os campos brutos continuam disponíveis no dict
    de saída de agents/clinical_agent.py para quem quiser consumi-los
    granularmente, mas não são persistidos individualmente.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DiagnosticoOutputDetalhado(BaseModel):
    """
    Saída estruturada do Motor de Inteligência Clínica Veterinária ArkIve,
    com o raciocínio clínico dividido em 4 campos independentes (perfil,
    correlação, predisposição, limitações), conforme instruído em
    prompts/diagnostic.py.

    Mapeamento para TB_ARKIVE_DIAGNOSTICO (gravação pelo serviço Java):
    ┌────────────────────────┬───────────────────────────────────────────────┐
    │ Campo Python           │ Coluna Oracle / Tipo / Restrição               │
    ├────────────────────────┼───────────────────────────────────────────────┤
    │ ds_diagnostico         │ DS_DIAGNOSTICO  VARCHAR2(500)                  │
    │ tp_severidade          │ TP_SEVERIDADE   VARCHAR2(20)                   │
    │                        │   CHECK IN ('LEVE', 'MODERADA', 'GRAVE')       │
    │ insight_perfil         │ ┐                                              │
    │ insight_correlacao     │ ├─ compostos via to_ds_insight_ia() em         │
    │ insight_predisposicao  │ │  DS_INSIGHT_IA CLOB (coluna única)           │
    │ insight_limitacoes     │ ┘                                              │
    │ pc_confianca           │ PC_CONFIANCA    NUMBER(3) CHECK (0..100)       │
    │ fontes_pesquisadas     │ (não persistido — informativo ao chamador)     │
    └────────────────────────┴───────────────────────────────────────────────┘
    """

    ds_diagnostico: str = Field(
        min_length=5,
        max_length=500,
        description=(
            "Linha fina / título conciso da suspeita diagnóstica clínica sugerida. "
            "Deve nomear a hipótese principal de forma objetiva. "
            "Exemplo: 'Suspeita de Hipotireoidismo Canino'."
        ),
    )

    tp_severidade: Literal["LEVE", "MODERADA", "GRAVE"] = Field(
        description=(
            "Classificação de severidade do quadro clínico. Valores aceitos: "
            "LEVE (quadro estável, sem risco imediato), "
            "MODERADA (requer atenção e acompanhamento próximo), "
            "GRAVE (risco de vida — intervenção urgente indicada)."
        )
    )

    insight_perfil: str = Field(
        min_length=20,
        description="Perfil do paciente e apresentação clínica principal.",
    )

    insight_correlacao: str = Field(
        min_length=20,
        description=(
            "Correlação entre sintomas, bem-estar e hipótese diagnóstica. "
            "Inclui a correlação com diagnósticos anteriores, quando houver "
            "histórico relevante."
        ),
    )

    insight_predisposicao: str = Field(
        min_length=20,
        description=(
            "Papel das predisposições genéticas no raciocínio clínico, "
            "considerando que o catálogo local ainda está em povoamento — "
            "ausência de predisposição mapeada é limitação de dado, não "
            "evidência de ausência de risco genético."
        ),
    )

    insight_limitacoes: str = Field(
        min_length=20,
        description=(
            "Limitações do diagnóstico e exames complementares sugeridos. "
            "Menciona evidências de literatura veterinária quando fontes web "
            "foram consultadas, sem incluir URLs."
        ),
    )

    pc_confianca: int = Field(
        ge=0,
        le=100,
        description=(
            "Grau de certeza estimado pela IA (0 a 100%). Fornecido pronto "
            "pelo sistema (calculado deterministicamente em Python) — o "
            "modelo NUNCA deve recalcular este valor."
        ),
    )

    fontes_pesquisadas: list[str] = Field(
        default_factory=list,
        description=(
            "Lista de URLs ou identificadores de fontes consultadas na internet "
            "durante a etapa de busca web. "
            "DEVE estar vazia se os dados locais do Oracle foram suficientes para o diagnóstico."
        ),
    )

    @field_validator(
        "ds_diagnostico",
        "insight_perfil",
        "insight_correlacao",
        "insight_predisposicao",
        "insight_limitacoes",
        mode="before",
    )
    @classmethod
    def _strip_texto(cls, v: str) -> str:
        """Remove espaços extras das bordas dos campos de texto."""
        return v.strip() if isinstance(v, str) else v

    def to_ds_insight_ia(self) -> str:
        """
        Compõe os 4 campos de insight em um único texto, na ordem em que
        prompts/diagnostic.py instrui o modelo a produzi-los (perfil →
        correlação → predisposição → limitações).

        Usado por agents/clinical_agent.py para preencher o campo
        `ds_insight_ia` do payload final, mantendo compatibilidade com o
        contrato de coluna única (DS_INSIGHT_IA CLOB) esperado pelo serviço
        Java em TB_ARKIVE_DIAGNOSTICO.
        """
        return "\n\n".join(
            [
                self.insight_perfil,
                self.insight_correlacao,
                self.insight_predisposicao,
                self.insight_limitacoes,
            ]
        )
