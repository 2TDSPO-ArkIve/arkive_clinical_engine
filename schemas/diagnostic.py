"""
schemas/diagnostic.py
=====================
Schemas Pydantic v2 que definem o contrato de saída estruturada da IA.

DiagnosticoOutput:
    Schema principal mapeado para gravação futura na tabela TB_ARKIVE_DIAGNOSTICO
    pelo serviço Java/Spring downstream. Os nomes de campo e domínios de valores
    refletem exatamente as colunas e CHECK constraints do banco Oracle.

AmbiguityAssessment:
    Schema interno usado na etapa de decisão sobre necessidade de busca web.
    Não é persistido — serve apenas para controle de fluxo interno do agente.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator


class DiagnosticoOutput(BaseModel):
    """
    Saída estruturada do Motor de Inteligência Clínica Veterinária ArkIve.

    O raciocínio clínico da IA é estruturado em 4 campos (insight_perfil,
    insight_correlacao, insight_predisposicao, insight_limitacoes) em vez de
    um único texto corrido. Isso facilita a exibição em seções separadas por
    consumidores futuros (ex: front-end com abas/accordion), sem exigir
    parsing de texto livre.

    Para persistência, os 4 campos são concatenados automaticamente no campo
    calculado `ds_insight_ia`, que é o que efetivamente é gravado na coluna
    ─ mantém compatibilidade com o CLOB único do Oracle, sem migração de schema.

    Mapeamento para TB_ARKIVE_DIAGNOSTICO (gravação pelo serviço Java):
    ┌──────────────────────┬─────────────────────────────────────────────────┐
    │ Campo Python         │ Coluna Oracle / Tipo / Restrição                │
    ├──────────────────────┼─────────────────────────────────────────────────┤
    │ ds_diagnostico       │ DS_DIAGNOSTICO  CLOB (NOT NULL)                 │
    │ tp_severidade        │ TP_SEVERIDADE   VARCHAR2(20)                    │
    │                      │   CHECK IN ('LEVE', 'MODERADA', 'GRAVE')        │
    │ ds_insight_ia        │ DS_INSIGHT_IA   CLOB (calculado a partir dos    │
    │ (computed_field)     │   4 campos insight_* abaixo)                    │
    │ pc_confianca         │ PC_CONFIANCA    NUMBER(5,2) CHECK (0..100)      │
    │ fontes_pesquisadas   │ (não persistido — informativo ao chamador)      │
    └──────────────────────┴─────────────────────────────────────────────────┘
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
        description=(
            "Parágrafo 1: perfil do paciente e apresentação clínica principal "
            "(espécie, raça, sexo, status reprodutivo e sintomas centrais)."
        ),
    )

    insight_correlacao: str = Field(
        min_length=20,
        description=(
            "Parágrafo 2: correlação entre os sintomas relatados, os dados de "
            "bem-estar (apetite, atividade, comportamento) e a hipótese "
            "diagnóstica sugerida."
        ),
    )

    insight_predisposicao: str = Field(
        min_length=20,
        description=(
            "Parágrafo 3: papel das predisposições genéticas da raça/espécie "
            "no raciocínio clínico. Se não houver predisposições mapeadas, "
            "declare isso explicitamente."
        ),
    )

    insight_limitacoes: str = Field(
        min_length=20,
        description=(
            "Parágrafo 4: limitações do diagnóstico, exames complementares "
            "sugeridos e, se aplicável, menção a evidências da literatura "
            "veterinária consultada (sem colar URLs — elas ficam em "
            "fontes_pesquisadas)."
        ),
    )

    pc_confianca: int = Field(
        ge=0,
        le=100,
        description=(
            "Grau de certeza estimado pela IA (0 a 100%). "
            "Deve refletir honestamente a completude e especificidade dos dados clínicos. "
            "Dados insuficientes ou sintomas inespecíficos devem resultar em valor baixo (<50)."
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

    @field_validator("ds_diagnostico", mode="before")
    @classmethod
    def strip_diagnostico(cls, v: str) -> str:
        """Remove espaços extras e normaliza o título do diagnóstico."""
        return v.strip() if isinstance(v, str) else v

    @field_validator(
        "insight_perfil",
        "insight_correlacao",
        "insight_predisposicao",
        "insight_limitacoes",
        mode="before",
    )
    @classmethod
    def strip_insight_sections(cls, v: str) -> str:
        """Remove espaços extras de cada seção do raciocínio clínico."""
        return v.strip() if isinstance(v, str) else v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ds_insight_ia(self) -> str:
        """
        Concatena as 4 seções estruturadas em um único texto corrido,
        no formato historicamente persistido em TB_ARKIVE_DIAGNOSTICO.DS_INSIGHT_IA.
        Mantém compatibilidade com o serviço Java sem exigir migração de schema.
        """
        return "\n\n".join(
            [
                self.insight_perfil,
                self.insight_correlacao,
                self.insight_predisposicao,
                self.insight_limitacoes,
            ]
        )


class AmbiguityAssessment(BaseModel):
    """
    Avaliação interna de ambiguidade do quadro clínico.

    Usada pelo agente para decidir se a busca web deve ser acionada.
    Não é exposta externamente nem persistida no banco.
    """

    needs_web_search: bool = Field(
        description=(
            "True se os sintomas forem raros, ambíguos, inespecíficos ou insuficientes "
            "para sustentar uma hipótese diagnóstica confiável com base apenas nos dados "
            "locais. Exemplos de casos que exigem busca: espécies exóticas, apresentações "
            "atípicas de doenças comuns, sintomas isolados sem contexto clínico."
        )
    )

    confidence_with_local_data: int = Field(
        ge=0,
        le=100,
        description=(
            "Confiança diagnóstica estimada (0-100%) usando exclusivamente os dados "
            "extraídos do Oracle, sem consulta a fontes externas."
        ),
    )

    suggested_search_query: str = Field(
        default="",
        description=(
            "Query de busca veterinária sugerida em inglês ou português, a ser usada "
            "no DuckDuckGo se needs_web_search for True. "
            "Deve ser específica e incluir espécie, sintomas principais e contexto clínico. "
            "Deixar vazio se needs_web_search for False."
        ),
    )

    reasoning: str = Field(
        description=(
            "Explicação breve (1-3 frases) do motivo pelo qual a busca web é ou "
            "não é necessária neste caso clínico específico."
        )
    )