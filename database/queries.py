"""6 SELECTs parametrizados (somente leitura) e o ClinicalContext que os agrega."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import oracledb

from config import (
    DIAGNOSTIC_HISTORY_LIMIT,
    HISTORICO_CUIDADO_LIMIT,
    MAX_TRANSCRICAO_CHARS,
    PREDISPOSICAO_DESC_CHARS,
    RELATO_RESPONSAVEL_CHARS,
    RELATO_RESPONSAVEL_LIMIT,
)

logger = logging.getLogger(__name__)

# LEFT JOIN LATERAL exige Oracle 12c+.
CLINICAL_DATA_QUERY: str = """
SELECT
    a.ID_ANIMAL,
    a.NM_ANIMAL,
    a.DS_SEXO,
    a.DS_CASTRADO,
    a.DT_NASCIMENTO,

    e.ID_ESPECIE,
    e.NM_ESPECIE,

    r.ID_RACA,
    r.NM_RACA,
    r.TP_PORTE,

    c.ID_CONSULTA,
    c.DT_HORA,
    c.TP_MODALIDADE,
    c.DS_MOTIVO,
    c.DS_SINTOMAS,
    c.DS_OBSERVACAO    AS DS_OBS_CONSULTA,
    c.DS_TRANSCRICAO,
    c.DS_OBS_RESPONSAVEL,
    c.KG_PESO         AS KG_PESO_CONSULTA,

    abe.NR_IDADE,
    abe.KG_PESO        AS KG_PESO_BEM_ESTAR,
    abe.DS_APETITE,
    abe.DS_ATIVIDADE,
    abe.DS_COMPORTAMENTO,
    abe.DS_OBSERVACAO  AS DS_OBS_BEM_ESTAR

FROM       TB_ARKIVE_CONSULTA             c
JOIN       TB_ARKIVE_ANIMAL               a   ON a.ID_ANIMAL  = c.ID_ANIMAL
JOIN       TB_ARKIVE_ESPECIE              e   ON e.ID_ESPECIE = a.ID_ESPECIE
                                              AND e.ST_ATIVO   = 'S'
-- ST_ATIVO no ON, não no WHERE: senão vira INNER JOIN e some o animal SRD
LEFT JOIN  TB_ARKIVE_RACA                 r   ON r.ID_RACA    = a.ID_RACA
                                              AND r.ST_ATIVO   = 'S'

-- avaliação desta consulta primeiro, senão a mais recente do animal
LEFT JOIN LATERAL (
    SELECT
        NR_IDADE,
        KG_PESO,
        DS_APETITE,
        DS_ATIVIDADE,
        DS_COMPORTAMENTO,
        DS_OBSERVACAO
    FROM   TB_ARKIVE_AVALIACAO_BEM_ESTAR
    WHERE  ID_ANIMAL = a.ID_ANIMAL
    ORDER BY
        CASE WHEN ID_CONSULTA = :id_consulta THEN 0 ELSE 1 END,
        ID_AVALIACAO_BEM_ESTAR DESC
    FETCH FIRST 1 ROW ONLY
) abe ON 1 = 1

WHERE c.ID_CONSULTA = :id_consulta
"""

# Predisposição de espécie (ID_RACA nulo) vale sempre; a de raça, só para aquela raça exata.
PREDISPOSITION_QUERY: str = """
SELECT DISTINCT
    d.NM_DOENCA,
    d.DS_DOENCA,
    d.DS_SINTOMAS

FROM   TB_ARKIVE_DOENCA d
WHERE  d.ST_ATIVO = 'S'
  AND  d.ID_DOENCA IN (
    SELECT p.ID_DOENCA
    FROM   TB_ARKIVE_PREDISPOSICAO p
    WHERE  p.ID_ESPECIE = :id_especie
      AND  (
               p.ID_RACA IS NULL
            OR (:id_raca IS NOT NULL AND p.ID_RACA = :id_raca)
           )
)
ORDER BY d.NM_DOENCA
"""

# Diagnósticos de outras consultas do mesmo animal.
DIAGNOSTIC_HISTORY_QUERY: str = """
SELECT
    c2.DT_HORA,
    dg.DS_DIAGNOSTICO,
    dg.TP_SEVERIDADE,
    dg.PC_CONFIANCA,
    dg.ST_CONFIRMADO,
    dg.ST_VALIDACAO_VET,
    doe.NM_DOENCA

FROM       TB_ARKIVE_DIAGNOSTICO dg
JOIN       TB_ARKIVE_CONSULTA    c2  ON c2.ID_CONSULTA = dg.ID_CONSULTA
LEFT JOIN  TB_ARKIVE_DOENCA      doe ON doe.ID_DOENCA  = dg.ID_DOENCA

WHERE  c2.ID_ANIMAL   = :id_animal
  AND  c2.ID_CONSULTA != :id_consulta

ORDER BY c2.DT_HORA DESC
FETCH FIRST :limit ROWS ONLY
"""

# Prescrição liga em ID_CONSULTA, não em ID_ANIMAL. Adesão por subquery escalar para não multiplicar linhas.
PRESCRIPTION_HISTORY_QUERY: str = """
SELECT
    p.NM_MEDICAMENTO,
    p.DS_DOSAGEM,
    p.DS_FREQUENCIA,
    p.TP_VIA_ADMINISTRACAO,
    p.DT_INICIO,
    p.DT_FIM,
    p.DS_INSTRUCOES,
    c.DT_HORA AS DT_CONSULTA,
    (SELECT COUNT(*)
       FROM TB_ARKIVE_ADESAO_PRESCRICAO ad
      WHERE ad.ID_PRESCRICAO = p.ID_PRESCRICAO
        AND ad.ST_TOMOU = 'S')   AS QT_ADESAO_SIM,
    (SELECT COUNT(*)
       FROM TB_ARKIVE_ADESAO_PRESCRICAO ad
      WHERE ad.ID_PRESCRICAO = p.ID_PRESCRICAO
        AND ad.ST_TOMOU = 'N')   AS QT_ADESAO_NAO

FROM       TB_ARKIVE_PRESCRICAO p
JOIN       TB_ARKIVE_CONSULTA   c  ON c.ID_CONSULTA = p.ID_CONSULTA

WHERE  c.ID_ANIMAL = :id_animal

ORDER BY p.DT_INICIO DESC
FETCH FIRST :limit ROWS ONLY
"""

# ATRASADO e PENDENTE primeiro.
PREVENTIVE_STATUS_QUERY: str = """
SELECT
    pp.NM_PROTOCOLO,
    pp.TP_PROTOCOLO,
    ep.ST_STATUS,
    ep.DT_APLICACAO,
    ep.DT_PROXIMO

FROM       TB_ARKIVE_EVENTO_PREVENTIVO     ep
JOIN       TB_ARKIVE_PROTOCOLO_PREVENTIVO  pp  ON pp.ID_PROTOCOLO = ep.ID_PROTOCOLO

WHERE  ep.ID_ANIMAL = :id_animal

ORDER BY
    CASE ep.ST_STATUS
        WHEN 'ATRASADO' THEN 0
        WHEN 'PENDENTE' THEN 1
        ELSE 2
    END,
    ep.DT_PROXIMO
FETCH FIRST :limit ROWS ONLY
"""

# Relatos do responsável sobre o animal. Sem data de ocorrência, ordena pela de registro.
OWNER_REPORTS_QUERY: str = """
SELECT
    ra.DS_RELATO,
    ra.DT_OCORRENCIA,
    ra.DT_CRIACAO,
    ra.ST_ORIGEM_TRANSCRICAO,
    r.NM_RESPONSAVEL,
    r.TP_RESPONSAVEL

FROM       TB_ARKIVE_RELATO_ANIMAL  ra
LEFT JOIN  TB_ARKIVE_RESPONSAVEL    r   ON r.ID_RESPONSAVEL = ra.ID_RESPONSAVEL_AUTOR

WHERE  ra.ID_ANIMAL = :id_animal

ORDER BY COALESCE(ra.DT_OCORRENCIA, CAST(ra.DT_CRIACAO AS DATE)) DESC, ra.ID_RELATO DESC
FETCH FIRST :limit ROWS ONLY
"""

# Delimitam o texto do responsável no prompt: o que está entre eles é dado, nunca instrução.
_RELATO_INI, _RELATO_FIM = "<<<RELATO>>>", "<<<FIM_RELATO>>>"

@dataclass
class ClinicalContext:
    """Contrato entre a camada de banco e a de IA."""

    id_animal: int = 0
    nm_animal: str = ""
    ds_sexo: str = ""
    ds_castrado: str = ""
    dt_nascimento: datetime | None = None

    id_especie: int | None = None
    nm_especie: str = ""

    id_raca: int | None = None
    nm_raca: str = ""
    tp_porte: str = ""

    id_consulta: int = 0
    dt_hora: datetime | None = None
    tp_modalidade: str = ""
    ds_motivo: str = ""
    ds_sintomas: str = ""
    ds_obs_consulta: str = ""
    ds_transcricao: str = ""
    ds_obs_responsavel: str = ""
    kg_peso_consulta: float | None = None

    nr_idade: float | None = None
    kg_peso_bem_estar: float | None = None
    ds_apetite: str = ""
    ds_atividade: str = ""
    ds_comportamento: str = ""
    ds_obs_bem_estar: str = ""

    # cada lista guarda as colunas da query correspondente, em minúsculas
    predisposicoes: list[dict[str, str]] = field(default_factory=list)
    diagnosticos_anteriores: list[dict[str, Any]] = field(default_factory=list)
    prescricoes: list[dict[str, Any]] = field(default_factory=list)
    eventos_preventivos: list[dict[str, Any]] = field(default_factory=list)
    relatos_responsaveis: list[dict[str, Any]] = field(default_factory=list)

    @property
    def peso_efetivo_kg(self) -> float | None:
        """Peso da consulta, senão o da avaliação de bem-estar."""
        return self.kg_peso_consulta or self.kg_peso_bem_estar

    @property
    def idade_anos(self) -> float | None:
        """DT_NASCIMENTO até a consulta (ou hoje); NR_IDADE se não houver data válida."""
        if self.dt_nascimento:
            ref = self.dt_hora or datetime.now()
            dias = (ref - self.dt_nascimento).days
            if dias >= 0:
                return dias / 365.25
        return self.nr_idade

    def to_clinical_summary(self) -> str:
        """Resumo clínico em texto para o prompt."""
        _SEXO = {"M": "Macho", "F": "Fêmea"}
        _CASTRADO = {"S": "Castrado(a)", "N": "Inteiro(a)"}

        if self.predisposicoes:
            pred_lines = []
            for doenca in self.predisposicoes:
                nm = doenca.get("nm_doenca", "Desconhecida")
                ds = doenca.get("ds_doenca") or "Sem descrição disponível."
                pred_lines.append(f"  • {nm}: {_cortar(ds, PREDISPOSICAO_DESC_CHARS)}")
                sintomas_doenca = (doenca.get("ds_sintomas") or "").strip()
                if sintomas_doenca:
                    pred_lines.append(
                        "      Palavras-chave clínicas associadas: "
                        f"{_cortar(sintomas_doenca, PREDISPOSICAO_DESC_CHARS)}"
                    )
            predisposicoes_block = "\n".join(pred_lines)
        else:
            predisposicoes_block = "  Nenhuma predisposição genética mapeada para esta raça/espécie."

        transcricao = (self.ds_transcricao or "").strip()
        if transcricao:
            transcricao_block = (
                transcricao[:MAX_TRANSCRICAO_CHARS] + "... [transcrição truncada]"
                if len(transcricao) > MAX_TRANSCRICAO_CHARS
                else transcricao
            )
        else:
            transcricao_block = "  Transcrição de voz não disponível para esta consulta."

        welfare_items = {
            "Apetite": self.ds_apetite,
            "Atividade": self.ds_atividade,
            "Comportamento": self.ds_comportamento,
        }
        welfare_lines = [
            f"  {k}: {v}" for k, v in welfare_items.items() if v
        ] or ["  Avaliação de bem-estar não registrada nesta consulta."]

        peso_str = f"{self.peso_efetivo_kg:.2f} kg" if self.peso_efetivo_kg else "Não informado"
        dt_str = self.dt_hora.strftime("%d/%m/%Y %H:%M") if self.dt_hora else "Não informada"

        nascimento_str = (
            self.dt_nascimento.strftime("%d/%m/%Y")
            if hasattr(self.dt_nascimento, "strftime")
            else "Não informada"
        )
        idade = self.idade_anos
        idade_str = f"{idade:.1f} anos" if idade else "Não informada"
        idade_label = "Idade" if self.dt_nascimento else "Idade estimada"

        if self.diagnosticos_anteriores:
            _CONFIRMADO = {"S": "confirmado", "N": "não confirmado"}
            _VALIDACAO_VET = {
                "S": "validado pelo veterinário",
                "N": "não validado pelo veterinário",
            }
            hist_lines = []
            for diag in self.diagnosticos_anteriores:
                dt_diag = diag.get("dt_hora")
                dt_diag_str = (
                    dt_diag.strftime("%d/%m/%Y") if hasattr(dt_diag, "strftime") else "Data não informada"
                )
                nm_doenca = diag.get("nm_doenca")
                titulo = nm_doenca or diag.get("ds_diagnostico") or "Diagnóstico sem título"
                severidade = diag.get("tp_severidade") or "Não informada"
                confianca = diag.get("pc_confianca")
                confianca_str = f"{confianca:.0f}%" if confianca is not None else "N/A"
                status_conf = _CONFIRMADO.get(diag.get("st_confirmado"), "status desconhecido")
                # ST_VALIDACAO_VET (nullable): o vet validou o insight da IA, não o diagnóstico
                st_validacao = diag.get("st_validacao_vet")
                status_validacao = _VALIDACAO_VET.get(
                    st_validacao, "insight de IA não avaliado pelo veterinário"
                )
                hist_lines.append(
                    f"  • [{dt_diag_str}] {titulo} — Severidade: {severidade} "
                    f"| Confiança à época: {confianca_str} | {status_conf} "
                    f"| {status_validacao}"
                )
            historico_block = "\n".join(hist_lines)
        else:
            historico_block = "  Nenhum diagnóstico anterior registrado para este animal."

        if self.prescricoes:
            presc_lines = []
            for presc in self.prescricoes:
                nm = presc.get("nm_medicamento") or "Medicamento não informado"
                dose = presc.get("ds_dosagem") or "dose não informada"
                freq = presc.get("ds_frequencia")
                via = presc.get("tp_via_administracao")
                dt_ini = presc.get("dt_inicio")
                dt_fim = presc.get("dt_fim")
                dt_ini_str = dt_ini.strftime("%d/%m/%Y") if hasattr(dt_ini, "strftime") else "início não informado"
                dt_fim_str = dt_fim.strftime("%d/%m/%Y") if hasattr(dt_fim, "strftime") else "sem término previsto"
                detalhes = [f"{dose}"]
                if freq:
                    detalhes.append(freq)
                if via:
                    detalhes.append(f"via {via}")
                sim = presc.get("qt_adesao_sim") or 0
                nao = presc.get("qt_adesao_nao") or 0
                if sim or nao:
                    adesao_str = f" | adesão: seguida {sim}x, não seguida {nao}x"
                else:
                    adesao_str = " | adesão não registrada"
                presc_lines.append(
                    f"  • {nm} ({', '.join(detalhes)}) — {dt_ini_str} a {dt_fim_str}{adesao_str}"
                )
                instrucoes = (presc.get("ds_instrucoes") or "").strip()
                if instrucoes:
                    presc_lines.append(f"      Instruções: {instrucoes}")
            medicacoes_block = "\n".join(presc_lines)
        else:
            medicacoes_block = "  Nenhuma prescrição registrada para este animal."

        if self.eventos_preventivos:
            prev_lines = []
            for evt in self.eventos_preventivos:
                nm = evt.get("nm_protocolo") or "Protocolo não informado"
                tipo = evt.get("tp_protocolo") or "tipo não informado"
                status = (evt.get("st_status") or "status desconhecido").upper()
                dt_prox = evt.get("dt_proximo")
                dt_prox_str = dt_prox.strftime("%d/%m/%Y") if hasattr(dt_prox, "strftime") else "data não informada"
                destaque = " ⚠️ ATRASADO" if status == "ATRASADO" else ""
                prev_lines.append(
                    f"  • {nm} [{tipo}] — status: {status} | próxima data prevista: {dt_prox_str}{destaque}"
                )
            preventivo_block = "\n".join(prev_lines)
        else:
            preventivo_block = "  Nenhum evento de cuidado preventivo registrado para este animal."

        relato_lines = []
        if self.ds_obs_responsavel.strip():
            relato_lines += [
                "  Observação do responsável para esta consulta:",
                f"    {_relato_seguro(self.ds_obs_responsavel)}",
            ]
        _ORIGEM = {"S": "áudio transcrito", "N": "texto"}
        for rel in self.relatos_responsaveis:
            dt_oc, dt_cr = rel.get("dt_ocorrencia"), rel.get("dt_criacao")
            dt_oc_str = dt_oc.strftime("%d/%m/%Y") if hasattr(dt_oc, "strftime") else "não informada"
            dt_cr_str = dt_cr.strftime("%d/%m/%Y") if hasattr(dt_cr, "strftime") else "data não informada"
            autor = rel.get("nm_responsavel") or "responsável não identificado"
            if rel.get("tp_responsavel"):
                autor += f" ({rel['tp_responsavel']})"
            origem = _ORIGEM.get(rel.get("st_origem_transcricao"), "origem não informada")
            relato_lines += [
                f"  • Ocorrência: {dt_oc_str} | registrado em {dt_cr_str} | autor: {autor} | origem: {origem}",
                f"    {_relato_seguro(rel.get('ds_relato') or '')}",
            ]
        relatos_block = "\n".join(relato_lines) or "  Nenhum relato do responsável registrado para este animal."

        return (
            "=== DADOS DO PACIENTE ===\n"
            f"Nome:           {self.nm_animal}\n"
            f"Espécie:        {self.nm_especie}\n"
            f"Raça:           {self.nm_raca or 'SRD / Não informada'}"
            f" | Porte: {self.tp_porte or 'Não informado'}\n"
            f"Sexo:           {_SEXO.get(self.ds_sexo, self.ds_sexo)}\n"
            f"Status reprod.: {_CASTRADO.get(self.ds_castrado, self.ds_castrado)}\n"
            f"Nascimento:     {nascimento_str}\n"
            f"{(idade_label + ':').ljust(15)} {idade_str}\n"
            f"Peso:           {peso_str}\n"
            "\n=== RELATO CLÍNICO DO VETERINÁRIO (TRANSCRIÇÃO DA CONSULTA) ===\n"
            + transcricao_block
            + "\n\n=== DADOS DA CONSULTA ===\n"
            f"ID Consulta:    {self.id_consulta}\n"
            f"Data/Hora:      {dt_str}\n"
            f"Modalidade:     {self.tp_modalidade}\n"
            f"Motivo:         {self.ds_motivo or 'Não informado'}\n"
            f"Sintomas:       {self.ds_sintomas or 'Não descritos'}\n"
            f"Observações:    {self.ds_obs_consulta or 'Sem observações'}\n"
            "\n=== RELATOS DOS RESPONSÁVEIS (informação do tutor, NÃO confirmada pelo veterinário) ===\n"
            + relatos_block
            + "\n"
            "\n=== AVALIAÇÃO DE BEM-ESTAR (mais recente) ===\n"
            + "\n".join(welfare_lines)
            + (f"\n  Observações: {self.ds_obs_bem_estar}" if self.ds_obs_bem_estar else "")
            + "\n\n=== PREDISPOSIÇÕES GENÉTICAS DA RAÇA/ESPÉCIE ===\n"
            + predisposicoes_block
            + "\n\n=== HISTÓRICO DE DIAGNÓSTICOS ANTERIORES (outras consultas) ===\n"
            + historico_block
            + "\n\n=== MEDICAÇÕES / PRESCRIÇÕES RECENTES ===\n"
            + medicacoes_block
            + "\n\n=== CUIDADO PREVENTIVO (VACINAS / VERMÍFUGO / CHECK-UP) ===\n"
            + preventivo_block
        )


def fetch_clinical_data(conn: oracledb.Connection, id_consulta: int) -> ClinicalContext:
    """
    Raises:
        ValueError: Nenhuma consulta encontrada para o ID.
        oracledb.DatabaseError: Erro de banco de dados.
    """
    ctx = ClinicalContext(id_consulta=id_consulta)

    logger.info("Executando CLINICAL_DATA_QUERY para ID_CONSULTA=%d", id_consulta)
    with conn.cursor() as cur:
        cur.execute(CLINICAL_DATA_QUERY, {"id_consulta": id_consulta})
        columns: list[str] = [col[0].lower() for col in cur.description]
        row: tuple[Any, ...] | None = cur.fetchone()

    if row is None:
        raise ValueError(
            f"Nenhuma consulta encontrada para ID_CONSULTA={id_consulta}. "
            "Verifique se o registro existe e se o usuário possui acesso SELECT."
        )

    row_dict: dict[str, Any] = dict(zip(columns, row))
    logger.debug("Dados clínicos retornados: %s", list(row_dict.keys()))

    ctx.id_animal = int(row_dict["id_animal"] or 0)
    ctx.nm_animal = str(row_dict.get("nm_animal") or "")
    ctx.ds_sexo = str(row_dict.get("ds_sexo") or "")
    ctx.ds_castrado = str(row_dict.get("ds_castrado") or "")
    ctx.dt_nascimento = row_dict.get("dt_nascimento")

    ctx.id_especie = row_dict.get("id_especie")
    ctx.nm_especie = str(row_dict.get("nm_especie") or "")

    ctx.id_raca = row_dict.get("id_raca")
    ctx.nm_raca = str(row_dict.get("nm_raca") or "")
    ctx.tp_porte = str(row_dict.get("tp_porte") or "")

    ctx.dt_hora = row_dict.get("dt_hora")
    ctx.tp_modalidade = str(row_dict.get("tp_modalidade") or "")
    ctx.ds_motivo = str(row_dict.get("ds_motivo") or "")
    ctx.ds_sintomas = str(row_dict.get("ds_sintomas") or "")
    ctx.ds_obs_consulta = str(row_dict.get("ds_obs_consulta") or "")
    ctx.ds_transcricao = str(row_dict.get("ds_transcricao") or "")
    ctx.ds_obs_responsavel = str(row_dict.get("ds_obs_responsavel") or "")
    ctx.kg_peso_consulta = _safe_float(row_dict.get("kg_peso_consulta"))

    ctx.nr_idade = _safe_float(row_dict.get("nr_idade"))
    ctx.kg_peso_bem_estar = _safe_float(row_dict.get("kg_peso_bem_estar"))
    ctx.ds_apetite = str(row_dict.get("ds_apetite") or "")
    ctx.ds_atividade = str(row_dict.get("ds_atividade") or "")
    ctx.ds_comportamento = str(row_dict.get("ds_comportamento") or "")
    ctx.ds_obs_bem_estar = str(row_dict.get("ds_obs_bem_estar") or "")

    if ctx.id_especie is not None:
        logger.info(
            "Executando PREDISPOSITION_QUERY | ID_ESPECIE=%s | ID_RACA=%s",
            ctx.id_especie,
            ctx.id_raca,
        )
        with conn.cursor() as cur:
            cur.execute(
                PREDISPOSITION_QUERY,
                {"id_especie": ctx.id_especie, "id_raca": ctx.id_raca},
            )
            pred_columns = [col[0].lower() for col in cur.description]
            for pred_row in cur.fetchall():
                pred_dict = dict(zip(pred_columns, pred_row))
                ctx.predisposicoes.append(
                    {k: str(v) if v is not None else "" for k, v in pred_dict.items()}
                )

        logger.info(
            "%d predisposição(ões) genética(s) encontrada(s) para a raça/espécie.",
            len(ctx.predisposicoes),
        )
    else:
        logger.warning("ID_ESPECIE é NULL — pulando query de predisposições.")

    if ctx.id_animal:
        logger.info(
            "Executando DIAGNOSTIC_HISTORY_QUERY | ID_ANIMAL=%s | limit=%d",
            ctx.id_animal,
            DIAGNOSTIC_HISTORY_LIMIT,
        )
        with conn.cursor() as cur:
            cur.execute(
                DIAGNOSTIC_HISTORY_QUERY,
                {
                    "id_animal": ctx.id_animal,
                    "id_consulta": id_consulta,
                    "limit": DIAGNOSTIC_HISTORY_LIMIT,
                },
            )
            hist_columns = [col[0].lower() for col in cur.description]
            for hist_row in cur.fetchall():
                ctx.diagnosticos_anteriores.append(dict(zip(hist_columns, hist_row)))

        logger.info(
            "%d diagnóstico(s) anterior(es) encontrado(s) para o animal.",
            len(ctx.diagnosticos_anteriores),
        )

    if ctx.id_animal:
        logger.info(
            "Executando PRESCRIPTION_HISTORY_QUERY | ID_ANIMAL=%s | limit=%d",
            ctx.id_animal,
            HISTORICO_CUIDADO_LIMIT,
        )
        with conn.cursor() as cur:
            cur.execute(
                PRESCRIPTION_HISTORY_QUERY,
                {"id_animal": ctx.id_animal, "limit": HISTORICO_CUIDADO_LIMIT},
            )
            presc_columns = [col[0].lower() for col in cur.description]
            for presc_row in cur.fetchall():
                ctx.prescricoes.append(dict(zip(presc_columns, presc_row)))

        logger.info(
            "%d prescrição(ões) recente(s) encontrada(s) para o animal.",
            len(ctx.prescricoes),
        )

    if ctx.id_animal:
        logger.info(
            "Executando PREVENTIVE_STATUS_QUERY | ID_ANIMAL=%s | limit=%d",
            ctx.id_animal,
            HISTORICO_CUIDADO_LIMIT,
        )
        with conn.cursor() as cur:
            cur.execute(
                PREVENTIVE_STATUS_QUERY,
                {"id_animal": ctx.id_animal, "limit": HISTORICO_CUIDADO_LIMIT},
            )
            prev_columns = [col[0].lower() for col in cur.description]
            for prev_row in cur.fetchall():
                ctx.eventos_preventivos.append(dict(zip(prev_columns, prev_row)))

        logger.info(
            "%d evento(s) de cuidado preventivo encontrado(s) para o animal.",
            len(ctx.eventos_preventivos),
        )

    if ctx.id_animal:
        logger.info(
            "Executando OWNER_REPORTS_QUERY | ID_ANIMAL=%s | limit=%d",
            ctx.id_animal,
            RELATO_RESPONSAVEL_LIMIT,
        )
        with conn.cursor() as cur:
            # +1: cobre o relato igual à observação da consulta, descartado em _dedup_relatos
            cur.execute(
                OWNER_REPORTS_QUERY,
                {"id_animal": ctx.id_animal, "limit": RELATO_RESPONSAVEL_LIMIT + 1},
            )
            rel_columns = [col[0].lower() for col in cur.description]
            relatos = []
            for rel_row in cur.fetchall():
                rel = dict(zip(rel_columns, rel_row))
                rel["ds_relato"] = str(rel.get("ds_relato") or "")  # CLOB
                relatos.append(rel)
        ctx.relatos_responsaveis = _dedup_relatos(relatos, ctx.ds_obs_responsavel, RELATO_RESPONSAVEL_LIMIT)

        logger.info(
            "%d relato(s) do responsável encontrado(s) para o animal.",
            len(ctx.relatos_responsaveis),
        )

    return ctx


def _normalizar(texto: str) -> str:
    return " ".join(texto.split()).casefold()


def _dedup_relatos(relatos: list[dict[str, Any]], obs_consulta: str, limite: int) -> list[dict[str, Any]]:
    """Tira o relato igual à observação já vinculada à consulta (ela vai no prompt à parte)."""
    obs = _normalizar(obs_consulta)
    return [r for r in relatos if not obs or _normalizar(r.get("ds_relato") or "") != obs][:limite]


def _relato_seguro(texto: str) -> str:
    """Uma linha, sem os delimitadores: o texto do responsável não fecha o bloco nem imita um cabeçalho."""
    limpo = " ".join(texto.replace(_RELATO_INI, " ").replace(_RELATO_FIM, " ").split())
    return f"{_RELATO_INI}{_cortar(limpo, RELATO_RESPONSAVEL_CHARS)}{_RELATO_FIM}"


def _cortar(texto: str, limite: int) -> str:
    """DS_DOENCA é CLOB e DS_SINTOMAS não tem limite: sem corte, cada doença pesa no prompt."""
    return texto[:limite] + "..." if len(texto) > limite else texto


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None