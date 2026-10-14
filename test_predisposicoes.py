"""python test_predisposicoes.py — corte das predisposições enviadas ao LLM."""

from agents.clinical_agent import ClinicalIntelligenceEngine, _top_predisposicoes

doencas = [{"nm_doenca": f"Doenca {i}", "ds_sintomas": "coceira"} for i in range(50)]
doencas.append({"nm_doenca": "Displasia coxofemoral", "ds_sintomas": "claudicacao dor quadril"})
relato = "Animal com claudicacao na pata traseira"

# lexical (fallback sem RAG)
top = _top_predisposicoes(doencas, relato, 5)
assert len(top) == 5 and top[0]["nm_doenca"] == "Displasia coxofemoral", top
assert top[1:] == doencas[:4]  # empate mantém a ordem da query
assert _top_predisposicoes([], relato, 5) == []

# semântico: só com o índice/modelo do RAG presentes
engine = ClinicalIntelligenceEngine()
if engine._rag:
    pele = [{"nm_doenca": n, "ds_sintomas": s, "ds_doenca": ""} for n, s in (
        ("Dermatite atópica", "prurido coceira lambedura das patas vermelhidão"),
        ("Sarna demodécica", "queda de pelo crostas descamação"),
        ("Otite externa", "coceira na orelha secreção odor"),
        ("Piodermite", "pústulas crostas coceira"),
        ("Seborreia", "descamação pele oleosa odor"),
        ("Alopecia X", "queda de pelo simétrica no tronco"),
    )]
    displasia = {"nm_doenca": "Displasia coxofemoral",
                 "ds_sintomas": "claudicação dos membros posteriores dor no quadril", "ds_doenca": ""}
    sem = engine._rank_predisposicoes(pele + [displasia], "Cão mancando da pata traseira ao levantar", 3)
    assert sem[0] is displasia, [d["nm_doenca"] for d in sem]
    assert len(engine._vetores_doenca) == 7
    engine._rank_predisposicoes(pele + [displasia], "outro relato", 3)
    assert len(engine._vetores_doenca) == 7  # cache: nada re-embedado
print("ok", "(com semântico)" if engine._rag else "(só lexical: RAG indisponível)")
