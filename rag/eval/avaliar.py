"""
rag/eval/avaliar.py
===================
Fase 0 do PLANO_FINETUNING: mede o retriever no conjunto rotulado (casos.jsonl).

- recall@3 e MRR@10: ranking bruto (qualidade do modelo, sem corte de score);
- ruído@3 e vazio correto: top-3 após RAG_MIN_SCORE (o que de fato vai ao LLM);
- gap: score do melhor relevante − score do melhor não relevante no top-10.

Uso:
    python -m rag.eval.avaliar            # compara com baseline.json
    python -m rag.eval.avaliar --salvar   # grava baseline.json
    python -m rag.eval.avaliar --teste    # só o self-check das métricas
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DIR = Path(__file__).parent
CASOS = DIR / "casos.jsonl"
BASELINE = DIR / "baseline.json"


def metricas(casos: list[dict], resultados: list[list[tuple[str, float]]], min_score: float) -> dict:
    """resultados[i] = top-10 [(titulo, score)] do caso i, em ordem."""
    hits, rrs, ruidos, vazios, gaps = [], [], [], [], []
    for caso, top in zip(casos, resultados):
        rel = set(caso["relevantes"])
        cortado = [t for t, s in top[:3] if s >= min_score]
        ruidos.append(sum(t not in rel for t in cortado) / 3)
        if not rel:
            vazios.append(not cortado)
            continue
        titulos = [t for t, _ in top]
        hits.append(any(t in rel for t in titulos[:3]))
        rrs.append(next((1 / (n + 1) for n, t in enumerate(titulos[:10]) if t in rel), 0.0))
        s_rel = [s for t, s in top if t in rel]
        s_out = [s for t, s in top if t not in rel]
        if s_rel and s_out:
            gaps.append(max(s_rel) - max(s_out))
    media = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {"recall@3": media(hits), "mrr@10": media(rrs), "ruido@3": media(ruidos),
            "vazio_correto": media(vazios), "gap": media(gaps), "n_casos": len(casos)}


def _teste() -> None:
    casos = [{"relevantes": ["A"]}, {"relevantes": ["B"]}, {"relevantes": []}]
    res = [[("A", 0.9), ("X", 0.8)], [("X", 0.9), ("Y", 0.8), ("Z", 0.7), ("B", 0.6)], [("X", 0.5)]]
    m = metricas(casos, res, min_score=0.6)
    assert m["recall@3"] == 0.5 and m["mrr@10"] == round((1 + 0.25) / 2, 4), m
    assert m["vazio_correto"] == 1.0, m  # X (0.5) cai no corte
    assert m["ruido@3"] == round((1 / 3 + 1 + 0) / 3, 4), m
    assert m["gap"] == round((0.1 + -0.3) / 2, 4), m
    print("ok")


def main() -> None:
    from config import RAG_MIN_SCORE
    from rag.retriever import RagRetriever

    casos = [json.loads(l) for l in CASOS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rag = RagRetriever()
    titulos_indice = {c["titulo"] for c in rag.chunks.values()}
    for n, caso in enumerate(casos, 1):
        if faltam := set(caso["relevantes"] + caso["ruido"]) - titulos_indice:
            print(f"AVISO caso {n}: títulos fora do índice {sorted(faltam)}")

    resultados = []
    for n, caso in enumerate(casos, 1):
        top = [(d["titulo"], d["score"]) for d in rag.buscar(caso["relato"], caso["especie"], k=10)]
        resultados.append(top)
        rel, cortado = set(caso["relevantes"]), [t for t, s in top[:3] if s >= RAG_MIN_SCORE]
        falhou = (rel and not any(t in rel for t, _ in top[:3])) or (not rel and cortado) \
            or set(caso["ruido"]) & set(cortado)
        if falhou:
            print(f"\nFALHA {n} [{caso['especie']}] {caso['relato'][:70]}...")
            print(f"  esperado: {sorted(rel) or 'vazio'}")
            print("  top-3:", "; ".join(f"{t} ({s:.3f})" for t, s in top[:3]))

    m = metricas(casos, resultados, RAG_MIN_SCORE)
    m["min_score"] = RAG_MIN_SCORE
    print("\n" + json.dumps(m, indent=2))
    if "--salvar" in sys.argv:
        BASELINE.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
        print(f"baseline gravado em {BASELINE}")
    elif BASELINE.exists():
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
        print("\ndelta vs baseline:")
        for k, v in m.items():
            if isinstance(v, float) and isinstance(base.get(k), float):
                print(f"  {k}: {base[k]:.4f} -> {v:.4f} ({v - base[k]:+.4f})")


if __name__ == "__main__":
    _teste() if "--teste" in sys.argv else main()
