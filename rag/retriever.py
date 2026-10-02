"""
rag/retriever.py
================
Busca semântica na base de conhecimento veterinária (fichas WOAH, AAZV,
CFSPH, CAPC, ABCD, ESCCAP) indexada por rag/build_index.ipynb.

Mesma lógica da seção "11. Busca" do notebook — qualquer mudança em
embedding, MAPA_ESPECIE ou agregação deve ser feita nos dois lugares.

Uso:
    python -m rag.retriever             # teste com um relato de exemplo
    python -m rag.retriever --download  # só baixa o modelo (build do Render)
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from config import RAG_INDEX_DIR, RAG_MODEL_DIR

# Precisam bater com o notebook (cell "2. Configuração") — senão o índice é inválido.
MODEL_NAME = "Xenova/multilingual-e5-small"
MODEL_FILE = "onnx/model_quantized.onnx"
DIM = 384
QUERY_WINDOW_CHARS = 1500

CATEGORIAS_SEMPRE = {"Multiple Species", "Other Diseases"}
MAPA_ESPECIE = {
    "bovin": {"Bovinae"}, "vaca": {"Bovinae"}, "bufal": {"Bovinae"},
    "ovin": {"Caprinae"}, "ovelha": {"Caprinae"}, "caprin": {"Caprinae"}, "cabra": {"Caprinae"},
    "equin": {"Equidae"}, "cavalo": {"Equidae"}, "egua": {"Equidae"}, "asin": {"Equidae"}, "mul": {"Equidae"},
    "suin": {"Suidae"}, "porc": {"Suidae"},
    "ave": {"Aves"}, "galinha": {"Aves"}, "frango": {"Aves"}, "canari": {"Aves"}, "pato": {"Aves"}, "peru": {"Aves"},
    "coelho": {"Leporidae"}, "lebre": {"Leporidae"},
    "camel": {"Camelidae"}, "dromedario": {"Camelidae"},
    "abelha": {"Apinae"},
    "peixe": {"Fish"}, "tilapia": {"Fish"}, "salmao": {"Fish"}, "carpa": {"Fish"},
    "camarao": {"Crustaceans"}, "crustace": {"Crustaceans"}, "lagost": {"Crustaceans"},
    "ostra": {"Molluscs"}, "molusc": {"Molluscs"}, "mexilhao": {"Molluscs"},
    "anfibi": {"Amphibians"}, "sapo": {"Amphibians"}, "ra": {"Amphibians"},
    "canin": {"Canidae"}, "cao": {"Canidae"}, "cachorro": {"Canidae"},
    "felin": {"Felidae"}, "gato": {"Felidae"},
    "cervo": {"Cervidae"}, "veado": {"Cervidae"}, "cervid": {"Cervidae"},
    "morcego": {"Chiroptera"},
    "elefant": {"Elephantidae"},
    "canguru": {"Macropodidae"}, "walab": {"Macropodidae"},
    "golfinho": {"Marine Mammals"}, "foca": {"Marine Mammals"}, "baleia": {"Marine Mammals"},
    "furao": {"Mustelidae"}, "mustel": {"Mustelidae"},
    "primata": {"Primates"}, "macaco": {"Primates"}, "sagui": {"Primates"},
    "reptil": {"Reptilia"}, "serpente": {"Reptilia"}, "cobra": {"Reptilia"}, "lagarto": {"Reptilia"},
    "jabuti": {"Reptilia"}, "tartaruga": {"Reptilia"}, "iguana": {"Reptilia"},
    "roedor": {"Rodentia"}, "hamster": {"Rodentia"}, "rato": {"Rodentia"}, "camundongo": {"Rodentia"},
    "chinchila": {"Rodentia"}, "porquinho": {"Rodentia"},
}


# Registro único por processo (registrar de novo levanta ValueError).
TextEmbedding.add_custom_model(
    model=MODEL_NAME,
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf=MODEL_NAME),
    dim=DIM,
    model_file=MODEL_FILE,
)


def carregar_modelo(cache_dir: Path = RAG_MODEL_DIR) -> TextEmbedding:
    """Modelo e5 int8. cache_dir fixo no projeto para o build do Render deixá-lo pronto."""
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=str(cache_dir))


def fatiar(texto: str, tam: int, overlap: int = 200) -> list[str]:
    """Janelas de ~tam chars, cortando em fim de frase quando possível."""
    texto = re.sub(r"\s+", " ", texto).strip()
    out, i = [], 0
    while i < len(texto):
        fim = min(i + tam, len(texto))
        if fim < len(texto):
            corte = texto.rfind(". ", i + tam // 2, fim)
            if corte != -1:
                fim = corte + 1
        out.append(texto[i:fim].strip())
        if fim >= len(texto):
            break
        prox = texto.find(" ", fim - overlap)
        i = prox + 1 if i < prox < fim else fim
    return out


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def categorias_da_especie(especie: str | None) -> set[str] | None:
    """None = sem filtro (espécie vazia ou desconhecida)."""
    if not especie:
        return None
    palavras = re.findall(r"[a-z]+", _norm(especie))
    cats, achou = set(CATEGORIAS_SEMPRE), False
    for chave, valor in MAPA_ESPECIE.items():
        # chaves curtas ("ra", "ave") exigem palavra inteira/prefixo, não substring solta
        if any(p == chave or (len(chave) >= 3 and p.startswith(chave)) for p in palavras):
            cats |= valor
            achou = True
    return cats if achou else None


def montar_contexto_rag(top: list[dict], max_chars: int = 3000) -> str:
    """Bloco de texto para o prompt do Groq. Orçamento dividido igualmente entre os documentos."""
    if not top:
        return ""
    por_doc = max_chars // len(top)
    partes = ["BASE DE CONHECIMENTO VETERINÁRIA (RAG — WOAH, AAZV, CFSPH, CAPC, ABCD, ESCCAP; em inglês). "
              "Documentos mais relevantes para o relato (fichas de doença ou guias clínicos), "
              "por similaridade semântica (não é probabilidade):"]
    for n, d in enumerate(top, 1):
        cab = f"\n[{n}] {d['titulo']} ({d['categoria']}, fonte: {d['fonte'] or 'n/d'}, relevância {d['score']:.2f})\n"
        corpo = " […] ".join([d["resumo"]["texto"]] + [f"({c['secao']}) {c['texto']}" for c in d["trechos"]])
        partes.append(cab + corpo[: por_doc - len(cab)])
    return "\n".join(partes)


class RagRetriever:
    """Índice FAISS + chunks + modelo carregados uma vez; buscar() por consulta."""

    def __init__(self, index_dir: Path = RAG_INDEX_DIR) -> None:
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"]["model"] != MODEL_NAME:
            raise RuntimeError(f"Índice gerado com {manifest['config']['model']}, esperado {MODEL_NAME}")
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = {c["id"]: c for c in map(json.loads, f)}
        # resumo = primeiro chunk do arquivo (SUMMARY / Scope)
        self.resumos = {c["arquivo"]: c for c in self.chunks.values() if c["ordem"] == 0}
        self.modelo = carregar_modelo()

    def embed(self, textos: list[str]) -> np.ndarray:
        """Vetores float32 normalizados (produto interno = cosseno)."""
        return np.array(list(self.modelo.embed(textos)), dtype="float32")

    def buscar(self, texto: str, especie: str | None = None, k: int = 3, k_chunks: int = 100) -> list[dict]:
        """Top-k documentos distintos (fonte + título). Score = maior similaridade chunk × janela do relato."""
        janelas = fatiar(texto, QUERY_WINDOW_CHARS) or [texto]
        scores, ids = self.index.search(self.embed([f"query: {j}" for j in janelas]),
                                        min(k_chunks, self.index.ntotal))
        permitidas = categorias_da_especie(especie)

        melhor_por_chunk: dict[int, float] = {}
        for linha_s, linha_i in zip(scores, ids):
            for s, i in zip(linha_s, linha_i):
                if i != -1 and s > melhor_por_chunk.get(int(i), -1):
                    melhor_por_chunk[int(i)] = float(s)

        docs: dict[tuple[str, str], dict] = {}
        for i, s in sorted(melhor_por_chunk.items(), key=lambda x: -x[1]):
            c = self.chunks[i]
            if permitidas is not None and c["categoria"] not in permitidas:
                continue
            # (fonte, titulo): guias ESCCAP repetidos em Canidae/Felidae contam uma vez só
            d = docs.setdefault((c["fonte"], c["titulo"]), {"titulo": c["titulo"], "categoria": c["categoria"],
                                                             "fonte": c["fonte"], "score": s, "trechos": []})
            if len(d["trechos"]) < 2:
                d["trechos"].append(c)

        top = sorted(docs.values(), key=lambda d: -d["score"])[:k]
        for d in top:
            d["resumo"] = self.resumos[d["trechos"][0]["arquivo"]]
            d["trechos"] = [c for c in d["trechos"] if c["id"] != d["resumo"]["id"]]
        return top

    def contexto(self, texto: str, especie: str | None = None, k: int = 3,
                 max_chars: int = 3000) -> tuple[str, list[str]]:
        """(bloco para o prompt, ["fonte: título", ...] para log)."""
        top = self.buscar(texto, especie, k)
        return montar_contexto_rag(top, max_chars), [f"{d['fonte']}: {d['titulo']}" for d in top]


if __name__ == "__main__":
    if "--download" in sys.argv:
        carregar_modelo()
        print(f"Modelo pronto em {RAG_MODEL_DIR}")
        sys.exit(0)

    rag = RagRetriever()
    relato = ("Vaca leiteira com queda brusca na produção de leite, mucosas pálidas e amareladas, "
              "fraqueza, perda de peso e muitos carrapatos no rebanho. Sem urina escura.")
    bloco, docs = rag.contexto(relato, "Bovino")
    assert docs and bloco.startswith("BASE DE CONHECIMENTO"), docs
    assert categorias_da_especie("Cão") >= {"Canidae"} and categorias_da_especie("Xyz") is None
    print("\n".join(docs), "\n\n", bloco, sep="")
