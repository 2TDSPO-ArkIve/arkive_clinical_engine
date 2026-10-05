"""
rag/retriever.py
================
Busca semântica na base de conhecimento veterinária (fichas WOAH, AAZV,
CFSPH, CAPC, ABCD, ESCCAP, USGS, ARWH, WHA) indexada por rag/build_index.ipynb.

Mesma lógica da seção "11. Busca" do notebook — qualquer mudança em
embedding, MAPA_ESPECIE ou agregação deve ser feita nos dois lugares.

Uso:
    python -m rag.retriever             # teste com um relato de exemplo
    python -m rag.retriever --download  # baixa o modelo + grava tokenizer podado (build do Render)
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import faiss
import numpy as np

from config import RAG_INDEX_DIR, RAG_MODEL_DIR

# Precisam bater com o notebook (cell "2. Configuração") — senão o índice é inválido.
MODEL_NAME = "Xenova/multilingual-e5-small"
MODEL_FILE = "onnx/model_quantized.onnx"
DIM = 384
QUERY_WINDOW_CHARS = 1500

CATEGORIAS_SEMPRE = {"Multiple Species", "Other Diseases"}
# Fichas genéricas passam em qualquer filtro de espécie e lotavam o top-k com ruído
# (ex.: micobactéria de ungulados num cão). Com espécie conhecida, perdem um pouco de score.
PENALIDADE_GENERICA = 0.01
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
    "wombat": {"Vombatidae"}, "vombat": {"Vombatidae"},
    "equidna": {"Tachyglossidae"},
    "possum": {"Phalangeridae", "Pseudocheiridae"}, "falanger": {"Phalangeridae"}, "cuscus": {"Phalangeridae"},
    "marsupia": {"Macropodidae", "Vombatidae", "Phalangeridae", "Pseudocheiridae",
                 "Phascolarctidae", "Dasyuridae", "Peramelidae"},
    "koala": {"Phascolarctidae"}, "coala": {"Phascolarctidae"},
    "diabo": {"Dasyuridae"}, "dasyur": {"Dasyuridae"},
    "bandicoot": {"Peramelidae"},
    "ornitorrinco": {"Ornithorhynchidae"}, "platypus": {"Ornithorhynchidae"},
    "golfinho": {"Marine Mammals"}, "foca": {"Marine Mammals"}, "baleia": {"Marine Mammals"},
    "furao": {"Mustelidae"}, "mustel": {"Mustelidae"},
    "primata": {"Primates"}, "macaco": {"Primates"}, "sagui": {"Primates"},
    "reptil": {"Reptilia"}, "serpente": {"Reptilia"}, "cobra": {"Reptilia"}, "lagarto": {"Reptilia"},
    "jabuti": {"Reptilia"}, "tartaruga": {"Reptilia"}, "iguana": {"Reptilia"},
    "roedor": {"Rodentia"}, "hamster": {"Rodentia"}, "rato": {"Rodentia"}, "camundongo": {"Rodentia"},
    "chinchila": {"Rodentia"}, "porquinho": {"Rodentia"},
}

# Sinais clínicos/histórico PT (sem acento) -> termos EN anexados à consulta.
# e5-small alinha PT↔EN mal; os documentos são em inglês. Só sinais, nunca diagnósticos.
GLOSSARIO = {
    "febre": "fever", "hipertermia": "hyperthermia", "hipotermia": "hypothermia",
    "apatia": "lethargy depression", "prostracao": "lethargy prostration", "fraqueza": "weakness",
    "letargia": "lethargy", "anorexia": "anorexia", "inapetencia": "inappetence anorexia",
    "sem apetite": "inappetence anorexia", "perda de peso": "weight loss", "emagrecimento": "weight loss emaciation",
    "caquexia": "cachexia emaciation", "desidratacao": "dehydration",
    "vomito": "vomiting", "diarreia": "diarrhea", "diarreia com sangue": "bloody diarrhea",
    "fezes com sangue": "bloody feces", "constipacao": "constipation", "colica": "colic abdominal pain",
    "dor abdominal": "abdominal pain", "distensao abdominal": "abdominal distension", "timpanismo": "bloat",
    "ictericia": "jaundice icterus", "amarelad": "jaundice icterus", "mucosas palidas": "pale mucous membranes anemia",
    "palidez": "pallor anemia", "anemia": "anemia", "urina escura": "dark urine hemoglobinuria",
    "sangue na urina": "hematuria", "hematuria": "hematuria", "poliuria": "polyuria", "polidipsia": "polydipsia",
    "bebe muita agua": "polydipsia", "urina muito": "polyuria",
    "tosse": "cough", "espirro": "sneezing", "secrecao nasal": "nasal discharge", "corrimento nasal": "nasal discharge",
    "dispneia": "dyspnea", "dificuldade respiratoria": "dyspnea respiratory distress", "respiracao ofegante": "dyspnea",
    "secrecao ocular": "ocular discharge", "conjuntivite": "conjunctivitis", "lacrimejamento": "lacrimation",
    "salivacao": "salivation hypersalivation", "baba": "drooling salivation", "dificuldade para engolir": "dysphagia",
    "lesoes na boca": "oral lesions", "ulceras": "ulcers", "vesiculas": "vesicles",
    "convulsao": "seizures", "convulsoes": "seizures", "tremor": "tremors", "ataxia": "ataxia",
    "incoordenacao": "incoordination ataxia", "andar cambaleante": "ataxia", "paralisia": "paralysis",
    "paresia": "paresis", "patas traseiras": "hind limbs", "membros posteriores": "hind limbs",
    "cegueira": "blindness", "andar em circulo": "circling", "inclinacao da cabeca": "head tilt",
    "mudanca de comportamento": "behavior change", "agressiv": "aggression", "agitacao": "agitation",
    "claudicacao": "lameness", "manqueira": "lameness", "mancando": "lameness", "rigidez": "stiffness",
    "dor articular": "joint pain arthritis", "inchaco": "swelling edema", "edema": "edema",
    "linfonodos aumentados": "lymphadenopathy", "ingua": "lymphadenopathy",
    "coceira": "pruritus itching", "prurido": "pruritus", "queda de pelo": "alopecia hair loss",
    "alopecia": "alopecia", "pelagem opaca": "dull coat", "crostas": "crusts", "descamacao": "scaling",
    "feridas": "skin lesions wounds", "nodulos": "nodules", "abscesso": "abscess",
    "aborto": "abortion", "natimorto": "stillbirth", "retencao de placenta": "retained placenta",
    "infertilidade": "infertility", "mastite": "mastitis", "queda na producao de leite": "drop in milk production",
    "producao de leite": "milk production", "queda na postura": "drop in egg production",
    "ovos": "eggs", "casca fina": "thin eggshell", "casca deformada": "misshapen eggs",
    "mortalidade": "mortality", "morte subita": "sudden death", "morreram": "deaths mortality",
    "carrapato": "ticks", "pulga": "fleas", "piolho": "lice", "verme": "worms helminths", "sarna": "mange mites",
    "mordid": "bite", "morcego": "bat", "rato": "rodents", "caca": "hunting predation",
    "carne crua": "raw meat", "agua parada": "stagnant water", "contato com": "contact with",
    "rebanho": "herd", "plantel": "flock", "filhote": "puppy kitten young", "pintinho": "chicks",
    "bezerro": "calf", "leitao": "piglet", "potro": "foal",
}
_GLOSSARIO_RE = [(re.compile(rf"\b{re.escape(k)}"), v) for k, v in GLOSSARIO.items()]
# "sem urina escura", "nega vômito": negação na mesma oração, logo antes do termo.
_NEGACAO = re.compile(r"\b(sem|nao|nega|negou|ausencia de)\b[^.,;]{0,20}$")


# Embedding sem fastembed: o tokenizer.json completo do e5 (250k peças, todos os
# alfabetos) ocupa ~250 MB no `tokenizers` e estoura os 512 MB do Render. O build
# grava uma versão podada às peças com caracteres do corpus/latinos (~80 MB, mesmos
# ids após remapear) e a busca roda ONNX direto, mean pooling + L2 como o fastembed.
TOKENIZER_RAG = RAG_MODEL_DIR / "tokenizer_rag.json"
IDS_RAG = RAG_MODEL_DIR / "ids_rag.npy"
MAX_TOKENS = 512  # model_max_length do tokenizer_config.json


def preparar_modelo(index_dir: Path = RAG_INDEX_DIR, cache_dir: Path = RAG_MODEL_DIR) -> None:
    """Baixa o modelo e grava o tokenizer podado (build do Render; precisa do índice)."""
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    snap = Path(snapshot_download(MODEL_NAME, cache_dir=str(cache_dir),
                                  allow_patterns=[MODEL_FILE, "tokenizer.json"]))
    cfg = json.loads((snap / "tokenizer.json").read_text(encoding="utf-8"))
    # só o normalizador (vocab de 1 peça): o tokenizer inteiro custaria ~250 MB também no build
    vazio = {**cfg, "added_tokens": [], "post_processor": None,
             "model": {**cfg["model"], "vocab": [["<unk>", 0.0]], "unk_id": 0}}
    norm = Tokenizer.from_str(json.dumps(vazio)).normalizer
    # latim, grego (α, µ), pontuação tipográfica (— “ ”), moedas, setas e símbolos matemáticos
    faixas = [(0, 0x250), (0x370, 0x400), (0x1E00, 0x1F00), (0x2000, 0x2300)]
    chars = {chr(i) for a, b in faixas for i in range(a, b)} | {"▁"}
    with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
        for c in map(json.loads, f):
            chars.update(norm.normalize_str(c["texto"]))
    vocab = cfg["model"]["vocab"]
    manter = [i for i, (peca, _) in enumerate(vocab) if i < 4 or set(peca) <= chars]
    novo = {antigo: n for n, antigo in enumerate(manter)}
    cfg["model"]["vocab"] = [vocab[i] for i in manter]
    cfg["model"]["unk_id"] = novo[cfg["model"]["unk_id"]]
    for t in cfg["added_tokens"]:
        t["id"] = novo[t["id"]]
    for t in cfg["post_processor"]["special_tokens"].values():
        t["ids"] = [novo[i] for i in t["ids"]]
    TOKENIZER_RAG.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    np.save(IDS_RAG, np.array(manter, dtype=np.int64))


class Embedder:
    """e5 int8 via onnxruntime + tokenizer podado. Mesmo vetor do fastembed do notebook."""

    def __init__(self, cache_dir: Path = RAG_MODEL_DIR) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        if not TOKENIZER_RAG.exists():
            raise RuntimeError("tokenizer podado ausente — rode `python -m rag.retriever --download`")
        self.tok = Tokenizer.from_file(str(TOKENIZER_RAG))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.enable_padding(pad_id=1, pad_token="<pad>")  # <pad> = id 1 nos dois vocabulários
        self.ids = np.load(IDS_RAG)
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False  # arena guarda ~120 MB após a 1ª inferência
        onnx = hf_hub_download(MODEL_NAME, MODEL_FILE, cache_dir=str(cache_dir), local_files_only=True)
        self.sessao = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])

    def embed(self, textos: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(textos)
        ids = self.ids[np.array([e.ids for e in enc])]
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        saida = self.sessao.run(None, {"input_ids": ids, "attention_mask": mask,
                                       "token_type_ids": np.zeros_like(ids)})[0]
        v = (saida * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1e-9)
        return (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)).astype("float32")


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


def expandir(texto: str) -> str:
    """Relato PT + termos EN do GLOSSARIO encontrados (aproxima a consulta dos documentos em inglês)."""
    n = _norm(texto)
    termos = list(dict.fromkeys(v for r, v in _GLOSSARIO_RE
                                if any(not _NEGACAO.search(n[:m.start()]) for m in r.finditer(n))))
    return f"{texto} | {' '.join(termos)}" if termos else texto


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
    partes = ["BASE DE CONHECIMENTO VETERINÁRIA (RAG — WOAH, AAZV, CFSPH, CAPC, ABCD, ESCCAP, USGS, ARWH, WHA; em inglês). "
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
        # modelo primeiro: o pico transitório da sessão ONNX (~80 MB) não soma com chunks/índice
        self.modelo = Embedder()
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = {c["id"]: c for c in map(json.loads, f)}
        # resumo = primeiro chunk do arquivo (SUMMARY / Scope)
        self.resumos = {c["arquivo"]: c for c in self.chunks.values() if c["ordem"] == 0}

    def embed(self, textos: list[str]) -> np.ndarray:
        """Vetores float32 normalizados (produto interno = cosseno). Um texto por vez:
        lote de janelas de 512 tokens soma ~100 MB de ativações (limite 512 MB do Render)."""
        return np.concatenate([self.modelo.embed([t]) for t in textos])

    def buscar(self, texto: str, especie: str | None = None, k: int = 3, k_chunks: int = 100,
               min_score: float = 0.0) -> list[dict]:
        """Top-k documentos distintos (fonte + título) com score >= min_score; pode voltar vazio.
        Score = maior similaridade chunk × janela do relato (menos PENALIDADE_GENERICA)."""
        janelas = [expandir(j) for j in fatiar(texto, QUERY_WINDOW_CHARS) or [texto]]
        scores, ids = self.index.search(self.embed([f"query: {j}" for j in janelas]),
                                        min(k_chunks, self.index.ntotal))
        permitidas = categorias_da_especie(especie)

        melhor_por_chunk: dict[int, float] = {}
        for linha_s, linha_i in zip(scores, ids):
            for s, i in zip(linha_s, linha_i):
                if i != -1 and s > melhor_por_chunk.get(int(i), -1):
                    melhor_por_chunk[int(i)] = float(s)
        if permitidas is not None:
            for i in melhor_por_chunk:
                if self.chunks[i]["categoria"] in CATEGORIAS_SEMPRE:
                    melhor_por_chunk[i] -= PENALIDADE_GENERICA

        docs: dict[tuple[str, str], dict] = {}
        for i, s in sorted(melhor_por_chunk.items(), key=lambda x: -x[1]):
            c = self.chunks[i]
            if s < min_score or (permitidas is not None and c["categoria"] not in permitidas):
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
                 max_chars: int = 3000, min_score: float = 0.0) -> tuple[str, list[str]]:
        """(bloco para o prompt, ["fonte: título", ...] para log). ("", []) se nada passar do corte."""
        top = self.buscar(texto, especie, k, min_score=min_score)
        return montar_contexto_rag(top, max_chars), [f"{d['fonte']}: {d['titulo']}" for d in top]


if __name__ == "__main__":
    if "--download" in sys.argv:
        preparar_modelo()
        print(f"Modelo pronto em {RAG_MODEL_DIR}")
        sys.exit(0)

    rag = RagRetriever()
    relato = ("Vaca leiteira com queda brusca na produção de leite, mucosas pálidas e amareladas, "
              "fraqueza, perda de peso e muitos carrapatos no rebanho. Sem urina escura.")
    bloco, docs = rag.contexto(relato, "Bovino")
    assert docs and bloco.startswith("BASE DE CONHECIMENTO"), docs
    # embedding daqui ≈ embedding do notebook (fastembed) que gerou o índice. Não é 1.0:
    # o modelo int8 quantiza ativações por lote, e o notebook embutiu em lotes (~0.998).
    amostra = list(rag.chunks.values())[::750]
    vet = rag.embed([f"passage: {c['titulo']} — {c['secao']}\n{c['texto']}" for c in amostra])
    linha = {int(i): n for n, i in enumerate(faiss.vector_to_array(rag.index.id_map))}
    ref = np.stack([rag.index.index.reconstruct(linha[c["id"]]) for c in amostra])
    assert (vet * ref).sum(1).min() > 0.995, (vet * ref).sum(1)
    # corte de score e penalidade multiespécie (RAG_MIN_SCORE do config)
    from config import RAG_MIN_SCORE
    assert rag.buscar("Cão mancando da pata traseira direita após correr no parque, sem febre.",
                      "Cão", min_score=RAG_MIN_SCORE) == []
    gato = rag.buscar("Gato com espirros, secreção nasal e ocular, úlceras na língua e febre há 4 dias.",
                      "Gato", min_score=RAG_MIN_SCORE)
    assert {"Feline Herpesvirus infection", "Feline calicivirus infection"} & {d["titulo"] for d in gato}, gato
    pica = rag.buscar("Vômitos frequentes após ingerir plantas ou objetos, lambe móveis, sem apetite.", "Cão")
    assert all(d["categoria"] == "Canidae" for d in pica), [(d["titulo"], d["categoria"]) for d in pica]
    assert categorias_da_especie("Cão") >= {"Canidae"} and categorias_da_especie("Xyz") is None
    assert expandir("Febre e vômito") == "Febre e vômito | fever vomiting" and expandir("ok") == "ok"
    assert expandir("Sem urina escura") == "Sem urina escura"
    assert expandir("Febre, sem vômito. Vômito ontem") == "Febre, sem vômito. Vômito ontem | fever vomiting"
    assert expandir("Febre, sem vômito") == "Febre, sem vômito | fever"
    print("\n".join(docs), "\n\n", bloco, sep="")
