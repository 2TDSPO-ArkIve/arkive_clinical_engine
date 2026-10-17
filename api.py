import json
import logging
import queue
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from config import validate_config
from agents.clinical_agent import ClinicalIntelligenceEngine, _observador

# fail-fast: config inválida derruba o boot, não a primeira request
try:
    validate_config()
except RuntimeError as exc:
    print(f"Erro de configuração: {exc}")
    raise SystemExit(1) from exc

app = FastAPI(title="API Clínica ArkIve")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # sem cookie; "*" com credentials o navegador rejeita
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ClinicalIntelligenceEngine()

@app.get("/health")
def health():
    """Liveness. Não toca Oracle nem Groq: a config já foi validada no boot."""
    return {"status": "ok", "rag": engine._rag is not None}


@app.get("/diagnostico/{id_consulta}")
def gerar_diagnostico(id_consulta: int):
    if id_consulta <= 0:
        raise HTTPException(status_code=400, detail="ID deve ser um inteiro positivo.")

    try:
        return engine.analyze(id_consulta=id_consulta)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Falha no motor de IA: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {exc}")


# ── Pipeline ao vivo (apresentação): mesma análise, cada etapa transmitida por SSE ──

PIPELINE_HTML = Path(__file__).parent / "static" / "pipeline.html"


def _json(o):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, BaseModel):
        return o.model_dump()
    return str(o)  # datas do Oracle, Decimal


@app.get("/pipeline", include_in_schema=False)
def pipeline_page():
    return FileResponse(PIPELINE_HTML)


@app.get("/pipeline/stream/{id_consulta}")
def pipeline_stream(id_consulta: int):
    """Server-Sent Events: uma mensagem por etapa do analyze() + cada linha de log da request."""
    if id_consulta <= 0:
        raise HTTPException(status_code=400, detail="ID deve ser um inteiro positivo.")

    fila: queue.Queue[str | None] = queue.Queue()
    inicio = time.perf_counter()

    # serializa já na thread do motor: o ctx e a saída do LLM são mutados depois da emissão
    def emitir(etapa: str, dados: dict) -> None:
        t_ms = round((time.perf_counter() - inicio) * 1000)
        fila.put(json.dumps({"etapa": etapa, "t_ms": t_ms, "dados": dados}, default=_json, ensure_ascii=False))

    class _LogParaFila(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.thread == worker.ident:
                emitir("log", {"nivel": record.levelname, "origem": record.name, "msg": record.getMessage()})

    def rodar() -> None:
        _observador.set(emitir)
        try:
            engine.analyze(id_consulta=id_consulta)
        except ValueError as exc:
            emitir("erro", {"tipo": "Consulta não encontrada", "msg": str(exc)})
        except RuntimeError as exc:
            emitir("erro", {"tipo": "Falha no motor de IA", "msg": str(exc)})
        except Exception as exc:
            emitir("erro", {"tipo": "Erro inesperado", "msg": str(exc)})
        finally:
            emitir("fim", {})
            fila.put(None)

    handler = _LogParaFila(level=logging.INFO)
    worker = threading.Thread(target=rodar, daemon=True)
    logging.getLogger().addHandler(handler)
    worker.start()

    def eventos():
        try:
            yield f"data: {json.dumps({'etapa': 'inicio', 't_ms': 0, 'dados': {'id_consulta': id_consulta}})}\n\n"
            while (msg := fila.get()) is not None:
                yield f"data: {msg}\n\n"
        finally:
            logging.getLogger().removeHandler(handler)

    return StreamingResponse(eventos(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
