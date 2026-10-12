from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config import validate_config
from agents.clinical_agent import ClinicalIntelligenceEngine

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ClinicalIntelligenceEngine()

@app.get("/health")
def health():
    """Liveness. Não toca Oracle nem Groq: a config já foi validada no boot."""
    return {"status": "ok"}


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
