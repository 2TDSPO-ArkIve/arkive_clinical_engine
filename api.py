from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import validate_config
from agents.clinical_agent import ClinicalIntelligenceEngine

# Guarda a instância do motor de IA; é populada no lifespan, após validar config.
_engine: ClinicalIntelligenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Fail-fast real: valida as variáveis de ambiente (ORACLE_USER, GROQ_API_KEY,
    etc.) e inicializa o motor de IA ANTES do servidor aceitar requisições.
    Se a config estiver inválida ou nenhum modelo Groq responder, o processo
    falha no startup em vez de subir "quebrado" e falhar só na 1ª requisição.
    """
    global _engine
    validate_config()
    _engine = ClinicalIntelligenceEngine()
    yield
    _engine = None


app = FastAPI(title="API Clínica ArkIve", lifespan=lifespan)

# Configuração de CORS para permitir que aplicações web (HTML/JS) chamem a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """
    Liveness/readiness check simples: confirma que o processo subiu e que o
    motor de IA foi inicializado com sucesso (ou seja, que passou pelo
    fail-fast do lifespan). Não faz round-trip ao Oracle/Groq a cada chamada
    para não sobrecarregar as dependências — é um check de processo, não de
    dependência externa.
    """
    return {"status": "ok" if _engine is not None else "unavailable"}


@app.get("/diagnostico/{id_consulta}")
def gerar_diagnostico(id_consulta: int):
    if id_consulta <= 0:
        raise HTTPException(status_code=400, detail="ID deve ser um inteiro positivo.")
        
    try:
        # Aciona o fluxo: Oracle -> Heurística -> Groq
        result = _engine.analyze(id_consulta=id_consulta)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Falha no motor de IA: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {exc}")