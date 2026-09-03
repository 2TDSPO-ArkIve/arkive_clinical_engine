from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config import validate_config
from agents.clinical_agent import ClinicalIntelligenceEngine

# Valida as variáveis de ambiente (ORACLE_USER, GROQ_API_KEY, etc.)
# Fail-fast: se a configuração for inválida, a API não deve subir "pela
# metade" e quebrar de forma inesperada na primeira request (ex.: ao
# instanciar ClinicalIntelligenceEngine sem GROQ_API_KEY). Interrompe o
# processo aqui, com uma mensagem de erro clara.
try:
    validate_config()
except RuntimeError as exc:
    print(f"Erro de configuração: {exc}")
    raise SystemExit(1) from exc

app = FastAPI(title="API Clínica ArkIve")

# Configuração de CORS para permitir que aplicações web (HTML/JS) chamem a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o motor de IA[cite: 5]
engine = ClinicalIntelligenceEngine()

@app.get("/diagnostico/{id_consulta}")
def gerar_diagnostico(id_consulta: int):
    if id_consulta <= 0:
        raise HTTPException(status_code=400, detail="ID deve ser um inteiro positivo.")
        
    try:
        # Aciona o fluxo: Oracle -> Heurística -> Groq[cite: 5]
        result = engine.analyze(id_consulta=id_consulta)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Falha no motor de IA: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {exc}")