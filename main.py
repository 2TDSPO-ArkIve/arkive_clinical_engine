"""
Uso: python main.py <ID_CONSULTA>
Imprime o diagnóstico em JSON no stdout (erros também, com o campo "error"). Exit code 0 ou 1.
"""

from __future__ import annotations

import json
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Uso: python main.py <ID_CONSULTA>\n"
            "Exemplo: python main.py 42",
            file=sys.stderr,
        )
        return 1

    try:
        id_consulta = int(sys.argv[1])
        if id_consulta <= 0:
            raise ValueError("ID deve ser um inteiro positivo.")
    except ValueError as exc:
        print(
            f"Erro: ID_CONSULTA inválido — '{sys.argv[1]}'. {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        # importar config configura o logging antes dos outros módulos
        from config import validate_config
        validate_config()
    except RuntimeError as exc:
        print(f"Erro de configuração: {exc}", file=sys.stderr)
        return 1

    try:
        from agents.clinical_agent import ClinicalIntelligenceEngine

        engine = ClinicalIntelligenceEngine()
        result: dict = engine.analyze(id_consulta=id_consulta)

    except ValueError as exc:
        logger.error("Consulta não encontrada: %s", exc)
        error_payload = {
            "error": "CONSULTA_NAO_ENCONTRADA",
            "message": str(exc),
            "id_consulta": id_consulta,
        }
        print(json.dumps(error_payload, indent=2, ensure_ascii=False))
        return 1

    except RuntimeError as exc:
        logger.error("Falha no motor de IA: %s", exc)
        error_payload = {
            "error": "FALHA_MOTOR_IA",
            "message": str(exc),
            "id_consulta": id_consulta,
        }
        print(json.dumps(error_payload, indent=2, ensure_ascii=False))
        return 1

    except Exception as exc:
        logger.exception("Erro inesperado durante a análise clínica: %s", exc)
        error_payload = {
            "error": "ERRO_INESPERADO",
            "message": str(exc),
            "id_consulta": id_consulta,
        }
        print(json.dumps(error_payload, indent=2, ensure_ascii=False))
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("── Análise clínica concluída com sucesso. ──")
    return 0


if __name__ == "__main__":
    sys.exit(main())
