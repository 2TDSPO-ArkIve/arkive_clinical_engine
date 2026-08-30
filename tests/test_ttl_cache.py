"""
tests/test_ttl_cache.py
========================
Testes unitários do cache/ttl_cache.py usado para cachear buscas do
DuckDuckGo (agents/clinical_agent.py::_perform_web_search).
"""

from __future__ import annotations

import time

import pytest

from cache.ttl_cache import TTLCache, normalize_cache_key


class TestTTLCache:
    def test_set_e_get_basico(self):
        cache = TTLCache(max_size=10, ttl_seconds=60)
        cache.set("chave1", "valor1")
        assert cache.get("chave1") == "valor1"

    def test_get_de_chave_inexistente_retorna_none(self):
        cache = TTLCache()
        assert cache.get("nao_existe") is None

    def test_expiracao_por_ttl(self):
        cache = TTLCache(max_size=10, ttl_seconds=0.05)
        cache.set("chave1", "valor1")
        assert cache.get("chave1") == "valor1"
        time.sleep(0.1)
        assert cache.get("chave1") is None

    def test_lru_descarta_menos_usado_ao_exceder_max_size(self):
        cache = TTLCache(max_size=2, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # deve descartar "a" (o menos recentemente usado)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_get_atualiza_recencia_para_lru(self):
        cache = TTLCache(max_size=2, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # "a" agora é o mais recentemente usado
        cache.set("c", 3)  # deve descartar "b", não "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_clear_remove_tudo(self):
        cache = TTLCache()
        cache.set("a", 1)
        cache.clear()
        assert cache.get("a") is None
        assert len(cache) == 0

    def test_len_reflete_quantidade_de_entradas(self):
        cache = TTLCache(max_size=10, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2


class TestNormalizeCacheKey:
    def test_normaliza_espacos_e_caixa(self):
        assert normalize_cache_key("  Canina   Labrador  ") == "canina labrador"

    def test_queries_equivalentes_geram_mesma_chave(self):
        q1 = "Canina Labrador vomito diarreia"
        q2 = "canina  labrador   vomito diarreia  "
        assert normalize_cache_key(q1) == normalize_cache_key(q2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
