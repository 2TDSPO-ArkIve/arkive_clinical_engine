"""
cache/ttl_cache.py
===================
Cache TTL + LRU simples, em memória, thread-safe.

Usado para evitar buscas repetidas no DuckDuckGo (mesma espécie/raça/sintomas
gerando a mesma query) dentro de uma janela de tempo — reduz latência e risco
de rate limit sem depender de infraestrutura externa (Redis, etc).

Limitação conhecida: o cache é por processo. Se a API rodar com múltiplos
workers (ex: `uvicorn --workers 4`), cada worker terá seu próprio cache —
o que ainda reduz buscas repetidas dentro de cada worker, mas não é
compartilhado entre eles. Para isso, trocar por um cache compartilhado
(Redis) seria o próximo passo natural.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """
    Cache com expiração por tempo (TTL) e despejo por LRU (Least Recently Used)
    quando o tamanho máximo é atingido.
    """

    def __init__(self, max_size: int = 200, ttl_seconds: float = 3600.0) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Retorna o valor em cache, ou None se ausente/expirado."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if time.monotonic() >= expires_at:
                # Expirado: remove e trata como cache miss.
                del self._store[key]
                return None

            # Move para o fim (mais recentemente usado).
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Armazena um valor no cache, com o TTL configurado."""
        with self._lock:
            expires_at = time.monotonic() + self._ttl_seconds
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)

            while len(self._store) > self._max_size:
                self._store.popitem(last=False)  # remove o menos recentemente usado

    def clear(self) -> None:
        """Limpa todo o cache (útil em testes)."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def normalize_cache_key(query: str) -> str:
    """Normaliza uma query de busca para uso como chave de cache."""
    return " ".join(query.strip().lower().split())
