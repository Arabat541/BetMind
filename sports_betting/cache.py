# ============================================================
# cache.py — Cache Redis pour les appels API répétés
#
# Si Redis n'est pas disponible, fallback transparent sur un
# dict en mémoire (TTL approximatif — pas persisté).
#
# Usage :
#   from cache import cached_get
#
#   data = cached_get("odds:ligue1", fetch_fn, ttl=300)
#
# Pour activer Redis, définir REDIS_URL dans .env :
#   REDIS_URL=redis://localhost:6379/0
# ============================================================

import json
import logging
import os
import time
from typing import Callable

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")
DEFAULT_TTL = 300   # 5 minutes

_redis_client = None
_mem_cache: dict = {}   # {key: (value, expires_at)}


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not REDIS_URL:
        return None
    try:
        import redis
        r = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        r.ping()
        _redis_client = r
        logger.info(f"cache.py → Redis ({REDIS_URL.split('@')[-1] if '@' in REDIS_URL else REDIS_URL})")
        return r
    except Exception as e:
        logger.warning(f"Redis indisponible ({e}) — cache mémoire utilisé")
        return None


def get(key: str):
    """Lit depuis Redis ou cache mémoire. Retourne None si absent/expiré."""
    r = _get_redis()
    if r:
        try:
            val = r.get(key)
            return json.loads(val) if val else None
        except Exception:
            pass
    # Mémoire
    entry = _mem_cache.get(key)
    if entry:
        value, exp = entry
        if time.time() < exp:
            return value
        del _mem_cache[key]
    return None


def set(key: str, value, ttl: int = DEFAULT_TTL):
    """Écrit dans Redis ou cache mémoire."""
    r = _get_redis()
    if r:
        try:
            r.setex(key, ttl, json.dumps(value, default=str))
            return
        except Exception:
            pass
    _mem_cache[key] = (value, time.time() + ttl)


def cached_get(key: str, fetch_fn: Callable, ttl: int = DEFAULT_TTL):
    """
    Retourne la valeur depuis le cache si présente,
    sinon appelle fetch_fn(), stocke le résultat, et le retourne.
    """
    cached = get(key)
    if cached is not None:
        logger.debug(f"cache HIT: {key}")
        return cached
    logger.debug(f"cache MISS: {key}")
    value = fetch_fn()
    if value:
        set(key, value, ttl=ttl)
    return value


def invalidate(key: str):
    """Supprime une entrée du cache."""
    r = _get_redis()
    if r:
        try:
            r.delete(key)
        except Exception:
            pass
    _mem_cache.pop(key, None)


def invalidate_prefix(prefix: str):
    """Supprime toutes les entrées dont la clé commence par prefix."""
    r = _get_redis()
    if r:
        try:
            keys = r.keys(f"{prefix}*")
            if keys:
                r.delete(*keys)
        except Exception:
            pass
    to_del = [k for k in _mem_cache if k.startswith(prefix)]
    for k in to_del:
        del _mem_cache[k]
