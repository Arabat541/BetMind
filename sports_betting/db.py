# ============================================================
# db.py — Abstraction SQLite / PostgreSQL
#
# Par défaut : SQLite (fichier data/betting.db).
# Pour activer PostgreSQL, définir DATABASE_URL dans .env :
#   DATABASE_URL=postgresql://user:pass@host:5432/betmind
#
# Usage :
#   from db import get_conn, ph
#
#   with get_conn() as conn:
#       conn.execute(f"SELECT * FROM predictions WHERE id = {ph}", (42,))
#
# ph = placeholder : "?" (SQLite) ou "%s" (PostgreSQL)
# ============================================================

import os
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

_USE_PG = bool(DATABASE_URL and DATABASE_URL.startswith("postgresql"))

if _USE_PG:
    try:
        import psycopg2
        import psycopg2.extras
        _PSYCOPG2_OK = True
        logger.info(f"db.py → PostgreSQL ({DATABASE_URL.split('@')[-1]})")
    except ImportError:
        _PSYCOPG2_OK = False
        _USE_PG = False
        logger.warning("psycopg2 non installé — fallback SQLite")
else:
    _PSYCOPG2_OK = False


# Placeholder compatible requêtes SQL
ph = "%s" if _USE_PG else "?"


def _sqlite_path() -> str:
    from config import DB_PATH
    return DB_PATH


class _PgConnWrapper:
    """
    Wrapper psycopg2 qui expose conn.execute() comme sqlite3.
    psycopg2 n'a pas de .execute() sur la connexion directement —
    on délègue à un curseur interne pour une API uniforme.
    """
    def __init__(self, conn):
        self._conn = conn
        self._cur  = conn.cursor()

    def execute(self, sql, params=None):
        if params is not None:
            self._cur.execute(sql, params)
        else:
            self._cur.execute(sql)
        return self._cur

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def cursor(self):
        return self._conn.cursor()

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        try:
            self._cur.close()
        except Exception:
            pass
        self._conn.close()


@contextmanager
def get_conn():
    """
    Context manager qui fournit une connexion DB.
    Commit automatique si pas d'exception, rollback sinon.
    Compatible sqlite3 et psycopg2 — conn.execute() fonctionne dans les deux cas.
    """
    if _USE_PG and _PSYCOPG2_OK:
        raw = psycopg2.connect(DATABASE_URL)
        raw.autocommit = False
        conn = _PgConnWrapper(raw)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        import sqlite3
        conn = sqlite3.connect(_sqlite_path())
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def raw_conn():
    """Connexion directe sans context manager (compatibilité ancienne API)."""
    if _USE_PG and _PSYCOPG2_OK:
        return psycopg2.connect(DATABASE_URL)
    else:
        import sqlite3
        return sqlite3.connect(_sqlite_path())


def adapt_ddl(sql: str) -> str:
    """Adapte le DDL SQLite vers PostgreSQL si nécessaire."""
    if not _USE_PG:
        return sql
    sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
    sql = sql.replace("datetime('now')", "NOW()")
    sql = sql.replace("TEXT DEFAULT (datetime('now'))", "TIMESTAMP DEFAULT NOW()")
    return sql


def is_postgres() -> bool:
    return _USE_PG and _PSYCOPG2_OK
