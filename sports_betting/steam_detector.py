# ============================================================
# steam_detector.py — AT : Détecteur de Steam Move
# ============================================================
# Un "steam move" = variation de cote > 5% en < 3 minutes
# sur plusieurs bookmakers simultanément.
#
# Implémentation : polling The Odds API toutes les 60s
# avec stockage en mémoire des snapshots précédents.
# Résolution sub-minute impossible sans WebSocket, mais
# en pratique 60s suffit pour capter les gros moves.
# ============================================================

import logging
import os
import time
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

# Seuils steam move
STEAM_PCT_THRESHOLD = 0.05   # variation cote ≥ 5%
STEAM_BOOKS_MIN     = 2      # nombre minimum de books confirmant le move
STEAM_WINDOW_SEC    = 180    # fenêtre de temps (3 minutes)

# Cache en mémoire : {league_key: {event_id: {bookmaker: {H: odd, D: odd, A: odd}}}}
_snapshots: dict = {}
_snapshot_ts: dict = {}   # {league_key: timestamp dernier snapshot}


def _fetch_odds_snapshot(league_key: str, api_key: str) -> dict:
    """
    Récupère un snapshot des cotes actuelles pour toute la ligue.
    Retourne {event_id: {bookmaker: {outcome: odd}}}.
    """
    try:
        resp = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{league_key}/odds",
            params={
                "apiKey":     api_key,
                "markets":    "h2h",
                "regions":    "eu,uk",
                "oddsFormat": "decimal",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return {}

        snapshot = {}
        for event in resp.json():
            eid = event.get("id", "")
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            snapshot[eid] = {"home_team": home, "away_team": away, "books": {}}
            for book in event.get("bookmakers", []):
                bk = book.get("title", "")
                for mkt in book.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    odds = {}
                    for o in mkt.get("outcomes", []):
                        nm = o.get("name", "")
                        p  = float(o.get("price", 0))
                        if nm == home:
                            odds["H"] = p
                        elif nm == away:
                            odds["A"] = p
                        elif nm == "Draw":
                            odds["D"] = p
                    if odds:
                        snapshot[eid]["books"][bk] = odds
        return snapshot
    except Exception as e:
        logger.debug("steam_detector fetch error: %s", e)
        return {}


def detect_steam_moves(league_key: str, api_key: str = "") -> list[dict]:
    """
    Compare le snapshot actuel au snapshot précédent pour détecter les steam moves.

    Retourne une liste de steam moves détectés :
    [{"event_id", "home_team", "away_team", "outcome", "old_odd", "new_odd",
      "change_pct", "books_moved", "detected_at"}, ...]
    """
    if not api_key:
        api_key = os.getenv("THE_ODDS_API_KEY", "")
    if not api_key:
        return []

    now = datetime.now()
    current = _fetch_odds_snapshot(league_key, api_key)
    if not current:
        return []

    previous = _snapshots.get(league_key, {})
    prev_ts  = _snapshot_ts.get(league_key)

    # Sauvegarder le snapshot actuel
    _snapshots[league_key]   = current
    _snapshot_ts[league_key] = now

    if not previous or prev_ts is None:
        return []

    age_sec = (now - prev_ts).total_seconds()
    if age_sec > STEAM_WINDOW_SEC * 2:
        return []   # snapshot trop vieux — pas de steam valide

    steam_moves = []
    for eid, data in current.items():
        if eid not in previous:
            continue
        prev_data = previous[eid]
        home = data["home_team"]
        away = data["away_team"]

        for outcome in ("H", "D", "A"):
            books_moved = []
            for bk, new_odds in data["books"].items():
                new_odd = new_odds.get(outcome, 0)
                old_odd = prev_data.get("books", {}).get(bk, {}).get(outcome, 0)
                if old_odd <= 0 or new_odd <= 0:
                    continue
                pct_change = abs(new_odd - old_odd) / old_odd
                if pct_change >= STEAM_PCT_THRESHOLD:
                    books_moved.append({
                        "bookmaker": bk,
                        "old_odd":   old_odd,
                        "new_odd":   new_odd,
                        "change_pct": round(pct_change, 4),
                    })

            if len(books_moved) >= STEAM_BOOKS_MIN:
                # Mouvement consensuel → steam move confirmé
                avg_old = sum(b["old_odd"] for b in books_moved) / len(books_moved)
                avg_new = sum(b["new_odd"] for b in books_moved) / len(books_moved)
                direction = "down" if avg_new < avg_old else "up"
                steam_moves.append({
                    "event_id":    eid,
                    "home_team":   home,
                    "away_team":   away,
                    "outcome":     outcome,
                    "direction":   direction,
                    "avg_old_odd": round(avg_old, 3),
                    "avg_new_odd": round(avg_new, 3),
                    "change_pct":  round(abs(avg_new - avg_old) / avg_old, 4),
                    "books_moved": len(books_moved),
                    "books_detail": books_moved,
                    "elapsed_sec": round(age_sec),
                    "detected_at": now.isoformat(),
                })
                logger.info(
                    "Steam move: %s vs %s [%s] %.1f%% in %ds on %d books",
                    home, away, outcome,
                    abs(avg_new - avg_old) / avg_old * 100,
                    age_sec, len(books_moved),
                )

    return steam_moves


def run_steam_monitor_cycle(league_keys: list[str],
                             api_key: str = "",
                             alert_fn=None) -> list[dict]:
    """
    Lance un cycle de détection steam sur toutes les ligues.
    `alert_fn(steam_move)` appelée pour chaque move détecté.
    Retourne la liste de tous les steam moves trouvés.
    """
    if not api_key:
        api_key = os.getenv("THE_ODDS_API_KEY", "")

    all_moves = []
    for lg_key in league_keys:
        moves = detect_steam_moves(lg_key, api_key)
        for m in moves:
            all_moves.append(m)
            if alert_fn:
                try:
                    alert_fn(m)
                except Exception as e:
                    logger.warning("alert_fn error: %s", e)
    return all_moves
