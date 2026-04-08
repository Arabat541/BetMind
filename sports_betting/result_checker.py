# ============================================================
# result_checker.py — Vérification automatique des résultats
# ============================================================

import sqlite3
import logging
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from config import FOOTBALL_DATA_KEY, FOOTBALL_DATA_BASE, DB_PATH, BALLDONTLIE_KEY
from bankroll import BankrollTracker
from telegram_bot import send_message

logger  = logging.getLogger(__name__)
tracker = BankrollTracker()

# ── Cache par date (évite les appels répétés pour la même journée) ──
_nba_results_cache  = {}   # {date_str: [games]}
_foot_results_cache = {}   # {date_str: [matches]}


# ════════════════════════════════════════════════════════════
# WRAPPERS API
# ════════════════════════════════════════════════════════════

def _fd_get(endpoint: str, params: dict = {}) -> dict:
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    url = f"{FOOTBALL_DATA_BASE}/{endpoint}"
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 429:
            logger.warning("Rate limit football-data.org, pause 60s...")
            time.sleep(60)
            r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"football-data.org error [{endpoint}]: {e}")
        return {}


def _balldontlie_get(endpoint: str, params: dict = {}) -> dict:
    url = f"https://api.balldontlie.io/v1/{endpoint}"
    headers = {"Authorization": BALLDONTLIE_KEY} if BALLDONTLIE_KEY else {}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"BallDontLie error [{endpoint}]: {e}")
        return {}


# ════════════════════════════════════════════════════════════
# CACHE PAR DATE — 1 appel API par date, pas par match
# ════════════════════════════════════════════════════════════

def _get_nba_games_for_date(date_str: str) -> list:
    """Récupère TOUS les matchs NBA d'une date en un seul appel."""
    if date_str in _nba_results_cache:
        return _nba_results_cache[date_str]
    time.sleep(2)
    data = _balldontlie_get("games", {
        "start_date": date_str,
        "end_date":   date_str,
        "per_page":   100
    })
    games = data.get("data", [])
    _nba_results_cache[date_str] = games
    logger.info(f"NBA cache chargé pour {date_str}: {len(games)} matchs")
    return games


def _get_football_matches_for_date(date_str: str) -> list:
    """Récupère TOUS les matchs foot terminés d'une date en un seul appel."""
    if date_str in _foot_results_cache:
        return _foot_results_cache[date_str]
    time.sleep(0.5)
    data = _fd_get("matches", {
        "dateFrom": date_str,
        "dateTo":   date_str,
        "status":   "FINISHED"
    })
    matches = data.get("matches", [])
    _foot_results_cache[date_str] = matches
    logger.info(f"Football cache chargé pour {date_str}: {len(matches)} matchs")
    return matches


# ════════════════════════════════════════════════════════════
# MATCHING & RÉSULTATS
# ════════════════════════════════════════════════════════════

def _fuzzy_match(name1: str, name2: str) -> bool:
    """Matching flexible sur les mots significatifs."""
    SKIP = {"fc", "as", "sc", "ac", "rc", "us", "cf", "af", "bk", "la", "los"}
    def key_words(name: str) -> set:
        return {w for w in name.lower().split() if w not in SKIP and len(w) > 2}
    kw1 = key_words(name1)
    kw2 = key_words(name2)
    return len(kw1 & kw2) > 0 if (kw1 and kw2) else False


def get_football_result(home_team: str, away_team: str, match_date: str) -> str | None:
    """Résultat d'un match foot : H, D, A ou None."""
    date_str = match_date[:10]
    matches  = _get_football_matches_for_date(date_str)

    for match in matches:
        h = match["homeTeam"]["name"]
        a = match["awayTeam"]["name"]
        if not (_fuzzy_match(h, home_team) and _fuzzy_match(a, away_team)):
            continue
        score = match.get("score", {}).get("fullTime", {})
        hg = score.get("home")
        ag = score.get("away")
        if hg is None or ag is None:
            return None
        return "H" if hg > ag else ("A" if ag > hg else "D")

    return None


def get_nba_result(home_team: str, away_team: str, match_date: str) -> str | None:
    """Résultat d'un match NBA : H, A ou None."""
    date_str = match_date[:10]
    games    = _get_nba_games_for_date(date_str)

    for g in games:
        h = g["home_team"]["full_name"]
        a = g["visitor_team"]["full_name"]
        if not (_fuzzy_match(h, home_team) and _fuzzy_match(a, away_team)):
            continue
        hs  = g.get("home_team_score")
        as_ = g.get("visitor_team_score")
        if hs is None or as_ is None:
            return None
        if str(g.get("status", "")).lower() not in ["final"]:
            return None
        return "H" if hs > as_ else "A"

    return None


# ════════════════════════════════════════════════════════════
# MISE À JOUR DB + BANKROLL
# ════════════════════════════════════════════════════════════

def settle_prediction(pred_id: int, outcome: str, pred_result: str,
                      kelly_stake: float, odd_used: float):
    """Met à jour une prédiction et calcule le PnL."""
    won   = (pred_result == outcome)
    stake = kelly_stake or 0
    odd   = odd_used or 1.0
    pnl   = round(stake * (odd - 1), 0) if (won and stake > 0 and odd > 1) else (-stake if stake > 0 else 0.0)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE predictions SET outcome=?, pnl=? WHERE id=?", (outcome, pnl, pred_id))
    conn.commit()
    conn.close()

    if stake > 0:
        tracker.settle_bet(pred_id, outcome)

    return won, pnl


def _send_result_alert(pred, outcome: str, won: bool, pnl: float):
    """Alerte Telegram avec le résultat."""
    NAMES = {"H": "Victoire domicile", "D": "Match nul", "A": "Victoire extérieur"}
    emoji       = "✅" if won else "❌"
    sport_emoji = "⚽" if pred["sport"] == "football" else "🏀"
    balance     = tracker.get_balance()

    lines = [
        f"{emoji} <b>RÉSULTAT</b> {sport_emoji}",
        "",
        f"<b>{pred['home_team']} vs {pred['away_team']}</b>",
        f"🏆 {pred['league']}",
        "",
        f"Prédiction : {NAMES.get(pred['pred_result'], pred['pred_result'])}",
        f"Résultat   : {NAMES.get(outcome, outcome)}",
        "",
        f"💰 P&L      : <b>{'+'if pnl>0 else ''}{pnl:,.0f} FCFA</b>",
        f"📊 Bankroll : {balance:,.0f} FCFA",
        "",
        "🤖 BetMind Agent"
    ]
    send_message("\n".join(lines))


# ════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════

def run_result_checker():
    """
    Vérifie tous les paris en attente dont la date est passée.
    Utilise un cache par date — 1 appel API par date, pas par match.
    """
    logger.info("═══ RESULT CHECKER ═══")

    conn = sqlite3.connect(DB_PATH)
    pending = pd.read_sql_query("""
        SELECT * FROM predictions
        WHERE outcome IS NULL
        AND match_date IS NOT NULL
        AND match_date <= date('now', '-1 day')
        ORDER BY match_date ASC
    """, conn)
    conn.close()

    if pending.empty:
        logger.info("Aucun résultat à vérifier.")
        return

    logger.info(f"{len(pending)} prédictions à vérifier...")

    settled = wins = losses = 0

    for _, pred in pending.iterrows():
        try:
            sport      = pred["sport"]
            home_team  = pred["home_team"]
            away_team  = pred["away_team"]
            match_date = str(pred["match_date"])

            # Résultat réel — via cache (1 appel par date)
            if sport == "football":
                outcome = get_football_result(home_team, away_team, match_date)
            elif sport == "nba":
                outcome = get_nba_result(home_team, away_team, match_date)
            else:
                continue

            if outcome is None:
                logger.info(f"Résultat non disponible: {home_team} vs {away_team} ({match_date})")
                continue

            # Mise à jour DB + bankroll
            won, pnl = settle_prediction(
                pred_id=int(pred["id"]),
                outcome=outcome,
                pred_result=pred["pred_result"],
                kelly_stake=float(pred["kelly_stake"] or 0),
                odd_used=float(pred["odd_used"] or 0),
            )

            settled += 1
            wins   += 1 if won else 0
            losses += 0 if won else 1

            status = "✅ GAGNÉ" if won else "❌ PERDU"
            logger.info(
                f"{status} | {home_team} vs {away_team} | "
                f"Prédit: {pred['pred_result']} | Réel: {outcome} | "
                f"P&L: {pnl:+,.0f} FCFA"
            )

            # Alerte Telegram si mise > 0
            if float(pred["kelly_stake"] or 0) > 0:
                _send_result_alert(pred, outcome, won, pnl)

        except Exception as e:
            logger.error(f"Erreur résultat {pred.get('home_team')} vs {pred.get('away_team')}: {e}")
            continue

    if settled > 0:
        balance = tracker.get_balance()
        logger.info(
            f"Result check done: {settled} réglés | "
            f"{wins}W / {losses}L | "
            f"Bankroll: {balance:,.0f} FCFA"
        )
        if settled >= 3:
            from telegram_bot import send_daily_summary
            send_daily_summary(tracker.get_stats())
    else:
        logger.info("Aucun résultat réglé ce cycle.")


if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_result_checker()