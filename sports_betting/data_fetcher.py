# ============================================================
# data_fetcher.py — Collecte des données
# Sources : football-data.org (foot) + BallDontLie (NBA) + The Odds API
# ============================================================

import time
import logging
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from config import (
    FOOTBALL_DATA_KEY, FOOTBALL_DATA_BASE,
    BALLDONTLIE_BASE, ODDS_API_BASE, THE_ODDS_API_KEY,
    FOOTBALL_LEAGUES, NBA_SEASON, DATA_DIR, DB_PATH
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cache en mémoire pour les stats NBA (évite les 429)
_nba_stats_cache = {}

def _fd_get(endpoint: str, params: dict = {}) -> dict:
    """Wrapper pour football-data.org."""
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    url = f"{FOOTBALL_DATA_BASE}/{endpoint}"
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 429:
            logger.warning("Rate limit atteint, pause 60s...")
            time.sleep(60)
            r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"football-data.org error [{endpoint}]: {e}")
        return {}


def _odds_get(endpoint: str, params: dict) -> dict:
    """Wrapper pour The Odds API."""
    params["apiKey"] = THE_ODDS_API_KEY
    url = f"{ODDS_API_BASE}/{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Odds API error [{endpoint}]: {e}")
        return {}


def _balldontlie_get(endpoint: str, params: dict = {}) -> dict:
    """Wrapper pour BallDontLie (NBA)."""
    from config import BALLDONTLIE_KEY
    url = f"{BALLDONTLIE_BASE}/{endpoint}"
    headers = {"Authorization": BALLDONTLIE_KEY} if BALLDONTLIE_KEY else {}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"BallDontLie error [{endpoint}]: {e}")
        return {}

def fetch_upcoming_football_fixtures(days_ahead: int = 3) -> pd.DataFrame:
    """Récupère les matchs à venir via football-data.org."""
    all_fixtures = []
    date_from = datetime.now().strftime("%Y-%m-%d")
    date_to   = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    for league_name, league_code in FOOTBALL_LEAGUES.items():
        logger.info(f"Fetching fixtures: {league_name}")
        data = _fd_get(f"competitions/{league_code}/matches", {
            "dateFrom": date_from,
            "dateTo":   date_to,
            "status":   "SCHEDULED"
        })
        for match in data.get("matches", []):
            all_fixtures.append({
                "fixture_id":  match["id"],
                "league":      league_name,
                "league_code": league_code,
                "date":        match["utcDate"],
                "home_team":   match["homeTeam"]["name"],
                "home_id":     match["homeTeam"]["id"],
                "away_team":   match["awayTeam"]["name"],
                "away_id":     match["awayTeam"]["id"],
                "matchday":    match.get("matchday", 0),
            })
        time.sleep(0.5)

    df = pd.DataFrame(all_fixtures)
    logger.info(f"Upcoming fixtures fetched: {len(df)}")
    return df


def fetch_team_stats(team_id: int, league_code: str = "") -> dict:
    """Calcule les stats d'une équipe depuis ses derniers matchs."""
    data = _fd_get(f"teams/{team_id}/matches", {"status": "FINISHED", "limit": 20})
    matches = data.get("matches", [])
    if not matches:
        return {}

    wins = draws = losses = clean_sheets = failed_to_score = 0
    goals_for, goals_ag, form_chars = [], [], []

    for m in matches[-15:]:
        is_home = m["homeTeam"]["id"] == team_id
        hg = m.get("score", {}).get("fullTime", {}).get("home") or 0
        ag = m.get("score", {}).get("fullTime", {}).get("away") or 0
        gf = hg if is_home else ag
        ga = ag if is_home else hg

        goals_for.append(gf)
        goals_ag.append(ga)

        if gf > ga:   wins += 1;   form_chars.append("W")
        elif gf == ga: draws += 1;  form_chars.append("D")
        else:          losses += 1; form_chars.append("L")

        if ga == 0: clean_sheets += 1
        if gf == 0: failed_to_score += 1

    n = len(goals_for)
    if n == 0:
        return {}

    form_str = "".join(form_chars)
    return {
        "team_id":         team_id,
        "played":          n,
        "wins":            wins,
        "draws":           draws,
        "losses":          losses,
        "goals_for_avg":   round(sum(goals_for) / n, 4),
        "goals_ag_avg":    round(sum(goals_ag) / n, 4),
        "form_string":     form_str,
        "form_score":      _form_to_score(form_str),
        "clean_sheets":    clean_sheets,
        "failed_to_score": failed_to_score,
    }


def _form_to_score(form_str: str, window: int = 5) -> float:
    recent = form_str[-window:] if len(form_str) >= window else form_str
    if not recent:
        return 0.5
    score = sum(3 if c == "W" else (1 if c == "D" else 0) for c in recent)
    return round(score / (len(recent) * 3), 4)


def fetch_h2h(home_id: int, away_id: int, last_n: int = 10) -> pd.DataFrame:
    """Confrontations directes entre deux équipes."""
    data = _fd_get(f"teams/{home_id}/matches", {"status": "FINISHED", "limit": 50})
    rows = []
    for m in data.get("matches", []):
        h_id = m["homeTeam"]["id"]
        a_id = m["awayTeam"]["id"]
        if {h_id, a_id} != {home_id, away_id}:
            continue
        hg = m.get("score", {}).get("fullTime", {}).get("home") or 0
        ag = m.get("score", {}).get("fullTime", {}).get("away") or 0
        rows.append({
            "date": m["utcDate"], "home_id": h_id, "away_id": a_id,
            "home_goals": hg, "away_goals": ag,
            "result": "H" if hg > ag else ("A" if ag > hg else "D")
        })
        if len(rows) >= last_n:
            break
    return pd.DataFrame(rows)


def fetch_standings(league_code: str) -> pd.DataFrame:
    """Classement d'une ligue."""
    data = _fd_get(f"competitions/{league_code}/standings")
    rows = []
    for standing in data.get("standings", []):
        if standing.get("type") != "TOTAL":
            continue
        for team in standing.get("table", []):
            rows.append({
                "position":  team["position"],
                "team_id":   team["team"]["id"],
                "team_name": team["team"]["name"],
                "played":    team["playedGames"],
                "won":       team["won"],
                "draw":      team["draw"],
                "lost":      team["lost"],
                "goals_for": team["goalsFor"],
                "goals_ag":  team["goalsAgainst"],
                "points":    team["points"],
            })
    return pd.DataFrame(rows)


def fetch_football_odds(league_key: str = "soccer_france_ligue_one") -> pd.DataFrame:
    """Cotes 1X2 via The Odds API."""
    data = _odds_get(f"sports/{league_key}/odds", {
        "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"
    })
    rows = []
    for event in (data if isinstance(data, list) else []):
        home, away, commence = event.get("home_team",""), event.get("away_team",""), event.get("commence_time","")
        for bookmaker in event.get("bookmakers", [])[:3]:
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                rows.append({
                    "event_id": event["id"], "home_team": home, "away_team": away,
                    "commence": commence, "bookmaker": bookmaker["title"],
                    "odd_home": outcomes.get(home), "odd_draw": outcomes.get("Draw"),
                    "odd_away": outcomes.get(away),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby(["event_id","home_team","away_team","commence"], as_index=False).agg(
            odd_home=("odd_home","mean"), odd_draw=("odd_draw","mean"), odd_away=("odd_away","mean"))
        df["impl_home"] = 1 / df["odd_home"]
        df["impl_draw"] = 1 / df["odd_draw"]
        df["impl_away"] = 1 / df["odd_away"]
        margin = df["impl_home"] + df["impl_draw"] + df["impl_away"]
        df[["impl_home","impl_draw","impl_away"]] = df[["impl_home","impl_draw","impl_away"]].div(margin, axis=0)
    logger.info(f"Odds fetched: {len(df)} events ({league_key})")
    return df


def fetch_upcoming_nba_games(days_ahead: int = 3) -> pd.DataFrame:
    """Matchs NBA à venir via BallDontLie."""
    date_from = datetime.now().strftime("%Y-%m-%d")
    date_to   = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    data = _balldontlie_get("games", {"start_date": date_from, "end_date": date_to, "per_page": 100})
    rows = []
    for g in data.get("data", []):
        rows.append({
            "game_id":   g["id"], "date": g["date"],
            "home_team": g["home_team"]["full_name"], "home_abbr": g["home_team"]["abbreviation"],
            "away_team": g["visitor_team"]["full_name"], "away_abbr": g["visitor_team"]["abbreviation"],
            "season": g["season"], "home_score": g.get("home_team_score"),
            "away_score": g.get("visitor_team_score"), "status": g["status"],
        })
    df = pd.DataFrame(rows)
    logger.info(f"NBA games fetched: {len(df)}")
    return df


def fetch_nba_team_stats(team_id: int, last_n: int = 10) -> dict:
    """Stats récentes d'une équipe NBA — avec cache mémoire."""
    global _nba_stats_cache
    if team_id in _nba_stats_cache:
        return _nba_stats_cache[team_id]

    time.sleep(2)  # respect rate limit BallDontLie
    data = _balldontlie_get("games", {
        "team_ids[]": team_id,
        "per_page":   last_n,
        "seasons[]":  2024
    })
    games = data.get("data", [])
    if not games:
        return {}

    pts_for, pts_ag, wins = [], [], 0
    for g in games:
        is_home = g["home_team"]["id"] == team_id
        pf = g["home_team_score"] if is_home else g["visitor_team_score"]
        pa = g["visitor_team_score"] if is_home else g["home_team_score"]
        if pf is not None and pa is not None:
            pts_for.append(pf)
            pts_ag.append(pa)
            if pf > pa:
                wins += 1

    n = len(pts_for)
    if n == 0:
        return {}

    result = {
        "team_id":      team_id,
        "games":        n,
        "win_rate":     round(wins / n, 4),
        "pts_for_avg":  round(sum(pts_for) / n, 2),
        "pts_ag_avg":   round(sum(pts_ag) / n, 2),
        "pts_diff_avg": round((sum(pts_for) - sum(pts_ag)) / n, 2),
    }
    _nba_stats_cache[team_id] = result
    return result


def fetch_nba_odds() -> pd.DataFrame:
    return fetch_football_odds("basketball_nba")


def init_db():
    """Initialise la base de données SQLite."""
    import os
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT    DEFAULT (datetime('now')),
            sport        TEXT, league TEXT, home_team TEXT, away_team TEXT,
            match_date   TEXT, pred_result TEXT,
            prob_home    REAL, prob_draw REAL, prob_away REAL,
            confidence   REAL, is_value_bet INTEGER DEFAULT 0,
            edge         REAL, kelly_stake REAL, odd_used REAL,
            outcome      TEXT DEFAULT NULL, pnl REAL DEFAULT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS bankroll (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            updated_at TEXT DEFAULT (datetime('now')),
            balance    REAL, total_bets INTEGER, wins INTEGER, losses INTEGER, roi REAL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized.")


def save_prediction(pred: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
        (sport,league,home_team,away_team,match_date,pred_result,
         prob_home,prob_draw,prob_away,confidence,is_value_bet,edge,kelly_stake,odd_used)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        pred.get("sport"), pred.get("league"), pred.get("home_team"), pred.get("away_team"),
        pred.get("match_date"), pred.get("pred_result"), pred.get("prob_home"),
        pred.get("prob_draw"), pred.get("prob_away"), pred.get("confidence"),
        int(pred.get("is_value_bet", 0)), pred.get("edge"), pred.get("kelly_stake"), pred.get("odd_used")
    ))
    conn.commit()
    conn.close()


def get_all_predictions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY created_at DESC", conn)
    conn.close()
    return df