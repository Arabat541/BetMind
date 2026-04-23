# ============================================================
# fixture_congestion.py — AN : Fixture congestion précise
# ============================================================
# Compte les matchs joués dans les 21 jours précédant une rencontre
# et détecte la présence en coupe européenne (UCL/UEL/UECL).
# Scores de congestion normalisés [0,1] : 7 matchs en 21j = 1.0.
# ============================================================

import logging
import os
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

# European sport keys on The Odds API
_EURO_SPORT_KEYS = [
    "soccer_uefa_champions_league",
    "soccer_uefa_europa_league",
    "soccer_uefa_europa_conference_league",
]

# Max realistic matches in 21 days for normalisation
_MAX_MATCHES = 7

# European bonus added to congestion score when team has euro fixture in window
_EURO_BONUS = 0.20


def count_matches_in_window(team_name: str,
                             before_date_str: str,
                             xg_history: dict,
                             window_days: int = 21) -> int:
    """
    Compte le nombre de matchs joués par `team_name` dans les `window_days` jours
    avant `before_date_str` en utilisant l'historique xG d'Understat.
    Retourne 0 si l'équipe est inconnue.
    """
    if not xg_history or team_name not in xg_history:
        return 0
    try:
        before_dt = datetime.strptime(before_date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return 0

    cutoff = before_dt - timedelta(days=window_days)
    count = 0
    for entry in xg_history[team_name]:
        try:
            match_dt = datetime.strptime(entry[0][:10], "%Y-%m-%d")
        except (ValueError, TypeError, IndexError):
            continue
        if cutoff <= match_dt < before_dt:
            count += 1
    return count


def detect_european_fixture(team_name: str,
                             match_date_str: str,
                             api_key: str = "",
                             window_days: int = 21) -> bool:
    """
    Retourne True si `team_name` a un match européen (UCL/UEL/UECL) dans les
    `window_days` jours avant ou après `match_date_str`.
    Nécessite THE_ODDS_API_KEY.  Retourne False silencieusement si erreur.
    """
    if not api_key:
        api_key = os.getenv("THE_ODDS_API_KEY", "")
    if not api_key:
        return False

    try:
        match_dt = datetime.strptime(match_date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return False

    lo = (match_dt - timedelta(days=window_days)).strftime("%Y-%m-%dT00:00:00Z")
    hi = (match_dt + timedelta(days=7)).strftime("%Y-%m-%dT23:59:59Z")

    team_lower = team_name.lower()

    for sport_key in _EURO_SPORT_KEYS:
        try:
            resp = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/events",
                params={
                    "apiKey":         api_key,
                    "commenceTimeFrom": lo,
                    "commenceTimeTo":   hi,
                    "dateFormat":       "iso",
                },
                timeout=8,
            )
            if resp.status_code != 200:
                continue
            for event in resp.json():
                h = event.get("home_team", "").lower()
                a = event.get("away_team", "").lower()
                # fuzzy match: check if team name words appear in event team names
                if _name_match(team_lower, h) or _name_match(team_lower, a):
                    return True
        except Exception:
            continue

    return False


def _name_match(name: str, candidate: str) -> bool:
    """Lightweight fuzzy match: any significant word of `name` in `candidate`."""
    words = [w for w in name.split() if len(w) >= 4]
    if not words:
        return name in candidate
    return any(w in candidate for w in words)


def congestion_score(match_count: int, has_europe: bool) -> float:
    """
    Score normalisé [0,1].
    7 matchs en 21 jours = 1.0 avant bonus européen.
    """
    base = min(match_count / _MAX_MATCHES, 1.0)
    if has_europe:
        base = min(base + _EURO_BONUS, 1.0)
    return round(base, 4)


def build_congestion_features(home_name: str,
                               away_name: str,
                               match_date: str,
                               xg_history: dict,
                               api_key: str = "") -> dict:
    """
    Retourne les features de congestion pour home et away.

    {
      "home_congestion":       int   — matchs 21j
      "away_congestion":       int
      "home_has_europe":       float — 0/1
      "away_has_europe":       float
      "home_congestion_score": float — 0-1
      "away_congestion_score": float
    }
    """
    h_count = count_matches_in_window(home_name, match_date, xg_history)
    a_count = count_matches_in_window(away_name, match_date, xg_history)

    h_euro  = detect_european_fixture(home_name, match_date, api_key=api_key)
    a_euro  = detect_european_fixture(away_name, match_date, api_key=api_key)

    return {
        "home_congestion":       h_count,
        "away_congestion":       a_count,
        "home_has_europe":       float(h_euro),
        "away_has_europe":       float(a_euro),
        "home_congestion_score": congestion_score(h_count, h_euro),
        "away_congestion_score": congestion_score(a_count, a_euro),
    }
