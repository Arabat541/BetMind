# ============================================================
# understat_fetcher.py — xG réels depuis understat.com (W)
# Télécharge et cache les données xG par équipe.
# Sauvegarde : data/understat_xg_history.json
# ============================================================

import json
import logging
import os
import time
from datetime import datetime
from difflib import get_close_matches

logger = logging.getLogger(__name__)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
XG_PATH   = os.path.join(DATA_DIR, "understat_xg_history.json")

# Understat league names → football-data.co.uk codes
UNDERSTAT_LEAGUES = {
    "EPL":        "E0",
    "La_Liga":    "SP1",
    "Bundesliga": "D1",
    "Serie_A":    "I1",
    "Ligue_1":    "F1",
}

SEASONS = ["2019", "2020", "2021", "2022", "2023", "2024"]

# Cache en mémoire (chargé une fois par process)
_xg_cache: dict | None = None


# ════════════════════════════════════════════════════════════
# SCRAPING
# ════════════════════════════════════════════════════════════

def fetch_and_save_xg_history(seasons: list | None = None) -> dict:
    """
    Scrape understat.com pour toutes les ligues et saisons spécifiées.
    Construit : {team_name → [(date_str, xg_for, xg_against), ...]} (chronologique).
    Sauvegarde dans data/understat_xg_history.json et retourne le dict.
    """
    try:
        import understatapi
    except ImportError:
        logger.error("understatapi non installé — pip install understatapi")
        return {}

    seasons = seasons or SEASONS
    history: dict[str, list] = {}

    with understatapi.UnderstatClient() as understat:
        for league_name in UNDERSTAT_LEAGUES:
            for season in seasons:
                try:
                    data = understat.league(league=league_name).get_team_data(season=season)
                    n_matches = 0
                    for team_id, team_info in data.items():
                        team = team_info.get("title", "")
                        if not team:
                            continue
                        matches = team_info.get("history", [])
                        for m in matches:
                            date_str = m.get("date", "")[:10]  # YYYY-MM-DD
                            xg  = float(m.get("xG",  0) or 0)
                            xga = float(m.get("xGA", 0) or 0)
                            if date_str:
                                history.setdefault(team, []).append((date_str, xg, xga))
                                n_matches += 1
                    logger.info(f"  Understat {league_name}/{season}: {len(data)} équipes, {n_matches} matchs")
                    time.sleep(0.4)   # politesse envers le serveur
                except Exception as e:
                    logger.warning(f"  Understat {league_name}/{season} erreur: {e}")

    # Tri chronologique par équipe
    for team in history:
        history[team].sort(key=lambda x: x[0])

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(XG_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f)

    total_matches = sum(len(v) for v in history.values())
    logger.info(f"xG Understat sauvegardé : {XG_PATH} ({len(history)} équipes, {total_matches} entrées)")
    return history


# ════════════════════════════════════════════════════════════
# CHARGEMENT
# ════════════════════════════════════════════════════════════

def load_xg_history() -> dict:
    """Charge et met en cache le fichier JSON. Retourne {} si absent."""
    global _xg_cache
    if _xg_cache is not None:
        return _xg_cache
    if not os.path.exists(XG_PATH):
        logger.debug("understat_xg_history.json absent — features xG désactivées")
        _xg_cache = {}
        return _xg_cache
    try:
        with open(XG_PATH, encoding="utf-8") as f:
            _xg_cache = json.load(f)
        logger.info(f"xG Understat chargé : {len(_xg_cache)} équipes")
    except Exception as e:
        logger.warning(f"Erreur lecture xG Understat : {e}")
        _xg_cache = {}
    return _xg_cache


# ════════════════════════════════════════════════════════════
# LOOKUP
# ════════════════════════════════════════════════════════════

def _normalize(name: str) -> str:
    """Normalise un nom d'équipe pour la comparaison floue."""
    return name.lower().replace("-", " ").replace(".", "").strip()


def _find_team(history: dict, team_name: str) -> str | None:
    """
    Trouve le nom canonique Understat le plus proche de team_name.
    Utilise une correspondance exacte puis difflib avec cutoff 0.6.
    Retourne None si aucune correspondance trouvée.
    """
    norm_target = _normalize(team_name)
    norm_map    = {_normalize(k): k for k in history}

    # Correspondance exacte
    if norm_target in norm_map:
        return norm_map[norm_target]

    # Correspondance floue
    matches = get_close_matches(norm_target, norm_map.keys(), n=1, cutoff=0.6)
    if matches:
        return norm_map[matches[0]]

    # Fallback : correspondance premier mot (ex. "Manchester" → "Manchester City")
    first_word = norm_target.split()[0] if norm_target else ""
    if first_word:
        candidates = [k for k in norm_map if k.startswith(first_word)]
        if len(candidates) == 1:
            return norm_map[candidates[0]]

    return None


def get_team_xg_rolling(history: dict, team_name: str,
                        before_date, window: int = 8) -> dict:
    """
    Retourne les moyennes xG des `window` derniers matchs avant `before_date`.
    `before_date` peut être str (YYYY-MM-DD) ou pandas Timestamp.

    Returns:
        {"xg_avg": float, "xga_avg": float}  (0.0 si données absentes)
    """
    default = {"xg_avg": 0.0, "xga_avg": 0.0}
    if not history:
        return default

    canonical = _find_team(history, team_name)
    if canonical is None:
        return default

    # Normalise before_date en string YYYY-MM-DD
    if hasattr(before_date, "strftime"):
        before_date = before_date.strftime("%Y-%m-%d")
    else:
        before_date = str(before_date)[:10]

    matches = history[canonical]  # [(date_str, xg, xga), ...] trié chronologique

    # Filtre matches avant la date cible
    past = [(d, xg, xga) for d, xg, xga in matches if d < before_date]
    if not past:
        return default

    recent = past[-window:]
    xg_avg  = round(sum(m[1] for m in recent) / len(recent), 4)
    xga_avg = round(sum(m[2] for m in recent) / len(recent), 4)
    return {"xg_avg": xg_avg, "xga_avg": xga_avg}


# ════════════════════════════════════════════════════════════
# ENTRÉE PRINCIPALE (pour refresh manuel)
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    history = fetch_and_save_xg_history()
    print(f"OK — {len(history)} équipes")
    # Exemple de lookup
    h = load_xg_history()
    sample = get_team_xg_rolling(h, "Arsenal", "2024-04-01")
    print(f"Arsenal xG avant 2024-04-01 : {sample}")
