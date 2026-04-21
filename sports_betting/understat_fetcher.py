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
# AK — PLAYER-LEVEL xG (contribution par joueur)
# ════════════════════════════════════════════════════════════

PLAYER_XG_PATH = os.path.join(DATA_DIR, "understat_player_xg.json")

# Cache en mémoire
_player_xg_cache: dict | None = None


def fetch_and_save_player_xg(seasons: list | None = None) -> dict:
    """
    Récupère les stats xG par joueur depuis understat.com.
    Stocke : {team_name: [(player_name, apps, xg_total, xg_per_game), ...]}
    Sauvegarde dans data/understat_player_xg.json.
    """
    try:
        import understatapi
    except ImportError:
        logger.error("understatapi non installé — pip install understatapi")
        return {}

    seasons = seasons or ["2024", "2023"]   # les 2 saisons les plus récentes suffisent
    player_data: dict[str, list] = {}

    with understatapi.UnderstatClient() as understat:
        for league_name in UNDERSTAT_LEAGUES:
            for season in seasons:
                try:
                    teams_data = understat.league(league=league_name).get_team_data(season=season)
                    for team_id, team_info in teams_data.items():
                        team = team_info.get("title", "")
                        if not team:
                            continue
                        try:
                            players = understat.team(team=team).get_player_data(season=season)
                            team_players = []
                            for p in players:
                                apps   = int(p.get("games", 0) or 0)
                                xg_tot = float(p.get("xG", 0) or 0)
                                if apps < 3:   # trop peu de matchs pour être fiable
                                    continue
                                team_players.append({
                                    "name":       p.get("player_name", ""),
                                    "apps":       apps,
                                    "xg_total":   round(xg_tot, 3),
                                    "xg_per_game": round(xg_tot / apps, 4) if apps > 0 else 0.0,
                                })
                            team_players.sort(key=lambda x: x["xg_per_game"], reverse=True)
                            player_data[team] = team_players
                            time.sleep(0.3)
                        except Exception as e:
                            logger.debug(f"  Player xG {team}/{season}: {e}")
                except Exception as e:
                    logger.warning(f"  Player xG {league_name}/{season}: {e}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PLAYER_XG_PATH, "w", encoding="utf-8") as f:
        json.dump(player_data, f)

    n_teams   = len(player_data)
    n_players = sum(len(v) for v in player_data.values())
    logger.info(f"Player xG sauvegardé : {PLAYER_XG_PATH} ({n_teams} équipes, {n_players} joueurs)")
    return player_data


def load_player_xg() -> dict:
    """Charge et met en cache le fichier player xG. Retourne {} si absent."""
    global _player_xg_cache
    if _player_xg_cache is not None:
        return _player_xg_cache
    if not os.path.exists(PLAYER_XG_PATH):
        logger.debug("understat_player_xg.json absent — features player xG désactivées")
        _player_xg_cache = {}
        return _player_xg_cache
    try:
        with open(PLAYER_XG_PATH, encoding="utf-8") as f:
            _player_xg_cache = json.load(f)
        n = sum(len(v) for v in _player_xg_cache.values())
        logger.info(f"Player xG chargé : {len(_player_xg_cache)} équipes, {n} joueurs")
    except Exception as e:
        logger.warning(f"Erreur lecture player xG : {e}")
        _player_xg_cache = {}
    return _player_xg_cache


def get_player_xg_loss(player_xg: dict, team_name: str,
                       n_absent: int = 0) -> dict:
    """
    Estime la perte xG due aux absences dans l'équipe.

    n_absent : nombre de joueurs OUT (depuis le rapport ESPN).
    Hypothèse : les joueurs absents sont tirés parmi les top contributeurs
    xG (worst-case : les meilleurs attaquants manquent).

    Retourne :
        {
          "xg_loss":             float,   # perte xG par match estimée
          "xg_loss_pct":         float,   # % de xG équipe perdu
          "top_player_xg_share": float,   # part xG du meilleur joueur
        }
    """
    default = {"xg_loss": 0.0, "xg_loss_pct": 0.0, "top_player_xg_share": 0.0}
    if not player_xg or n_absent <= 0:
        return default

    canonical = _find_team(player_xg, team_name)
    if canonical is None:
        return default

    players = player_xg[canonical]   # [{name, apps, xg_total, xg_per_game}, ...]
    if not players:
        return default

    # xG total de l'équipe = somme par joueur (sert à normaliser)
    total_xg_per_game = sum(p["xg_per_game"] for p in players)
    if total_xg_per_game <= 0:
        return default

    # Top contributeur
    top_share = round(players[0]["xg_per_game"] / total_xg_per_game, 4)

    # xG perdu = somme des n_absent meilleurs contributeurs (worst-case)
    lost = sum(p["xg_per_game"] for p in players[:n_absent])
    xg_loss_pct = round(min(lost / total_xg_per_game, 1.0), 4)

    return {
        "xg_loss":             round(lost,     4),
        "xg_loss_pct":         xg_loss_pct,
        "top_player_xg_share": top_share,
    }


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
