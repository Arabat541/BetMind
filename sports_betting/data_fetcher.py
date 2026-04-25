# ============================================================
# data_fetcher.py — Collecte des données
# Sources : football-data.org (foot) + BallDontLie (NBA) + The Odds API
# ============================================================

import json
import os
import time
import logging
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from config import (
    FOOTBALL_DATA_KEY, FOOTBALL_DATA_BASE,
    BALLDONTLIE_BASE, ODDS_API_BASE, THE_ODDS_API_KEY,
    ODDS_API_IO_KEY, ODDS_API_IO_BASE,
    API_FOOTBALL_KEY, API_FOOTBALL_BASE,
    FOOTBALL_LEAGUES, NBA_SEASON, DATA_DIR, DB_PATH,
    FORM_WINDOW_LONG, FD_DAILY_LIMIT,
    OPENWEATHER_KEY, STADIUM_COORDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cache en mémoire pour les stats NBA (évite les 429)
_nba_stats_cache = {}

# ── football-data.org : cache journalier + compteur de quota ────
_fd_cache: dict = {}          # cache endpoint+params → réponse
_fd_request_count: int = 0
_fd_request_date: str = ""


def _fd_cache_path(date: str) -> str:
    """Chemin du fichier cache pour une date donnée."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"fd_cache_{date}.json")


def _load_fd_cache() -> None:
    """Charge le cache du jour depuis le disque au démarrage du service."""
    global _fd_cache, _fd_request_count, _fd_request_date
    today = datetime.now().strftime("%Y-%m-%d")
    path  = _fd_cache_path(today)
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") != today:
            return
        _fd_cache         = data.get("cache", {})
        _fd_request_count = int(data.get("count", 0))
        _fd_request_date  = today
        logger.info(
            f"football-data.org: cache restauré ({len(_fd_cache)} entrées, "
            f"{_fd_request_count}/{FD_DAILY_LIMIT} requêtes)."
        )
    except Exception as e:
        logger.warning(f"Impossible de charger le cache FD: {e}")


def _save_fd_cache() -> None:
    """Persiste le cache courant sur le disque."""
    today = datetime.now().strftime("%Y-%m-%d")
    path  = _fd_cache_path(today)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"date": today, "count": _fd_request_count,
                       "cache": _fd_cache}, f)
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder le cache FD: {e}")


def _purge_old_fd_caches() -> None:
    """Supprime les fichiers cache de plus de 2 jours."""
    try:
        for fname in os.listdir(DATA_DIR):
            if fname.startswith("fd_cache_") and fname.endswith(".json"):
                date_str = fname[9:19]   # fd_cache_YYYY-MM-DD.json
                if date_str < (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"):
                    os.remove(os.path.join(DATA_DIR, fname))
    except Exception:
        pass


def get_fd_quota_used() -> int:
    """Retourne le nombre de requêtes football-data.org utilisées aujourd'hui."""
    return _fd_request_count


# Chargement automatique du cache au démarrage du module
_load_fd_cache()


def _http_get_with_retry(url: str, headers: dict = {}, params: dict = {},
                         max_retries: int = 3, timeout: int = 10) -> requests.Response:
    """GET avec backoff exponentiel sur 429 et erreurs 5xx (1s → 2s → 4s)."""
    last_exc: Exception = RuntimeError(f"Échec après {max_retries} tentatives: {url}")
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt          # 1s, 2s, 4s
                logger.warning(
                    f"HTTP {r.status_code} — attente {wait}s "
                    f"(tentative {attempt + 1}/{max_retries}) : {url}"
                )
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            logger.warning(
                f"Erreur réseau — attente {wait}s "
                f"(tentative {attempt + 1}/{max_retries}) : {e}"
            )
            last_exc = e
            time.sleep(wait)
    raise last_exc


def _fd_get(endpoint: str, params: dict = {}) -> dict:
    """Wrapper football-data.org avec cache journalier persistant et protection quota."""
    global _fd_cache, _fd_request_count, _fd_request_date

    today = datetime.now().strftime("%Y-%m-%d")
    if today != _fd_request_date:
        _fd_request_date  = today
        _fd_request_count = 0
        _fd_cache         = {}
        _purge_old_fd_caches()
        logger.info("football-data.org: compteur et cache réinitialisés (nouveau jour).")

    cache_key = f"{endpoint}|{sorted(params.items())}"
    if cache_key in _fd_cache:
        logger.debug(f"football-data.org cache hit: {endpoint}")
        return _fd_cache[cache_key]

    if _fd_request_count >= FD_DAILY_LIMIT:
        logger.warning(
            f"Quota football-data.org atteint ({_fd_request_count}/{FD_DAILY_LIMIT}). "
            f"Requête ignorée: {endpoint}"
        )
        return {}

    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    url = f"{FOOTBALL_DATA_BASE}/{endpoint}"
    try:
        r = _http_get_with_retry(url, headers=headers, params=params)
        data = r.json()
        _fd_request_count += 1
        _fd_cache[cache_key] = data
        _save_fd_cache()   # persiste immédiatement sur disque
        logger.info(
            f"football-data.org [{_fd_request_count}/{FD_DAILY_LIMIT}]: {endpoint}"
        )
        return data
    except Exception as e:
        logger.error(f"football-data.org error [{endpoint}]: {e}")
        return {}


_AFOOTBALL_LEAGUE_IDS = {
    "soccer_france_ligue_one":    61,
    "soccer_epl":                 39,
    "soccer_spain_la_liga":      140,
    "soccer_italy_serie_a":      135,
    "soccer_germany_bundesliga":  78,
    "soccer_uefa_champs_league":   2,
    "basketball_nba":             12,
}

# odds-api.io : slugs des ligues (différent de The Odds API)
_ODDS_IO_LEAGUE_SLUGS = {
    "soccer_france_ligue_one":   "france-ligue-1",
    "soccer_epl":                "england-premier-league",
    "soccer_spain_la_liga":      "spain-laliga",
    "soccer_italy_serie_a":      "italy-serie-a",
    "soccer_germany_bundesliga": "germany-bundesliga",
    "soccer_uefa_champs_league": "international-clubs-uefa-champions-league",
    "basketball_nba":            "usa-nba",
}


def _odds_get_io(league_key: str, market: str = "h2h") -> list:
    """
    Fallback 1 — odds-api.io (100 req/heure gratuit).
    Workflow : /events → filtre pending → /odds par event → format The Odds API.
    """
    if not ODDS_API_IO_KEY:
        return []
    slug = _ODDS_IO_LEAGUE_SLUGS.get(league_key)
    if not slug:
        return []

    base = ODDS_API_IO_BASE
    key  = ODDS_API_IO_KEY
    sport = "basketball" if "nba" in league_key else "football"
    cutoff = datetime.now() + timedelta(days=7)

    try:
        r = _http_get_with_retry(f"{base}/events",
                                  params={"apiKey": key, "sport": sport, "league": slug})
        events_raw = r.json()
        if not isinstance(events_raw, list):
            return []
    except Exception as e:
        logger.debug("odds-api.io events error [%s]: %s", league_key, e)
        return []

    # Filtrer les matchs à venir dans les 7 jours
    pending = [
        e for e in events_raw
        if e.get("status") == "pending"
        and datetime.fromisoformat(e["date"].replace("Z", "+00:00")).replace(tzinfo=None) < cutoff
    ]

    results = []
    for ev in pending[:10]:   # max 10 events par ligue pour économiser le quota
        eid       = ev["id"]
        home_name = ev.get("home", "")
        away_name = ev.get("away", "")
        try:
            ro = _http_get_with_retry(f"{base}/odds",
                                       params={"apiKey": key, "eventId": eid,
                                               "markets": "1x2" if market == "h2h" else "over-under",
                                               "bookmakers": "Bet365"})
            od = ro.json()
        except Exception:
            continue

        bookmakers_out = []
        for bk_name, bk_markets in (od.get("bookmakers") or {}).items():
            outcomes = []
            for mkt in (bk_markets if isinstance(bk_markets, list) else []):
                mkt_name = mkt.get("name", "")
                # Marché 1X2 → ML (Money Line)
                if market == "h2h" and mkt_name in ("ML", "1X2", "Match Winner"):
                    for o in mkt.get("odds", []):
                        h = float(o.get("home", 0) or 0)
                        d = float(o.get("draw", 0) or 0)
                        a = float(o.get("away", 0) or 0)
                        if h and d and a:
                            outcomes = [
                                {"name": home_name, "price": h},
                                {"name": "Draw",    "price": d},
                                {"name": away_name, "price": a},
                            ]
                # Marché Over/Under
                elif market == "totals" and "over" in mkt_name.lower():
                    for o in mkt.get("odds", []):
                        ov = float(o.get("over", 0) or 0)
                        un = float(o.get("under", 0) or 0)
                        if ov and un:
                            outcomes = [
                                {"name": "Over",  "price": ov, "point": 2.5},
                                {"name": "Under", "price": un, "point": 2.5},
                            ]
            if outcomes:
                bookmakers_out.append({
                    "title": bk_name,
                    "markets": [{"key": market, "outcomes": outcomes}],
                })

        if bookmakers_out:
            results.append({
                "id":           str(eid),
                "home_team":    home_name,
                "away_team":    away_name,
                "commence_time": ev.get("date", ""),
                "bookmakers":   bookmakers_out,
            })

    if results:
        logger.info("odds-api.io OK [%s]: %d events", league_key, len(results))
    return results


def _odds_get_api_football(league_key: str, params: dict) -> list:
    """
    Fallback 2 — api-sports.io (100 req/j gratuit).
    Convertit la réponse au format The Odds API {home_team, away_team, bookmakers[]}.
    """
    if not API_FOOTBALL_KEY:
        return []
    league_id = _AFOOTBALL_LEAGUE_IDS.get(league_key)
    if not league_id:
        return []
    market = params.get("markets", "h2h")
    now = datetime.now()
    season = now.year if now.month >= 8 else now.year - 1
    af_params = {"league": league_id, "season": season, "bookmaker": 8}  # bookmaker 8 = Bet365
    if market == "totals":
        af_params["bet"] = 5   # bet ID 5 = Over/Under 2.5
    else:
        af_params["bet"] = 1   # bet ID 1 = Match Winner (1X2)

    try:
        r = _http_get_with_retry(
            f"{API_FOOTBALL_BASE}/odds",
            headers={"x-apisports-key": API_FOOTBALL_KEY},
            params=af_params,
        )
        response = r.json().get("response", [])
    except Exception as e:
        logger.debug("api-football odds error: %s", e)
        return []

    events = []
    for item in response:
        fix = item.get("fixture", {})
        teams = item.get("teams", {})
        home_team = teams.get("home", {}).get("name", "")
        away_team = teams.get("away", {}).get("name", "")
        if not home_team or not away_team:
            continue
        bookmakers = []
        for bk in item.get("bookmakers", []):
            markets_out = []
            for bet in bk.get("bets", []):
                if market == "h2h" and bet.get("id") != 1:
                    continue
                if market == "totals" and bet.get("id") != 5:
                    continue
                outcomes = []
                for v in bet.get("values", []):
                    label = v.get("value", "")
                    price = float(v.get("odd", 0) or 0)
                    if market == "h2h":
                        name = home_team if label == "Home" else (away_team if label == "Away" else "Draw")
                    else:
                        name = label   # "Over" / "Under"
                        point = 2.5
                    entry = {"name": name, "price": price}
                    if market == "totals":
                        entry["point"] = 2.5
                    outcomes.append(entry)
                if outcomes:
                    markets_out.append({"key": market, "outcomes": outcomes})
            if markets_out:
                bookmakers.append({"title": bk.get("name", ""), "markets": markets_out})
        events.append({
            "id":           str(fix.get("id", "")),
            "home_team":    home_team,
            "away_team":    away_team,
            "commence_time": fix.get("date", ""),
            "bookmakers":   bookmakers,
        })
    if events:
        logger.info("api-football fallback OK [%s]: %d events", league_key, len(events))
    return events


def _odds_get(endpoint: str, params: dict, ttl: int = 300) -> list:
    """
    Wrapper Odds API avec cascade de fallbacks :
    1. The Odds API (the-odds-api.com) — 500 req/mois gratuit
    2. odds-api.io                      — 100 req/heure gratuit
    3. api-sports.io (API-Football)     — 100 req/jour gratuit
    """
    from cache import cached_get
    cache_params = {k: v for k, v in params.items() if k != "apiKey"}
    cache_key = f"odds:{endpoint}:{sorted(cache_params.items())}"

    def _fetch():
        # ── Source 1 : The Odds API ───────────────────────────
        if THE_ODDS_API_KEY:
            p = dict(params)
            p["apiKey"] = THE_ODDS_API_KEY
            url = f"{ODDS_API_BASE}/{endpoint}"
            try:
                r = _http_get_with_retry(url, params=p)
                data = r.json()
                if isinstance(data, list):
                    return data
                # Quota épuisé ou 401 → essayer fallbacks
                err = data.get("error_code", "") if isinstance(data, dict) else ""
                if err:
                    logger.warning("The Odds API quota/auth: %s — fallback activé", err)
            except Exception as e:
                logger.error("Odds API error [%s]: %s", endpoint, e)

        # ── Source 2 : odds-api.io ────────────────────────────
        league_key = endpoint.replace("sports/", "").replace("/odds", "")
        market     = params.get("markets", "h2h")
        result = _odds_get_io(league_key, market)
        if result:
            return result

        # ── Source 3 : API-Football ───────────────────────────
        result = _odds_get_api_football(league_key, params)
        if result:
            return result

        return []

    return cached_get(cache_key, _fetch, ttl=ttl)


def _balldontlie_get(endpoint: str, params: dict = {}) -> dict:
    """Wrapper pour BallDontLie (NBA)."""
    from config import BALLDONTLIE_KEY
    url = f"{BALLDONTLIE_BASE}/{endpoint}"
    headers = {"Authorization": BALLDONTLIE_KEY} if BALLDONTLIE_KEY else {}
    try:
        r = _http_get_with_retry(url, headers=headers, params=params)
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
    home_form_chars, away_form_chars = [], []

    for m in matches[-15:]:
        is_home = m["homeTeam"]["id"] == team_id
        hg = m.get("score", {}).get("fullTime", {}).get("home") or 0
        ag = m.get("score", {}).get("fullTime", {}).get("away") or 0
        gf = hg if is_home else ag
        ga = ag if is_home else hg

        goals_for.append(gf)
        goals_ag.append(ga)

        result_char = "W" if gf > ga else ("D" if gf == ga else "L")
        form_chars.append(result_char)
        if is_home:
            home_form_chars.append(result_char)
        else:
            away_form_chars.append(result_char)

        if gf > ga:    wins   += 1
        elif gf == ga: draws  += 1
        else:          losses += 1

        if ga == 0: clean_sheets += 1
        if gf == 0: failed_to_score += 1

    n = len(goals_for)
    if n == 0:
        return {}

    # ── Fatigue — E ─────────────────────────────────────────
    today        = datetime.now()
    match_dates  = []
    for m in matches[-15:]:
        d_str = m.get("utcDate", "")
        try:
            from datetime import timezone
            dt = datetime.fromisoformat(d_str.replace("Z", "+00:00"))
            match_dates.append(dt.replace(tzinfo=None))
        except Exception:
            pass

    if match_dates:
        last_match      = max(match_dates)
        days_since_last = max(0, (today - last_match).days)
        matches_10days  = sum(1 for d in match_dates if (today - d).days <= 10)
    else:
        days_since_last = 7
        matches_10days  = 1

    form_str = "".join(form_chars)
    return {
        "team_id":              team_id,
        "played":               n,
        "wins":                 wins,
        "draws":                draws,
        "losses":               losses,
        "goals_for_avg":        round(sum(goals_for) / n, 4),
        "goals_ag_avg":         round(sum(goals_ag) / n, 4),
        "form_string":          form_str,
        "form_score":           _form_to_score(form_str),
        "form_score_long":      _form_to_score(form_str, window=FORM_WINDOW_LONG),
        "home_form_score":      _form_to_score("".join(home_form_chars)),
        "away_form_score":      _form_to_score("".join(away_form_chars)),
        "clean_sheets":         clean_sheets,
        "failed_to_score":      failed_to_score,
        # Fatigue
        "days_since_last_match": days_since_last,
        "matches_last_10days":   matches_10days,
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


def get_team_standing(standings_df: pd.DataFrame, team_id: int) -> dict:
    """Retourne pts_per_game, rang et features de motivation depuis le classement."""
    default = {
        "rank": 10, "pts_per_game": 1.0,
        "relegation_gap": 0.0, "title_gap": 0.5,
    }
    if standings_df is None or standings_df.empty:
        return default
    n_teams = max(len(standings_df), 1)
    rel_zone = n_teams - 3   # seuil bas de la zone de relégation
    row = standings_df[standings_df["team_id"] == team_id]
    if row.empty:
        return default
    r      = row.iloc[0]
    played = int(r.get("played", 1)) or 1
    rank   = int(r.get("position", n_teams // 2))
    return {
        "rank":            rank,
        "pts_per_game":    round(r.get("points", 0) / played, 4),
        # Motivation — G
        # positif = en sécurité, négatif = en zone de relégation
        "relegation_gap":  round((rel_zone - rank) / n_teams, 4),
        # 0 = leader, 1 = dernier
        "title_gap":       round((rank - 1) / n_teams, 4),
    }


# ── Météo (OpenWeatherMap — J) ───────────────────────────────
_weather_cache: dict = {}

def fetch_match_weather(home_team: str, match_datetime=None) -> dict:
    """
    Retourne les conditions météo prévues au stade du club domicile.
    Nécessite OPENWEATHER_KEY dans .env (plan gratuit, 60 req/min).
    Retourne rainy_match=0 si clé absente ou équipe inconnue.
    """
    default = {"rainy_match": 0, "rain_mm": 0.0, "wind_kmh": 0.0, "temp_c": 20.0}

    if not OPENWEATHER_KEY:
        return default

    # Chercher les coords du stade (recherche partielle sur le nom)
    coords = STADIUM_COORDS.get(home_team)
    if not coords:
        first_word = home_team.split()[0].lower()
        for k, v in STADIUM_COORDS.items():
            if k.lower().startswith(first_word):
                coords = v
                break
    if not coords:
        return default

    lat, lon = coords
    cache_key = f"{lat:.2f},{lon:.2f}"

    if cache_key in _weather_cache:
        return _weather_cache[cache_key]

    try:
        url    = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY,
                  "units": "metric", "cnt": 8}   # 8 × 3h = 24h
        resp   = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data   = resp.json()

        # Prendre la prévision la plus proche de l'heure du match (ou la première)
        entry     = data["list"][0]
        rain_mm   = entry.get("rain", {}).get("3h", 0.0)
        wind_ms   = entry.get("wind", {}).get("speed", 0.0)
        wind_kmh  = round(wind_ms * 3.6, 1)
        temp_c    = entry.get("main", {}).get("temp", 20.0)
        rainy     = 1 if (rain_mm > 2.0 or wind_kmh > 30.0) else 0

        result = {"rainy_match": rainy, "rain_mm": rain_mm,
                  "wind_kmh": wind_kmh, "temp_c": temp_c}
        _weather_cache[cache_key] = result
        logger.debug(f"Météo {home_team}: pluie={rain_mm}mm vent={wind_kmh}km/h → rainy={rainy}")
        return result

    except Exception as e:
        logger.warning(f"fetch_match_weather({home_team}): {e}")
        return default


# ── ELO lookup (chargé depuis data/elo_ratings_current.json) ──
_elo_cache: dict = {}

def get_team_elo(team_name: str, default: float = 1500.0) -> float:
    """
    Retourne le rating ELO courant d'une équipe.
    Données générées par train_from_csv.py → data/elo_ratings_current.json.
    Fallback : 1500 (rating initial) si équipe inconnue.
    """
    global _elo_cache
    if not _elo_cache:
        elo_path = os.path.join(DATA_DIR, "elo_ratings_current.json")
        if os.path.exists(elo_path):
            try:
                with open(elo_path, "r", encoding="utf-8") as f:
                    _elo_cache = json.load(f)
            except Exception:
                _elo_cache = {}

    # Recherche exacte puis sur premier mot
    val = _elo_cache.get(team_name)
    if val is None:
        first = team_name.split()[0].lower()
        for k, v in _elo_cache.items():
            if k.lower().startswith(first):
                val = v
                break
    return float(val) if val is not None else default


# ── Shots lookup (chargé depuis data/team_shots_current.json) ─
_team_shots_cache: dict = {}

def get_team_shots_stats(team_name: str, league: str) -> dict:
    """
    Retourne les moyennes de tirs de la saison courante pour une équipe.
    Données générées par train_from_csv.py → data/team_shots_current.json.
    Fallback : 0.0 si équipe ou fichier non trouvés.
    """
    global _team_shots_cache
    if not _team_shots_cache:
        shots_path = os.path.join(DATA_DIR, "team_shots_current.json")
        if os.path.exists(shots_path):
            try:
                with open(shots_path, "r", encoding="utf-8") as f:
                    _team_shots_cache = json.load(f)
            except Exception:
                _team_shots_cache = {}

    league_data = _team_shots_cache.get(league, {})

    # Recherche exacte puis sur premier mot (gère "Arsenal" vs "Arsenal FC")
    team_data = league_data.get(team_name)
    if not team_data:
        first = team_name.split()[0].lower()
        for k, v in league_data.items():
            if k.lower().startswith(first):
                team_data = v
                break

    if not team_data:
        return {"sot_avg": 0.0, "shots_avg": 0.0, "sot_ag_avg": 0.0, "sot_ratio": 0.0}
    return {
        "sot_avg":    float(team_data.get("sot_avg",    0.0)),
        "shots_avg":  float(team_data.get("shots_avg",  0.0)),
        "sot_ag_avg": float(team_data.get("sot_ag_avg", 0.0)),
        "sot_ratio":  float(team_data.get("sot_ratio",  0.0)),
    }


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


def fetch_football_ou_odds(league_key: str = "soccer_france_ligue_one") -> pd.DataFrame:
    """Cotes Over/Under 2.5 buts via The Odds API (marché 'totals')."""
    data = _odds_get(f"sports/{league_key}/odds", {
        "regions": "eu", "markets": "totals", "oddsFormat": "decimal"
    })
    rows = []
    for event in (data if isinstance(data, list) else []):
        home, away = event.get("home_team", ""), event.get("away_team", "")
        for bookmaker in event.get("bookmakers", [])[:3]:
            for market in bookmaker.get("markets", []):
                if market["key"] != "totals":
                    continue
                outcomes = {}
                for o in market["outcomes"]:
                    if abs(o.get("point", 0) - 2.5) < 0.01:   # ligne 2.5 uniquement
                        outcomes[o["name"]] = o["price"]
                if "Over" in outcomes and "Under" in outcomes:
                    rows.append({
                        "event_id":  event["id"],
                        "home_team": home, "away_team": away,
                        "odd_over":  outcomes["Over"],
                        "odd_under": outcomes["Under"],
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby(["event_id", "home_team", "away_team"], as_index=False).agg(
            odd_over=("odd_over", "mean"), odd_under=("odd_under", "mean"))
        df["impl_over"]  = 1 / df["odd_over"]
        df["impl_under"] = 1 / df["odd_under"]
        margin = df["impl_over"] + df["impl_under"]
        df[["impl_over", "impl_under"]] = df[["impl_over", "impl_under"]].div(margin, axis=0)
    logger.info(f"OU odds fetched: {len(df)} events ({league_key})")
    return df


def fetch_football_ah_odds(league_key: str = "soccer_france_ligue_one") -> pd.DataFrame:
    """
    Cotes Asian Handicap -0.5 via The Odds API (marché 'asian_handicap').
    AH -0.5 = équipe doit gagner (élimine le nul, binaire H/A).
    """
    data = _odds_get(f"sports/{league_key}/odds", {
        "regions": "eu", "markets": "asian_handicap", "oddsFormat": "decimal"
    })
    rows = []
    for event in (data if isinstance(data, list) else []):
        home, away = event.get("home_team", ""), event.get("away_team", "")
        for bookmaker in event.get("bookmakers", [])[:3]:
            for market in bookmaker.get("markets", []):
                if market["key"] != "asian_handicap":
                    continue
                outcomes = {}
                for o in market["outcomes"]:
                    point = o.get("point", 0)
                    if abs(point + 0.5) < 0.01:    # AH -0.5 pour le favori
                        outcomes["home_minus"] = o["price"]
                    elif abs(point - 0.5) < 0.01:  # AH +0.5 pour l'outsider
                        outcomes["away_plus"] = o["price"]
                if "home_minus" in outcomes and "away_plus" in outcomes:
                    rows.append({
                        "event_id":     event["id"],
                        "home_team":    home,
                        "away_team":    away,
                        "odd_ah_home":  outcomes["home_minus"],
                        "odd_ah_away":  outcomes["away_plus"],
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby(["event_id", "home_team", "away_team"], as_index=False).agg(
            odd_ah_home=("odd_ah_home", "mean"),
            odd_ah_away=("odd_ah_away", "mean"),
        )
    logger.info(f"AH odds fetched: {len(df)} events ({league_key})")
    return df


def fetch_football_btts_odds(league_key: str = "soccer_france_ligue_one") -> pd.DataFrame:
    """Cotes BTTS (Both Teams To Score) via The Odds API (marché 'both_teams_to_score')."""
    data = _odds_get(f"sports/{league_key}/odds", {
        "regions": "eu", "markets": "both_teams_to_score", "oddsFormat": "decimal"
    })
    rows = []
    for event in (data if isinstance(data, list) else []):
        home, away = event.get("home_team", ""), event.get("away_team", "")
        for bookmaker in event.get("bookmakers", [])[:3]:
            for market in bookmaker.get("markets", []):
                if market["key"] != "both_teams_to_score":
                    continue
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                if "Yes" in outcomes and "No" in outcomes:
                    rows.append({
                        "event_id":    event["id"],
                        "home_team":   home,
                        "away_team":   away,
                        "odd_btts":    outcomes["Yes"],
                        "odd_no_btts": outcomes["No"],
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby(["event_id", "home_team", "away_team"], as_index=False).agg(
            odd_btts=("odd_btts", "mean"), odd_no_btts=("odd_no_btts", "mean"))
    logger.info(f"BTTS odds fetched: {len(df)} events ({league_key})")
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
    # NBA season year = start year (2025 for the 2025-26 season)
    now = datetime.now()
    nba_season = now.year if now.month >= 10 else now.year - 1
    data = _balldontlie_get("games", {
        "team_ids[]": team_id,
        "per_page":   last_n,
        "seasons[]":  nba_season
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


# ════════════════════════════════════════════════════════════
# NBA INJURIES (ESPN API non officielle — gratuit)
# ════════════════════════════════════════════════════════════

_ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
_ESPN_HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_nba_injuries() -> dict:
    """
    Retourne les blessures NBA du jour depuis ESPN.
    Dict: { "Atlanta Hawks": {"out": 1, "day_to_day": 3, "impact": 0.28}, ... }
    Impact = (out + 0.4 * day_to_day) / 5, borné à [0, 1].
    """
    try:
        r = requests.get(_ESPN_INJURIES_URL, timeout=10, headers=_ESPN_HEADERS)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"ESPN injuries fetch failed: {e}")
        return {}

    result = {}
    for team in data.get("injuries", []):
        name = team.get("displayName", "")
        players = team.get("injuries", [])
        out = sum(1 for p in players if p.get("status", "").lower() == "out")
        dtd = sum(1 for p in players
                  if any(s in p.get("status", "").lower()
                         for s in ("day-to-day", "questionable", "doubtful")))
        impact = min(out * 1.0 + dtd * 0.4, 5.0) / 5.0
        result[name] = {
            "out":        out,
            "day_to_day": dtd,
            "impact":     round(impact, 4),
        }
    logger.info(f"ESPN injuries: {len(result)} équipes récupérées")
    return result


def get_injury_stats(injuries_dict: dict, team_name: str) -> dict:
    """Lookup par nom d'équipe (match sur le dernier mot, ex: 'Celtics')."""
    default = {"out": 0, "day_to_day": 0, "impact": 0.0}
    if not injuries_dict or not team_name:
        return default
    keyword = team_name.strip().split()[-1].lower()
    for espn_name, stats in injuries_dict.items():
        if keyword in espn_name.lower():
            return stats
    return default


def init_db():
    """Initialise la base de données (SQLite ou PostgreSQL via db.py)."""
    import os
    from db import get_conn, adapt_ddl, is_postgres, raw_conn
    os.makedirs(DATA_DIR, exist_ok=True)

    # CREATE TABLE dans une transaction propre
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(adapt_ddl("""
            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at   TEXT    DEFAULT (datetime('now')),
                sport        TEXT, league TEXT, home_team TEXT, away_team TEXT,
                match_date   TEXT, pred_result TEXT,
                prob_home    REAL, prob_draw REAL, prob_away REAL,
                confidence   REAL, is_value_bet INTEGER DEFAULT 0,
                edge         REAL, kelly_stake REAL, odd_used REAL,
                outcome      TEXT DEFAULT NULL, pnl REAL DEFAULT NULL,
                home_injuries_out REAL DEFAULT 0, home_injuries_dtd REAL DEFAULT 0,
                away_injuries_out REAL DEFAULT 0, away_injuries_dtd REAL DEFAULT 0,
                method       TEXT DEFAULT NULL,
                odd_closing  REAL DEFAULT NULL
            )
        """))
        c.execute(adapt_ddl("""
            CREATE TABLE IF NOT EXISTS bankroll (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                updated_at TEXT DEFAULT (datetime('now')),
                balance    REAL, total_bets INTEGER, wins INTEGER, losses INTEGER, roi REAL
            )
        """))

    # Migrations colonnes — chacune dans sa propre transaction
    # PostgreSQL : utilise ADD COLUMN IF NOT EXISTS (9.6+)
    # SQLite     : try/except sur la transaction individuelle
    _migrations = [
        ("home_injuries_out",    "REAL DEFAULT 0"),
        ("home_injuries_dtd",    "REAL DEFAULT 0"),
        ("away_injuries_out",    "REAL DEFAULT 0"),
        ("away_injuries_dtd",    "REAL DEFAULT 0"),
        ("method",               "TEXT DEFAULT NULL"),
        ("odd_closing",          "REAL DEFAULT NULL"),
        ("odd_opening",          "REAL DEFAULT NULL"),
        ("opening_movement_pct", "REAL DEFAULT NULL"),
        ("market",               "TEXT DEFAULT NULL"),
        ("clv_realized",         "REAL DEFAULT NULL"),   # BB — CLV réalisé à la fermeture
    ]
    if_not_exists = "IF NOT EXISTS " if is_postgres() else ""
    for col, col_type in _migrations:
        try:
            with get_conn() as conn:
                conn.execute(
                    f"ALTER TABLE predictions ADD COLUMN {if_not_exists}{col} {col_type}"
                )
        except Exception:
            pass  # colonne déjà présente

    # Index unique requis par ON CONFLICT — COALESCE(market,'') car NULL≠NULL dans PG
    if is_postgres():
        try:
            with get_conn() as conn:
                # Supprimer les doublons avant de créer l'index (garde le MIN(id))
                conn.execute("""
                    DELETE FROM predictions
                    WHERE id NOT IN (
                        SELECT MIN(id) FROM predictions
                        GROUP BY home_team, away_team, match_date, sport, COALESCE(market,'')
                    )
                """)
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS predictions_unique_match "
                    "ON predictions (home_team, away_team, match_date, sport, COALESCE(market,''))"
                )
        except Exception as _e:
            logger.warning("Unique index creation: %s", _e)

    # PostgreSQL : resynchronise la séquence predictions_id_seq au cas où elle
    # serait désynchronisée après une migration SQLite→PG (rows copiés avec IDs explicites).
    if is_postgres():
        try:
            with get_conn() as conn:
                conn.execute(
                    "SELECT setval('predictions_id_seq', GREATEST(nextval('predictions_id_seq')-1,"
                    " (SELECT COALESCE(MAX(id),0) FROM predictions)))"
                )
        except Exception:
            pass

    logger.info("Database initialized.")


def save_prediction(pred: dict):
    from db import get_conn, ph, is_postgres
    try:
        with get_conn() as conn:
            # ON CONFLICT DO NOTHING — utilise COALESCE(market,'') qui correspond à l'index PG
            conflict_clause = (
                "ON CONFLICT (home_team, away_team, match_date, sport, COALESCE(market,'')) DO NOTHING"
                if is_postgres() else ""
            )
            conn.execute(f"""
                INSERT INTO predictions
                (sport,league,home_team,away_team,match_date,pred_result,
                 prob_home,prob_draw,prob_away,confidence,is_value_bet,edge,kelly_stake,odd_used,
                 home_injuries_out,home_injuries_dtd,away_injuries_out,away_injuries_dtd,
                 method,odd_opening,market)
                VALUES ({','.join([ph]*21)})
                {conflict_clause}
            """, (
                pred.get("sport"), pred.get("league"), pred.get("home_team"), pred.get("away_team"),
                pred.get("match_date"), pred.get("pred_result"), pred.get("prob_home"),
                pred.get("prob_draw"), pred.get("prob_away"), pred.get("confidence"),
                int(pred.get("is_value_bet", 0)), pred.get("edge"), pred.get("kelly_stake"), pred.get("odd_used"),
                pred.get("home_injuries_out", 0), pred.get("home_injuries_dtd", 0),
                pred.get("away_injuries_out", 0), pred.get("away_injuries_dtd", 0),
                pred.get("method"),
                pred.get("odd_used"),
                pred.get("market"),
            ))
    except Exception as e:
        logger.error(f"save_prediction ({pred.get('home_team')} vs {pred.get('away_team')}): {e}")
        raise


def get_team_exposure(team_name: str) -> float:
    """
    Somme des mises actives (outcome IS NULL) sur la victoire d'une équipe donnée.
    Compte les paris H où home_team = team et les paris A où away_team = team.
    Utilisé pour limiter la concentration sur un seul club (MAX_TEAM_EXPOSURE_PCT).
    """
    from db import get_conn, ph
    with get_conn() as conn:
        row = conn.execute(f"""
            SELECT COALESCE(SUM(kelly_stake), 0) FROM predictions
            WHERE outcome IS NULL AND kelly_stake > 0
              AND (
                (pred_result = 'H' AND home_team = {ph}) OR
                (pred_result = 'A' AND away_team = {ph})
              )
        """, (team_name, team_name)).fetchone()
    return float(row[0]) if row else 0.0


def count_active_bets_for_league(league: str) -> int:
    """
    Nombre de paris 1X2 actifs (outcome IS NULL, kelly_stake > 0) pour une ligue.
    Utilisé pour la diversification : max MAX_ACTIVE_BETS_PER_LEAGUE par ligue.
    Les signaux Over/Under ne sont pas comptés (marché indépendant).
    """
    from db import get_conn, ph
    with get_conn() as conn:
        row = conn.execute(f"""
            SELECT COUNT(*) FROM predictions
            WHERE league = {ph}
              AND outcome IS NULL
              AND kelly_stake > 0
              AND pred_result IN ('H', 'D', 'A')
        """, (league,)).fetchone()
    return int(row[0]) if row else 0


def get_all_predictions() -> pd.DataFrame:
    from db import get_conn
    with get_conn() as conn:
        cur  = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC")
        cols = [d[0] for d in cur.description] if cur.description else None
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=cols or [])
    return pd.DataFrame(rows, columns=cols)


def is_already_predicted(home_team: str, away_team: str, match_date: str) -> bool:
    """
    Vérifie si une prédiction existe déjà pour ce match, quelle que soit la date
    de création. Basé sur DATE(match_date) pour éviter les paris en double sur
    plusieurs cycles (08h/18h) ou plusieurs jours avant le match.
    """
    from db import get_conn, ph
    try:
        with get_conn() as conn:
            row = conn.execute(f"""
                SELECT id FROM predictions
                WHERE home_team = {ph} AND away_team = {ph}
                  AND DATE(match_date) = DATE({ph})
                LIMIT 1
            """, (home_team, away_team, match_date)).fetchone()
        return row is not None
    except Exception as e:
        logger.error(f"is_already_predicted error: {e}")
        return False