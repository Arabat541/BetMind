# ============================================================
# transfermarkt_fetcher.py — Valeur marchande effectifs (Y)
# Scrape transfermarkt.com pour obtenir la valeur totale des
# effectifs des équipes des 5 grandes ligues.
# Sauvegarde : data/squad_values.json
# ============================================================

import json
import logging
import os
import re
import time
from difflib import get_close_matches

import requests

logger = logging.getLogger(__name__)

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
VALUES_PATH = os.path.join(DATA_DIR, "squad_values.json")

# URL Transfermarkt par ligue (edition = saison courante)
LEAGUE_URLS = {
    "E0":  "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1",
    "SP1": "https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES1",
    "D1":  "https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/L1",
    "I1":  "https://www.transfermarkt.com/serie-a/startseite/wettbewerb/IT1",
    "F1":  "https://www.transfermarkt.com/ligue-1/startseite/wettbewerb/FR1",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Cache en mémoire
_values_cache: dict | None = None


# ════════════════════════════════════════════════════════════
# SCRAPING
# ════════════════════════════════════════════════════════════

def _parse_value_string(val_str: str) -> float:
    """
    Convertit une chaîne Transfermarkt (ex: '€1.40bn', '€540m', '€2.50m')
    en float en millions d'euros.
    """
    val_str = val_str.strip().replace(",", ".").replace("€", "")
    try:
        if "bn" in val_str:
            return float(val_str.replace("bn", "").strip()) * 1000.0
        if "m" in val_str:
            return float(val_str.replace("m", "").strip())
        if "k" in val_str:
            return float(val_str.replace("k", "").strip()) / 1000.0
        return float(val_str)
    except (ValueError, AttributeError):
        return 0.0


def _scrape_league(league_code: str, url: str) -> dict:
    """
    Scrape la page d'accueil d'une ligue sur Transfermarkt.
    Retourne {team_name: squad_value_mEUR} ou {} si erreur.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Transfermarkt {league_code} erreur réseau: {e}")
        return {}

    # Extraction via regex (plus robuste que BeautifulSoup pour TM)
    # TM affiche les valeurs sous forme "€1.40bn" ou "€540m" dans les td
    teams: dict[str, float] = {}

    # Pattern: href="/teamname/startseite/verein/123" ... "€Xbn" ou "€Xm"
    # Cherche les noms d'équipes et leurs valeurs dans le tableau
    pattern_team = re.compile(
        r'href="/[^/]+/startseite/verein/(\d+)"[^>]*>\s*<[^>]+>\s*([^<]+)</(?:a|td)',
        re.DOTALL
    )
    pattern_value = re.compile(r'€\s*([\d,.]+\s*(?:bn|m|k)?)', re.IGNORECASE)

    # Approche alternative : parser les lignes du tableau principal
    # Le tableau a class="items" et chaque ligne tr contient le nom et la valeur
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "items"})
        if not table:
            logger.warning(f"Transfermarkt {league_code}: tableau non trouvé")
            return {}

        for row in table.find_all("tr", {"class": ["odd", "even"]}):
            # Nom de l'équipe
            name_tag = row.find("td", {"class": "hauptlink"})
            if not name_tag:
                name_tag = row.find("td", {"class": "no-border-links"})
            if not name_tag:
                continue
            a_tag = name_tag.find("a")
            team_name = a_tag.get_text(strip=True) if a_tag else name_tag.get_text(strip=True)

            # Valeur marchande totale (dernière td avec €)
            value_tds = row.find_all("td")
            squad_value = 0.0
            for td in reversed(value_tds):
                text = td.get_text(strip=True)
                if "€" in text or "m" in text.lower():
                    squad_value = _parse_value_string(text)
                    if squad_value > 0:
                        break

            if team_name and squad_value > 0:
                teams[team_name] = round(squad_value, 2)

    except ImportError:
        # Fallback sans BeautifulSoup : regex brut
        html = resp.text
        # Pattern simple : trouve les paires nom/valeur dans le HTML
        rows = re.findall(
            r'class="hauptlink"[^>]*>.*?<a[^>]*>([^<]+)</a>.*?'
            r'(€[\d,.]+\s*(?:bn|m|k)?)',
            html, re.DOTALL
        )
        for name, val in rows[:30]:
            name = re.sub(r'\s+', ' ', name).strip()
            if name:
                teams[name] = round(_parse_value_string(val), 2)

    logger.info(f"  Transfermarkt {league_code}: {len(teams)} équipes scrapées")
    return teams


def fetch_and_save_squad_values() -> dict:
    """
    Scrape les 5 grandes ligues sur Transfermarkt.
    Sauvegarde dans data/squad_values.json.
    Retourne {team_name: value_mEUR}.
    """
    all_values: dict[str, float] = {}

    for league_code, url in LEAGUE_URLS.items():
        result = _scrape_league(league_code, url)
        all_values.update(result)
        time.sleep(1.5)   # politesse

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(VALUES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_values, f, ensure_ascii=False, indent=2)

    logger.info(f"Squad values sauvegardées : {VALUES_PATH} ({len(all_values)} équipes)")
    return all_values


# ════════════════════════════════════════════════════════════
# CHARGEMENT
# ════════════════════════════════════════════════════════════

def load_squad_values() -> dict:
    """Charge et met en cache le fichier JSON. Retourne {} si absent."""
    global _values_cache
    if _values_cache is not None:
        return _values_cache
    if not os.path.exists(VALUES_PATH):
        logger.debug("squad_values.json absent — features valeur marchande désactivées")
        _values_cache = {}
        return _values_cache
    try:
        with open(VALUES_PATH, encoding="utf-8") as f:
            _values_cache = json.load(f)
        logger.info(f"Squad values chargées : {len(_values_cache)} équipes")
    except Exception as e:
        logger.warning(f"Erreur lecture squad_values: {e}")
        _values_cache = {}
    return _values_cache


# ════════════════════════════════════════════════════════════
# LOOKUP
# ════════════════════════════════════════════════════════════

def _normalize(name: str) -> str:
    return name.lower().replace("-", " ").replace(".", "").strip()


def get_team_squad_value(values: dict, team_name: str,
                         default: float = 100.0) -> float:
    """
    Retourne la valeur marchande de l'effectif en millions EUR.
    Fallback à `default` (100M) si l'équipe n'est pas trouvée.
    """
    if not values:
        return default

    norm_target = _normalize(team_name)
    norm_map    = {_normalize(k): v for k, v in values.items()}

    if norm_target in norm_map:
        return norm_map[norm_target]

    matches = get_close_matches(norm_target, norm_map.keys(), n=1, cutoff=0.6)
    if matches:
        return norm_map[matches[0]]

    return default


def get_squad_value_features(values: dict, home_name: str,
                             away_name: str) -> dict:
    """
    Retourne les features de valeur marchande pour un match.
    value_ratio > 1 = équipe domicile plus chère → avantage financier.
    """
    home_val = get_team_squad_value(values, home_name)
    away_val = get_team_squad_value(values, away_name)
    ratio    = round(home_val / max(away_val, 1.0), 4)
    return {
        "home_squad_value": round(home_val, 2),
        "away_squad_value": round(away_val, 2),
        "squad_value_ratio": ratio,
    }


# ════════════════════════════════════════════════════════════
# ENTRÉE PRINCIPALE (pour refresh manuel)
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("BeautifulSoup non installé — pip install beautifulsoup4")
    values = fetch_and_save_squad_values()
    print(f"OK — {len(values)} équipes")
    # Exemple de lookup
    v = load_squad_values()
    print(f"Arsenal : {get_team_squad_value(v, 'Arsenal')} M€")
    print(f"Marseille : {get_team_squad_value(v, 'Marseille')} M€")
