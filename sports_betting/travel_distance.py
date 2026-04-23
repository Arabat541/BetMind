# ============================================================
# travel_distance.py — AO : Distance déplacement Away
# ============================================================
# Calcule la distance (km) et le décalage horaire estimé entre les stades
# des équipes home et away, puis dérive un score de fatigue voyage [0,1].
#
# Coordonnées des stades : fichier JSON statique (pas d'API payante).
# Haversine maison — pas de dépendance externe.
# Décalage horaire estimé depuis la longitude (1h par ~15°).
# ============================================================

import json
import logging
import math
import os

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STADIUMS_PATH = os.path.join(SCRIPT_DIR, "data", "stadiums.json")

# Distance max pour normalisation (vol Lisbonne→Varsovie ≈ 2400 km)
_MAX_KM = 2500.0


# ════════════════════════════════════════════════════════════
# DONNÉES STADES (top 5 ligues + UCL habituels)
# ════════════════════════════════════════════════════════════

_BUILTIN_STADIUMS: dict[str, tuple[float, float]] = {
    # Premier League
    "Arsenal":              (51.5549, -0.1084),
    "Aston Villa":          (52.5090, -1.8847),
    "Bournemouth":          (50.7352, -1.8383),
    "Brentford":            (51.4883, -0.2886),
    "Brighton":             (50.8618, -0.0832),
    "Burnley":              (53.7890, -2.2302),
    "Chelsea":              (51.4816, -0.1909),
    "Crystal Palace":       (51.3983, -0.0855),
    "Everton":              (53.4388, -2.9661),
    "Fulham":               (51.4749, -0.2217),
    "Ipswich":              (52.0552, 1.1447),
    "Leeds":                (53.7773, -1.5724),
    "Leicester":            (52.6204, -1.1423),
    "Liverpool":            (53.4308, -2.9608),
    "Luton":                (51.8832, -0.4320),
    "Man City":             (53.4831, -2.2004),
    "Man United":           (53.4631, -2.2913),
    "Newcastle":            (54.9756, -1.6218),
    "Nottm Forest":         (52.9399, -1.1328),
    "Sheffield United":     (53.3703, -1.4701),
    "Southampton":          (50.9058, -1.3914),
    "Tottenham":            (51.6042, -0.0665),
    "West Ham":             (51.5388, -0.0169),
    "Wolves":               (52.5902, -2.1302),
    # La Liga
    "Alaves":               (42.8393, -2.6741),
    "Almeria":              (36.8495, -2.4146),
    "Athletic Club":        (43.2641, -2.9499),
    "Atletico Madrid":      (40.4361, -3.5994),
    "Barcelona":            (41.3809, 2.1228),
    "Betis":                (37.3564, -5.9817),
    "Cadiz":                (36.5020, -6.2730),
    "Celta Vigo":           (42.2120, -8.7405),
    "Espanyol":             (41.3481, 2.0790),
    "Getafe":               (40.3262, -3.7144),
    "Girona":               (41.9617, 2.8281),
    "Granada":              (37.1527, -3.5953),
    "Las Palmas":           (28.1003, -15.4570),
    "Leganes":              (40.3213, -3.7657),
    "Mallorca":             (39.5897, 2.6451),
    "Osasuna":              (42.7968, -1.6373),
    "Rayo Vallecano":       (40.3913, -3.6559),
    "Real Madrid":          (40.4531, -3.6883),
    "Real Sociedad":        (43.3013, -1.9737),
    "Sevilla":              (37.3841, -5.9705),
    "Valencia":             (39.4745, -0.3584),
    "Valladolid":           (41.6429, -4.7447),
    "Villarreal":           (39.9441, -0.1032),
    # Bundesliga
    "Augsburg":             (48.3233, 10.8866),
    "Bayer Leverkusen":     (51.0383, 7.0021),
    "Bayern Munich":        (48.2188, 11.6247),
    "Bochum":               (51.4901, 7.2360),
    "Borussia Dortmund":    (51.4926, 7.4518),
    "Borussia Mgladbach":   (51.1744, 6.3855),
    "Darmstadt":            (49.8588, 8.6666),
    "Eintracht Frankfurt":  (50.0686, 8.6454),
    "Freiburg":             (47.9894, 7.8963),
    "Heidenheim":           (48.6752, 10.1524),
    "Hoffenheim":           (49.2375, 8.8878),
    "Koln":                 (50.9333, 6.8751),
    "Mainz":                (49.9842, 8.2241),
    "RB Leipzig":           (51.3457, 12.3479),
    "Stuttgart":            (48.7925, 9.2321),
    "Union Berlin":         (52.4575, 13.5683),
    "Werder Bremen":        (53.0665, 8.8375),
    "Wolfsburg":            (52.4320, 10.8034),
    # Serie A
    "AC Milan":             (45.4781, 9.1240),
    "Atalanta":             (45.7088, 9.6830),
    "Bologna":              (44.4929, 11.3093),
    "Cagliari":             (39.2048, 9.1338),
    "Empoli":               (43.7228, 10.9509),
    "Fiorentina":           (43.7803, 11.2823),
    "Frosinone":            (41.6477, 13.3431),
    "Genoa":                (44.4161, 8.9512),
    "Inter Milan":          (45.4781, 9.1240),
    "Juventus":             (45.1096, 7.6413),
    "Lazio":                (41.9341, 12.4547),
    "Lecce":                (40.3576, 18.0059),
    "Monza":                (45.5867, 9.2838),
    "Napoli":               (40.8279, 14.1931),
    "Roma":                 (41.9341, 12.4547),
    "Salernitana":          (40.6824, 14.7681),
    "Sassuolo":             (44.5472, 10.8517),
    "Torino":               (45.0408, 7.6501),
    "Udinese":              (46.0806, 13.2015),
    "Venezia":              (45.4573, 12.3260),
    "Verona":               (45.4341, 10.9765),
    # Ligue 1
    "Brest":                (48.4073, -4.4860),
    "Clermont":             (45.7856, 3.1419),
    "Lens":                 (50.4327, 2.8146),
    "Lille":                (50.6124, 3.1305),
    "Lorient":              (47.7488, -3.3720),
    "Lyon":                 (45.7655, 4.9822),
    "Marseille":            (43.2697, 5.3961),
    "Metz":                 (49.1101, 6.2182),
    "Monaco":               (43.7273, 7.4147),
    "Montpellier":          (43.6219, 3.8130),
    "Nantes":               (47.2557, -1.5264),
    "Nice":                 (43.7059, 7.1926),
    "Paris SG":             (48.8414, 2.2530),
    "Rennes":               (48.1079, -1.7119),
    "Reims":                (49.2466, 3.9959),
    "Strasbourg":           (48.5599, 7.7558),
    "Toulouse":             (43.5836, 1.4342),
    # Aliases fréquents dans football-data.co.uk
    "Man City":             (53.4831, -2.2004),
    "Man United":           (53.4631, -2.2913),
    "Nottm Forest":         (52.9399, -1.1328),
    "Sheffield Utd":        (53.3703, -1.4701),
    "Nott'm Forest":        (52.9399, -1.1328),
    "QPR":                  (51.5093, -0.2317),
    "Huddersfield":         (53.6543, -1.7681),
    "Swansea":              (51.5841, -3.9331),
    "Norwich":              (52.6224, 1.3095),
    "Cardiff":              (51.4728, -3.2030),
    "Middlesbrough":        (54.5780, -1.2160),
    "Sunderland":           (54.9143, -1.3888),
    "Stoke":                (52.9885, -2.1742),
    "Reading":              (51.4549, -0.9862),
    "Derby":                (52.9152, -1.4471),
    "Blackburn":            (53.7284, -2.4892),
    "Wigan":                (53.5557, -2.6657),
    "Watford":              (51.6498, -0.4017),
    "West Brom":            (52.5092, -1.9638),
    "Hull":                 (53.7461, -0.3671),
    "Ipswich":              (52.0552, 1.1447),
    "Preston":              (53.7730, -2.6913),
    "Sunderland":           (54.9143, -1.3888),
    "Paderborn":            (51.7268, 8.7500),
    "Schalke":              (51.5540, 7.0676),
    "Hertha":               (52.5147, 13.2394),
    "Greuther Furth":       (49.4870, 10.9820),
    "Espanyol":             (41.3481, 2.0790),
    "Eibar":                (43.1832, -2.4752),
    "Levante":              (39.4880, -0.3490),
    "Elche":                (38.2626, -0.7009),
}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en km entre deux points GPS (formule haversine)."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _load_stadiums() -> dict[str, tuple[float, float]]:
    """Charge le JSON stadiums.json si présent, fusionne avec les données intégrées."""
    stadiums = dict(_BUILTIN_STADIUMS)
    if os.path.exists(STADIUMS_PATH):
        try:
            with open(STADIUMS_PATH, encoding="utf-8") as f:
                extra = json.load(f)
            stadiums.update(extra)
        except Exception:
            pass
    return stadiums


_stadiums_cache: dict | None = None


def get_stadiums() -> dict[str, tuple[float, float]]:
    global _stadiums_cache
    if _stadiums_cache is None:
        _stadiums_cache = _load_stadiums()
    return _stadiums_cache


def _find_coords(team_name: str, stadiums: dict) -> tuple[float, float] | None:
    """Recherche exacte puis partielle (ignore la casse)."""
    if team_name in stadiums:
        return stadiums[team_name]
    tl = team_name.lower()
    for k, v in stadiums.items():
        if k.lower() == tl:
            return v
    # partial match : premier mot significatif (≥4 chars)
    words = [w for w in tl.split() if len(w) >= 4]
    for w in words:
        for k, v in stadiums.items():
            if w in k.lower():
                return v
    return None


def get_travel_distance(home_name: str, away_name: str) -> dict:
    """
    Retourne la distance (km) et le décalage horaire estimé entre les stades.

    {
      "travel_km":       float — distance haversine home→away stade
      "timezone_diff":   float — différence estimée en heures (|Δlongitude/15|)
      "travel_score":    float — 0-1, normalisé sur MAX_KM=2500
    }
    Retourne des zéros si les coordonnées sont introuvables.
    """
    stadiums = get_stadiums()
    h_coords = _find_coords(home_name, stadiums)
    a_coords = _find_coords(away_name, stadiums)

    if h_coords is None or a_coords is None:
        missing = []
        if h_coords is None:
            missing.append(home_name)
        if a_coords is None:
            missing.append(away_name)
        logger.debug("Coordonnées stade introuvables pour : %s", missing)
        return {"travel_km": 0.0, "timezone_diff": 0.0, "travel_score": 0.0}

    km = _haversine(*h_coords, *a_coords)
    tz_diff = abs(h_coords[1] - a_coords[1]) / 15.0  # longitude → heures UTC approx

    return {
        "travel_km":     round(km, 1),
        "timezone_diff": round(tz_diff, 2),
        "travel_score":  round(min(km / _MAX_KM, 1.0), 4),
    }


def build_travel_features(home_name: str, away_name: str) -> dict:
    """
    Features de déplacement pour le vecteur ML.
    Seul l'away subit le voyage (le home joue à domicile).

    {
      "away_travel_km":     float
      "away_timezone_diff": float
      "away_travel_score":  float
    }
    """
    t = get_travel_distance(home_name, away_name)
    return {
        "away_travel_km":     t["travel_km"],
        "away_timezone_diff": t["timezone_diff"],
        "away_travel_score":  t["travel_score"],
    }
