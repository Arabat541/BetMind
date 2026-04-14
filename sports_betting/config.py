# ============================================================
# config.py — Configuration centrale
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── APIs ────────────────────────────────────────────────────
FOOTBALL_DATA_KEY  = os.getenv("FOOTBALL_DATA_KEY", "")     # football-data.org
THE_ODDS_API_KEY   = os.getenv("THE_ODDS_API_KEY", "")      # the-odds-api.com
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
BALLDONTLIE_KEY    = os.getenv("BALLDONTLIE_KEY", "")
OPENWEATHER_KEY    = os.getenv("OPENWEATHER_KEY", "")       # openweathermap.org (gratuit)

# ── Endpoints ───────────────────────────────────────────────
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
BALLDONTLIE_BASE   = "https://api.balldontlie.io/v1"
ODDS_API_BASE      = "https://api.the-odds-api.com/v4"

# ── Sports couverts (codes football-data.org) ────────────────
FOOTBALL_LEAGUES = {
    "Ligue 1":          "FL1",
    "Premier League":   "PL",
    "La Liga":          "PD",
    "Serie A":          "SA",
    "Bundesliga":       "BL1",
    "Champions League": "CL",
}

NBA_SEASON = "2024-25"  # saison NBA courante

# ── Modèle ──────────────────────────────────────────────────
FORM_WINDOW        = 5       # N derniers matchs pour la forme (court terme)
FORM_WINDOW_LONG   = 10      # N derniers matchs pour la forme (long terme)
MIN_MATCHES_MODEL  = 5       # matchs minimum pour prédire
CONFIDENCE_THRESHOLD = 0.60  # seuil confiance minimum pour alerter

# ── Quota football-data.org ─────────────────────────────────
FD_DAILY_LIMIT     = 90      # max requêtes/jour (free plan = 100, marge de sécurité)

# ── Value Bet ───────────────────────────────────────────────
# Seuils différenciés : les paris extérieur/nul sont plus risqués
VALUE_BET_EDGE       = 0.05   # edge domicile (H) — gardé pour compatibilité
VALUE_BET_EDGE_HOME  = 0.05   # domicile : prédictibilité correcte
VALUE_BET_EDGE_AWAY  = 0.07   # extérieur : variance plus haute → seuil +2%
VALUE_BET_EDGE_DRAW  = 0.10   # nul : outcome le plus aléatoire → seuil +5%

# ── Value Bet sanity checks ──────────────────────────────────
MIN_ODD_ALLOWED    = 1.50   # cote min : en dessous le gain couvre à peine le vig
MAX_ODD_ALLOWED    = 5.00   # cote max : au-delà la variance ruine le bankroll
MAX_EDGE_SANITY    = 0.30   # edge max plausible : >30% = probablement erreur de modèle
MIN_EV_REQUIRED    = 0.02   # EV minimum requis (EV = p_model*odd - 1)
ALERT_MIN_EDGE     = 0.07   # edge minimum pour envoyer une alerte Telegram (7%)

# ── Alertes bankroll ─────────────────────────────────────────
LOW_BANKROLL_THRESHOLD = 50_000  # FCFA — alerte si bankroll en dessous

# ── Kelly Criterion ─────────────────────────────────────────
KELLY_FRACTION      = 0.25   # Kelly fractionnaire (prudent)
KELLY_FRACTION_DRAW = 0.125  # Kelly réduit pour les nuls (×0.5 — outcome très incertain)
MAX_BET_PCT         = 0.05   # max 5% de la bankroll par pari
MAX_DAILY_STAKE_PCT = 0.15   # max 15% de la bankroll exposée par jour (tous paris cumulés)
DAILY_STOP_LOSS_PCT      = 0.05   # stop-loss : si P&L récent < -5% bankroll, aucune nouvelle mise
ODDS_MOVEMENT_THRESHOLD  = 0.10   # annule la mise si la cote a baissé de > 10% depuis la prédiction
MAX_ACTIVE_BETS_PER_LEAGUE = 2    # max paris 1X2 actifs par ligue (diversification)
MAX_TEAM_EXPOSURE_PCT      = 0.10 # max 10% bankroll misé sur la victoire d'une même équipe
INITIAL_BANKROLL   = 100_000  # FCFA

# ── Fichiers locaux ─────────────────────────────────────────
DATA_DIR           = "data"
MODELS_DIR         = "models"
LOGS_DIR           = "logs"
DB_PATH            = "data/betting.db"

# ── Dashboard ───────────────────────────────────────────────
FLASK_PORT         = 5001
FLASK_DEBUG        = False
FLASK_HOST         = "0.0.0.0"

# ── Coordonnées GPS des stades (lat, lon) — J ────────────────
# Source : Wikipedia / Google Maps (approximatif, précision ~1 km suffisante)
STADIUM_COORDS: dict = {
    # Premier League
    "Arsenal":              (51.5549,  -0.1084),
    "Chelsea":              (51.4816,  -0.1910),
    "Liverpool":            (53.4308,  -2.9608),
    "Manchester City":      (53.4831,  -2.2004),
    "Manchester United":    (53.4631,  -2.2913),
    "Tottenham":            (51.6043,  -0.0665),
    "Newcastle":            (54.9756,  -1.6218),
    "Aston Villa":          (52.5090,  -1.8847),
    "West Ham":             (51.5386,  -0.0164),
    "Everton":              (53.4388,  -2.9663),
    "Brighton":             (50.8618,  -0.0836),
    "Brentford":            (51.4913,  -0.3088),
    "Fulham":               (51.4749,  -0.2217),
    "Wolves":               (52.5900,  -2.1302),
    "Crystal Palace":       (51.3983,  -0.0855),
    "Nottingham Forest":    (52.9399,  -1.1328),
    "Leicester":            (52.6204,  -1.1424),
    "Ipswich":              (52.0544,   1.1450),
    "Southampton":          (50.9058,  -1.3914),
    "Bournemouth":          (50.7352,  -1.8381),
    # Ligue 1
    "Paris Saint-Germain":  (48.8414,   2.2530),
    "Marseille":            (43.2700,   5.3954),
    "Lyon":                 (45.7650,   4.9822),
    "Monaco":               (43.7272,   7.4151),
    "Lille":                (50.6113,   3.1302),
    "Nice":                 (43.7050,   7.1926),
    "Lens":                 (50.4344,   2.8150),
    "Rennes":               (48.1075,  -1.7130),
    "Strasbourg":           (48.5600,   7.7521),
    "Nantes":               (47.2558,  -1.5257),
    "Montpellier":          (43.6226,   3.8136),
    "Reims":                (49.2486,   4.0354),
    "Toulouse":             (43.5831,   1.4346),
    "Brest":                (48.4073,  -4.4785),
    "Le Havre":             (49.4967,   0.1348),
    "Metz":                 (49.1086,   6.2178),
    # La Liga
    "Real Madrid":          (40.4531,  -3.6883),
    "Barcelona":            (41.3809,   2.1228),
    "Atletico Madrid":      (40.4361,  -3.5994),
    "Sevilla":              (37.3841,  -5.9705),
    "Real Sociedad":        (43.3014,  -1.9733),
    "Villarreal":           (39.9440,  -0.1030),
    "Athletic Bilbao":      (43.2642,  -2.9494),
    "Real Betis":           (37.3561,  -5.9814),
    "Valencia":             (39.4744,  -0.3582),
    "Getafe":               (40.3259,  -3.7143),
    # Serie A
    "Juventus":             (45.1096,   7.6412),
    "Inter Milan":          (45.4781,   9.1240),
    "AC Milan":             (45.4781,   9.1240),
    "Napoli":               (40.8279,  14.1933),
    "AS Roma":              (41.8337,  12.4875),
    "Lazio":                (41.9339,  12.4544),
    "Atalanta":             (45.7089,   9.6785),
    "Fiorentina":           (43.7808,  11.2820),
    "Bologna":              (44.4926,  11.3136),
    "Torino":               (45.0409,   7.6501),
    # Bundesliga
    "Bayern Munich":        (48.2188,  11.6248),
    "Borussia Dortmund":    (51.4926,   7.4519),
    "Bayer Leverkusen":     (51.0383,   6.9927),
    "RB Leipzig":           (51.3457,  12.3484),
    "Eintracht Frankfurt":  (50.0687,   8.6450),
    "VfB Stuttgart":        (48.7926,   9.2322),
    "SC Freiburg":          (47.9873,   7.8961),
    "Borussia Mönchengladbach": (51.1747, 6.3854),
    "Wolfsburg":            (52.4317,  10.8034),
    "Union Berlin":         (52.4573,  13.5680),
}