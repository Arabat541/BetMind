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