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
FORM_WINDOW        = 5       # N derniers matchs pour la forme
MIN_MATCHES_MODEL  = 5       # matchs minimum pour prédire
CONFIDENCE_THRESHOLD = 0.60  # seuil confiance minimum pour alerter

# ── Value Bet ───────────────────────────────────────────────
VALUE_BET_EDGE     = 0.05    # edge minimum : P_model - P_implied >= 5%

# ── Kelly Criterion ─────────────────────────────────────────
KELLY_FRACTION     = 0.25    # Kelly fractionnaire (prudent)
MAX_BET_PCT        = 0.05    # max 5% de la bankroll par pari
INITIAL_BANKROLL   = 100_000  # FCFA

# ── Fichiers locaux ─────────────────────────────────────────
DATA_DIR           = "data"
MODELS_DIR         = "models"
LOGS_DIR           = "logs"
DB_PATH            = "data/betting.db"

# ── Dashboard ───────────────────────────────────────────────
FLASK_PORT         = 5001
FLASK_DEBUG        = False