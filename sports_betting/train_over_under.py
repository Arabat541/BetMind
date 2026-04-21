# ============================================================
# train_over_under.py — Entraînement modèle Over/Under 2.5
# Source : football-data.co.uk (mêmes CSV que train_from_csv.py)
# Cible binaire : FTHG + FTAG > 2 → 1 (Over 2.5), sinon 0 (Under)
# ============================================================

import io
import os
import sys
import logging
import pickle
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
except ImportError:
    logger.error("XGBoost requis.")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from config import MODELS_DIR
from feature_engineering import get_feature_columns
from understat_fetcher import load_xg_history, get_team_xg_rolling

FEATURE_COLS = get_feature_columns("ou_football")
import numpy as np
MODEL_PATH   = os.path.join(MODELS_DIR, "ou_football_xgb_model.pkl")
BASE_URL     = "https://www.football-data.co.uk/mmz4281"
AVG_GOALS    = 2.5

LEAGUES = {
    "Ligue 1":        "F1",
    "Premier League": "E0",
    "La Liga":        "SP1",
    "Serie A":        "I1",
    "Bundesliga":     "D1",
}
SEASONS = ["2425", "2324", "2223", "2122"]


# ════════════════════════════════════════════════════════════
# TÉLÉCHARGEMENT
# ════════════════════════════════════════════════════════════

def download_csv(league_code: str, season: str) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), encoding="latin-1", on_bad_lines="skip")
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)
        df["season"] = season
        logger.info(f"  {league_code}/{season}: {len(df)} matchs")
        return df
    except Exception as e:
        logger.warning(f"  Erreur {league_code}/{season}: {e}")
        return None


# ════════════════════════════════════════════════════════════
# CONSTRUCTION DES FEATURES
# ════════════════════════════════════════════════════════════

def team_stats_before(all_df: pd.DataFrame, team: str, date, n: int = 15) -> dict:
    mask = (
        ((all_df["HomeTeam"] == team) | (all_df["AwayTeam"] == team)) &
        (all_df["Date"] < date)
    )
    past = all_df[mask].sort_values("Date").tail(n)
    if past.empty:
        return {}

    goals_for, goals_ag, home_form, away_form = [], [], [], []
    sot_for, sot_ag_l = [], []
    for _, m in past.iterrows():
        is_home = m["HomeTeam"] == team
        gf = int(m["FTHG"]) if is_home else int(m["FTAG"])
        ga = int(m["FTAG"]) if is_home else int(m["FTHG"])
        goals_for.append(gf)
        goals_ag.append(ga)
        res = "W" if gf > ga else ("D" if gf == ga else "L")
        if is_home:
            home_form.append(res)
        else:
            away_form.append(res)
        # Shots on target
        if "HST" in m.index and pd.notna(m.get("HST")):
            sot_for.append(float(m["HST"]) if is_home else float(m.get("AST", 0) or 0))
            sot_ag_l.append(float(m.get("AST", 0) or 0) if is_home else float(m["HST"]))

    n_total = len(goals_for)

    def form_score(chars, window=5):
        recent = chars[-window:] if len(chars) >= window else chars
        if not recent:
            return 0.5
        return sum(3 if c == "W" else (1 if c == "D" else 0) for c in recent) / (len(recent) * 3)

    gf_avg = sum(goals_for) / n_total
    ga_avg = sum(goals_ag)  / n_total
    failed = sum(1 for g in goals_for if g == 0)

    # Fatigue
    today = pd.Timestamp.now()
    dates_seen = list(past["Date"])
    last_date  = max(dates_seen) if dates_seen else today - pd.Timedelta(days=7)
    days_since = max(0, (today - last_date).days)
    fatigue    = sum(1 for d in dates_seen if (today - d).days <= 10)

    return {
        "goals_for_avg":  round(gf_avg, 4),
        "goals_ag_avg":   round(ga_avg, 4),
        "scoring_rate":   round(1 - failed / n_total, 4),
        "home_form":      round(form_score(home_form), 4),
        "away_form":      round(form_score(away_form), 4),
        "sot_avg":        round(sum(sot_for) / len(sot_for), 4) if sot_for else 0.0,
        "sot_ag_avg":     round(sum(sot_ag_l) / len(sot_ag_l), 4) if sot_ag_l else 0.0,
        "days_since":     days_since,
        "fatigue":        fatigue,
    }


def h2h_avg_goals(all_df: pd.DataFrame, home: str, away: str, date, last_n: int = 10) -> float:
    mask = (
        (((all_df["HomeTeam"] == home) & (all_df["AwayTeam"] == away)) |
         ((all_df["HomeTeam"] == away) & (all_df["AwayTeam"] == home))) &
        (all_df["Date"] < date)
    )
    past = all_df[mask].sort_values("Date").tail(last_n)
    if past.empty:
        return 2.5
    return round((past["FTHG"] + past["FTAG"]).mean(), 4)


def _xg_features(home: str, away: str, date, xg_history: dict | None) -> dict:
    default = {"home_xg_avg": 0.0, "away_xg_avg": 0.0,
               "home_xga_avg": 0.0, "away_xga_avg": 0.0,
               "xg_diff": 0.0, "xga_diff": 0.0}
    if not xg_history:
        return default
    h = get_team_xg_rolling(xg_history, home, before_date=date, window=8)
    a = get_team_xg_rolling(xg_history, away, before_date=date, window=8)
    return {
        "home_xg_avg":  h["xg_avg"],  "away_xg_avg":  a["xg_avg"],
        "home_xga_avg": h["xga_avg"], "away_xga_avg": a["xga_avg"],
        "xg_diff":  round(h["xg_avg"]  - a["xg_avg"],  4),
        "xga_diff": round(h["xga_avg"] - a["xga_avg"], 4),
    }


def build_features(row: pd.Series, all_df: pd.DataFrame,
                   xg_history: dict | None = None) -> dict | None:
    home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
    hs  = team_stats_before(all_df, home, date)
    as_ = team_stats_before(all_df, away, date)
    if not hs or not as_:
        return None

    avg          = AVG_GOALS / 2
    home_attack  = hs["goals_for_avg"] / max(avg, 0.1)
    home_defense = hs["goals_ag_avg"]  / max(avg, 0.1)
    away_attack  = as_["goals_for_avg"] / max(avg, 0.1)
    away_defense = as_["goals_ag_avg"]  / max(avg, 0.1)
    home_lambda  = max(home_attack * away_defense * avg, 0.1)
    away_lambda  = max(away_attack * home_defense * avg, 0.1)

    return {
        "home_goals_for_avg": hs["goals_for_avg"],
        "home_goals_ag_avg":  hs["goals_ag_avg"],
        "away_goals_for_avg": as_["goals_for_avg"],
        "away_goals_ag_avg":  as_["goals_ag_avg"],
        "home_attack_str":    round(home_attack, 4),
        "home_defense_str":   round(home_defense, 4),
        "away_attack_str":    round(away_attack, 4),
        "away_defense_str":   round(away_defense, 4),
        "home_scoring_rate":  hs["scoring_rate"],
        "away_scoring_rate":  as_["scoring_rate"],
        "home_lambda":        round(home_lambda, 4),
        "away_lambda":        round(away_lambda, 4),
        "total_goals_exp":    round(home_lambda + away_lambda, 4),
        "h2h_avg_goals":      h2h_avg_goals(all_df, home, away, date),
        # Shots (proxy xG) — A
        "home_sot_avg":       hs.get("sot_avg", 0.0),
        "away_sot_avg":       as_.get("sot_avg", 0.0),
        "home_sot_ag_avg":    hs.get("sot_ag_avg", 0.0),
        "away_sot_ag_avg":    as_.get("sot_ag_avg", 0.0),
        # Fatigue — E
        "home_days_since_last": hs.get("days_since", 7),
        "away_days_since_last": as_.get("days_since", 7),
        "home_fatigue":         hs.get("fatigue", 1),
        "away_fatigue":         as_.get("fatigue", 1),
        # xG réel Understat — W
        **_xg_features(home, away, date, xg_history),
    }


# ════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Télécharger toutes les données
    frames = []
    for league_name, code in LEAGUES.items():
        for season in SEASONS:
            df = download_csv(code, season)
            if df is not None:
                df["league"] = league_name
                frames.append(df)
    if not frames:
        logger.error("Aucune donnée téléchargée.")
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    logger.info(f"Total matchs chargés : {len(all_df)}")

    xg_history = load_xg_history()
    if xg_history:
        logger.info(f"xG Understat chargé : {len(xg_history)} équipes")

    # Construire le dataset
    rows_X, rows_y = [], []
    for i, row in all_df.iterrows():
        total = int(row["FTHG"]) + int(row["FTAG"])
        label = 1 if total > 2 else 0   # Over 2.5 = 3 buts ou plus

        feats = build_features(row, all_df, xg_history=xg_history)
        if feats is None:
            continue

        rows_X.append({c: feats.get(c, 0.0) for c in FEATURE_COLS})
        rows_y.append(label)

        if (i + 1) % 1000 == 0:
            logger.info(f"  {i + 1}/{len(all_df)} matchs traités…")

    X = pd.DataFrame(rows_X, columns=FEATURE_COLS)
    y = pd.Series(rows_y)
    logger.info(f"Dataset : {len(X)} matchs | Over={y.sum()} ({y.mean():.1%}) | Under={len(y)-y.sum()}")

    if len(X) < 200:
        logger.error("Pas assez de données (< 200).")
        sys.exit(1)

    # Walk-forward validation — B
    logger.info("Walk-forward validation (TimeSeriesSplit n_splits=5)...")
    tss       = TimeSeriesSplit(n_splits=5)
    fold_accs = []
    fold_lls  = []

    for fold, (tr_idx, val_idx) in enumerate(tss.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        base = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        m = CalibratedClassifierCV(base, cv=3, method="isotonic")
        m.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, m.predict(X_val))
        ll  = log_loss(y_val, m.predict_proba(X_val))
        fold_accs.append(acc)
        fold_lls.append(ll)
        logger.info(f"  Fold {fold + 1}: Accuracy={acc:.3f} | LogLoss={ll:.4f}")

    logger.info(
        f"Walk-forward OU — Accuracy: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f} "
        f"| LogLoss: {np.mean(fold_lls):.4f}"
    )

    # Modèle final sur toutes les données
    base_final = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    model = CalibratedClassifierCV(base_final, cv=5, method="isotonic")
    model.fit(X, y)

    acc = float(np.mean(fold_accs))
    ll  = float(np.mean(fold_lls))
    logger.info(f"Over/Under model — Accuracy (WF): {acc:.1%} | LogLoss: {ll:.4f} | {len(FEATURE_COLS)} features")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Modèle sauvegardé : {MODEL_PATH}")
    return acc


if __name__ == "__main__":
    train()
