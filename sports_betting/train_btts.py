# ============================================================
# train_btts.py — Entraînement modèle BTTS (Both Teams To Score)
# Source : football-data.co.uk (mêmes CSV que train_over_under.py)
# Cible binaire : FTHG > 0 ET FTAG > 0 → 1 (BTTS), sinon 0
# Edge : bookmakers calibrent moins ce marché que le 1X2
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
from understat_fetcher import load_xg_history, get_team_xg_rolling

MODEL_PATH = os.path.join(MODELS_DIR, "btts_football_xgb_model.pkl")
BASE_URL   = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    "Ligue 1":        "F1",
    "Premier League": "E0",
    "La Liga":        "SP1",
    "Serie A":        "I1",
    "Bundesliga":     "D1",
}
SEASONS = ["2425", "2324", "2223", "2122"]

FEATURE_COLS = [
    "home_goals_for_avg", "home_goals_ag_avg",
    "away_goals_for_avg", "away_goals_ag_avg",
    "home_attack_str", "home_defense_str",
    "away_attack_str", "away_defense_str",
    "home_scoring_rate", "away_scoring_rate",
    "home_clean_sheet_rate", "away_clean_sheet_rate",
    "h2h_avg_goals", "btts_h2h_rate",
    "home_sot_avg", "away_sot_avg",
    "home_sot_ag_avg", "away_sot_ag_avg",
    "home_days_since_last", "away_days_since_last",
    "home_fatigue", "away_fatigue",
    # xG réel Understat — W
    "home_xg_avg", "away_xg_avg",
    "home_xga_avg", "away_xga_avg",
    "xg_diff", "xga_diff",
]

AVG_GOALS = 2.5


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
# STATS PAR ÉQUIPE
# ════════════════════════════════════════════════════════════

def team_stats_before(all_df: pd.DataFrame, team: str, date, n: int = 15) -> dict:
    mask = (
        ((all_df["HomeTeam"] == team) | (all_df["AwayTeam"] == team)) &
        (all_df["Date"] < date)
    )
    past = all_df[mask].sort_values("Date").tail(n)
    if len(past) < 5:
        return {}

    goals_for, goals_ag = [], []
    sot_for, sot_ag_l   = [], []
    btts_results        = []

    for _, m in past.iterrows():
        is_home = m["HomeTeam"] == team
        gf = int(m["FTHG"]) if is_home else int(m["FTAG"])
        ga = int(m["FTAG"]) if is_home else int(m["FTHG"])
        goals_for.append(gf)
        goals_ag.append(ga)
        btts_results.append(1 if gf > 0 and ga > 0 else 0)

        if "HST" in m.index and pd.notna(m.get("HST")):
            sot_for.append(float(m["HST"]) if is_home else float(m.get("AST", 0) or 0))
            sot_ag_l.append(float(m.get("AST", 0) or 0) if is_home else float(m["HST"]))

    n_total    = len(goals_for)
    failed     = sum(1 for g in goals_for if g == 0)
    cs         = sum(1 for g in goals_ag  if g == 0)
    today      = pd.Timestamp.now()
    dates_seen = list(past["Date"])
    last_date  = max(dates_seen) if dates_seen else today - pd.Timedelta(days=7)

    return {
        "goals_for_avg":     round(sum(goals_for) / n_total, 4),
        "goals_ag_avg":      round(sum(goals_ag)  / n_total, 4),
        "scoring_rate":      round(1 - failed / n_total, 4),
        "clean_sheet_rate":  round(cs / n_total, 4),
        "btts_rate":         round(sum(btts_results) / n_total, 4),
        "sot_avg":           round(sum(sot_for)  / len(sot_for),  4) if sot_for  else 0.0,
        "sot_ag_avg":        round(sum(sot_ag_l) / len(sot_ag_l), 4) if sot_ag_l else 0.0,
        "days_since":        max(0, (today - last_date).days),
        "fatigue":           sum(1 for d in dates_seen if (today - d).days <= 10),
    }


def h2h_btts_stats(all_df: pd.DataFrame, home: str, away: str,
                   date, last_n: int = 10) -> dict:
    mask = (
        (((all_df["HomeTeam"] == home) & (all_df["AwayTeam"] == away)) |
         ((all_df["HomeTeam"] == away) & (all_df["AwayTeam"] == home))) &
        (all_df["Date"] < date)
    )
    past = all_df[mask].sort_values("Date").tail(last_n)
    if past.empty:
        return {"h2h_avg_goals": 2.5, "btts_h2h_rate": 0.5}

    total_goals = (past["FTHG"] + past["FTAG"]).sum()
    btts_count  = ((past["FTHG"] > 0) & (past["FTAG"] > 0)).sum()
    n = len(past)
    return {
        "h2h_avg_goals": round(total_goals / n, 4),
        "btts_h2h_rate": round(btts_count / n, 4),
    }


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

    h2h = h2h_btts_stats(all_df, home, away, date)

    return {
        "home_goals_for_avg":   hs["goals_for_avg"],
        "home_goals_ag_avg":    hs["goals_ag_avg"],
        "away_goals_for_avg":   as_["goals_for_avg"],
        "away_goals_ag_avg":    as_["goals_ag_avg"],
        "home_attack_str":      round(home_attack, 4),
        "home_defense_str":     round(home_defense, 4),
        "away_attack_str":      round(away_attack, 4),
        "away_defense_str":     round(away_defense, 4),
        "home_scoring_rate":    hs["scoring_rate"],
        "away_scoring_rate":    as_["scoring_rate"],
        "home_clean_sheet_rate": hs["clean_sheet_rate"],
        "away_clean_sheet_rate": as_["clean_sheet_rate"],
        "h2h_avg_goals":        h2h["h2h_avg_goals"],
        "btts_h2h_rate":        h2h["btts_h2h_rate"],
        "home_sot_avg":         hs["sot_avg"],
        "away_sot_avg":         as_["sot_avg"],
        "home_sot_ag_avg":      hs["sot_ag_avg"],
        "away_sot_ag_avg":      as_["sot_ag_avg"],
        "home_days_since_last": hs["days_since"],
        "away_days_since_last": as_["days_since"],
        "home_fatigue":         hs["fatigue"],
        "away_fatigue":         as_["fatigue"],
        # xG réel Understat — W
        **_xg_features(home, away, date, xg_history),
    }


# ════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

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

    rows_X, rows_y = [], []
    for i, row in all_df.iterrows():
        label = 1 if (int(row["FTHG"]) > 0 and int(row["FTAG"]) > 0) else 0

        feats = build_features(row, all_df, xg_history=xg_history)
        if feats is None:
            continue

        rows_X.append({c: feats.get(c, 0.0) for c in FEATURE_COLS})
        rows_y.append(label)

        if (i + 1) % 1000 == 0:
            logger.info(f"  {i + 1}/{len(all_df)} matchs traités…")

    X = pd.DataFrame(rows_X, columns=FEATURE_COLS)
    y = pd.Series(rows_y)
    btts_rate = y.mean()
    logger.info(
        f"Dataset : {len(X)} matchs | BTTS={y.sum()} ({btts_rate:.1%}) | "
        f"No-BTTS={len(y) - y.sum()} ({1 - btts_rate:.1%})"
    )

    if len(X) < 200:
        logger.error("Pas assez de données.")
        sys.exit(1)

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
        f"Walk-forward BTTS — Accuracy: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f} "
        f"| LogLoss: {np.mean(fold_lls):.4f}"
    )

    base_final = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    model = CalibratedClassifierCV(base_final, cv=5, method="isotonic")
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Modèle BTTS sauvegardé : {MODEL_PATH}")

    final_acc = float(np.mean(fold_accs))
    final_ll  = float(np.mean(fold_lls))

    # AG — MLflow / model registry
    try:
        from model_registry import log_run
        log_run("btts", {
            "accuracy":   final_acc,
            "log_loss":   final_ll,
            "n_samples":  len(X),
            "n_features": len(FEATURE_COLS),
        })
    except Exception as e:
        logger.warning(f"model_registry log failed: {e}")

    return final_acc


if __name__ == "__main__":
    train()
