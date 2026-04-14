# ============================================================
# train_from_csv.py — Entraînement modèle football 1X2
# Source     : football-data.co.uk (CSV gratuit, aucune clé API)
# Features   : forme, buts, xG, H2H, Poisson, cotes bookmaker,
#              fatigue, motivation/enjeu
# Validation : TimeSeriesSplit (5 folds — pas de fuite temporelle)
# Usage      : cd sports_betting && python train_from_csv.py
# ============================================================

import bisect
import io
import json
import logging
import os
import pickle
import sys
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
except ImportError:
    logger.error("XGBoost requis : pip install xgboost")
    sys.exit(1)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict

try:
    from lightgbm import LGBMClassifier
    LGBM_OK = True
except ImportError:
    LGBM_OK = False

try:
    import shap
    import matplotlib
    matplotlib.use("Agg")   # headless — pas de display requis
    import matplotlib.pyplot as plt
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from config import MODELS_DIR, DATA_DIR
from ensemble_model import EnsembleModel

# ── Configuration ────────────────────────────────────────────
LEAGUES = {
    "Premier League": "E0",
    "La Liga":        "SP1",
    "Serie A":        "I1",
    "Bundesliga":     "D1",
    "Ligue 1":        "F1",
}
LEAGUE_N_TEAMS = {"E0": 20, "SP1": 20, "I1": 20, "D1": 18, "F1": 18}
SEASONS        = ["2425", "2324", "2223", "2122", "2021"]
BASE_URL       = "https://www.football-data.co.uk/mmz4281"
AVG_GOALS      = 2.5
MODEL_PATH        = os.path.join(MODELS_DIR, "football_xgb_model.pkl")
ENSEMBLE_PATH     = os.path.join(MODELS_DIR, "football_ensemble_model.pkl")
BEST_PARAMS_PATH  = os.path.join(MODELS_DIR, "best_params_football.json")
TEAM_SHOTS_PATH   = os.path.join(DATA_DIR,   "team_shots_current.json")

# ── Colonnes features ────────────────────────────────────────
FEATURE_COLS = [
    # Forme récente / longue
    "home_form", "away_form", "form_diff",
    "home_form_long", "away_form_long", "form_diff_long",
    # Buts
    "home_goals_for_avg", "home_goals_ag_avg",
    "away_goals_for_avg", "away_goals_ag_avg",
    # Force attaque / défense
    "home_attack_str", "home_defense_str",
    "away_attack_str", "away_defense_str",
    # Win rate
    "home_win_rate", "away_win_rate", "win_rate_diff",
    # Efficacité
    "home_clean_sheet_rate", "away_clean_sheet_rate",
    "home_scoring_rate", "away_scoring_rate",
    # Classement
    "home_pts_per_game", "away_pts_per_game", "pts_per_game_diff",
    # Forme H/A spécifique
    "home_form_home", "away_form_away", "home_away_form_diff",
    # Avantage domicile
    "home_advantage",
    # H2H
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
    "h2h_avg_goals", "h2h_matches",
    # Poisson lambda
    "home_lambda", "away_lambda", "total_goals_exp",
    # Cotes bookmaker (signal le plus fort — Bet365 ou moyenne)
    "impl_home", "impl_draw", "impl_away",
    "odd_home", "odd_draw", "odd_away", "has_odds",
    # Shots (proxy xG) — A
    "home_sot_avg", "away_sot_avg",
    "home_sot_ag_avg", "away_sot_ag_avg",
    "home_shots_avg", "away_shots_avg",
    "home_sot_ratio", "away_sot_ratio",
    # Fatigue / calendrier — E
    "home_days_since_last", "away_days_since_last",
    "home_fatigue",         "away_fatigue",
    # Motivation / enjeu — G
    "home_relegation_gap", "away_relegation_gap",
    "home_title_gap",      "away_title_gap",
]


# ════════════════════════════════════════════════════════════
# TÉLÉCHARGEMENT
# ════════════════════════════════════════════════════════════

def download_csv(league_code: str, season: str) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), encoding="latin-1", on_bad_lines="skip")
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        df["FTHG"]   = df["FTHG"].astype(int)
        df["FTAG"]   = df["FTAG"].astype(int)
        df["season"] = season
        logger.info(f"  {league_code}/{season}: {len(df)} matchs")
        return df
    except Exception as e:
        logger.warning(f"  {league_code}/{season}: {e}")
        return None


# ════════════════════════════════════════════════════════════
# PRÉCOMPUTATION (O(n) — évite le O(n²) naïf)
# ════════════════════════════════════════════════════════════

def _precompute_histories(all_df: pd.DataFrame) -> tuple:
    """
    Construit pour chaque équipe une liste triée de
    (date, is_home, gf, ga, sot, shots, sot_ag, shots_ag).
    Retourne (history_dict, dates_dict, has_shots).
    """
    has_shots = (
        "HST" in all_df.columns and
        all_df["HST"].notna().sum() > 100
    )
    history: dict = {}

    for row in all_df.itertuples(index=False):
        h, a = row.HomeTeam, row.AwayTeam
        hg, ag = int(row.FTHG), int(row.FTAG)
        d = row.Date

        if has_shots:
            h_sot  = float(getattr(row, "HST", 0) or 0)
            h_sh   = float(getattr(row, "HS",  0) or 0)
            a_sot  = float(getattr(row, "AST", 0) or 0)
            a_sh   = float(getattr(row, "AS",  0) or 0)
        else:
            h_sot = h_sh = a_sot = a_sh = 0.0

        # (date, is_home, gf, ga, sot_for, shots_for, sot_ag, shots_ag)
        for team, is_home, gf, ga, sot, shots, sot_ag, shots_ag in [
            (h, True,  hg, ag, h_sot, h_sh, a_sot, a_sh),
            (a, False, ag, hg, a_sot, a_sh, h_sot, h_sh),
        ]:
            if team not in history:
                history[team] = []
            history[team].append((d, is_home, gf, ga, sot, shots, sot_ag, shots_ag))

    for team in history:
        history[team].sort(key=lambda x: x[0])

    dates = {team: [x[0] for x in hist] for team, hist in history.items()}
    return history, dates, has_shots


def _precompute_h2h(all_df: pd.DataFrame) -> dict:
    """h2h_db[frozenset({h, a})] = [(date, home_name, hg, ag), ...] trié."""
    h2h: dict = {}
    for row in all_df.itertuples(index=False):
        key = frozenset({row.HomeTeam, row.AwayTeam})
        if key not in h2h:
            h2h[key] = []
        h2h[key].append((row.Date, row.HomeTeam, int(row.FTHG), int(row.FTAG)))
    for key in h2h:
        h2h[key].sort(key=lambda x: x[0])
    return h2h


def _build_standings_timeline(df_ls: pd.DataFrame, n_teams: int) -> dict:
    """
    Retourne {date: {team: rank}} calculé AVANT chaque journée.
    Traite les multi-matchs d'un même jour en batch.
    """
    pts: dict = {}
    timeline: dict = {}

    for date, grp in df_ls.sort_values("Date").groupby("Date"):
        # Rang avant ce groupe de matchs
        if pts:
            sorted_teams = sorted(pts.keys(), key=lambda t: -pts[t])
            ranks = {t: i + 1 for i, t in enumerate(sorted_teams)}
        else:
            ranks = {}

        day_ranks: dict = {}
        for row in grp.itertuples(index=False):
            for t in (row.HomeTeam, row.AwayTeam):
                day_ranks[t] = ranks.get(t, n_teams)
        timeline[date] = day_ranks

        # Mettre à jour les points
        for row in grp.itertuples(index=False):
            h, a, hg, ag = row.HomeTeam, row.AwayTeam, int(row.FTHG), int(row.FTAG)
            for t in (h, a):
                if t not in pts:
                    pts[t] = 0
            if hg > ag:
                pts[h] += 3
            elif hg == ag:
                pts[h] += 1
                pts[a] += 1
            else:
                pts[a] += 3

    return timeline


# ════════════════════════════════════════════════════════════
# CALCUL DES FEATURES PAR MATCH
# ════════════════════════════════════════════════════════════

def _form_score(chars: list, window: int = 5) -> float:
    recent = chars[-window:] if len(chars) >= window else chars
    if not recent:
        return 0.5
    return round(
        sum(3 if c == "W" else (1 if c == "D" else 0) for c in recent)
        / (len(recent) * 3), 4
    )


def team_rolling(history: list, dates_list: list,
                 before_date, n: int = 15) -> dict | None:
    """Stats roulantes d'une équipe sur ses N derniers matchs avant before_date."""
    idx  = bisect.bisect_left(dates_list, before_date)
    past = history[max(0, idx - n):idx]

    if len(past) < 5:
        return None

    goals_for, goals_ag = [], []
    sot_for, shots_for, sot_ag_l, shots_ag_l = [], [], [], []
    form_all, form_home, form_away = [], [], []
    match_dates = []
    pts = 0

    for d, is_home, gf, ga, sot, shots, sot_ag, shots_ag in past:
        goals_for.append(gf)
        goals_ag.append(ga)
        match_dates.append(d)
        if sot > 0 or shots > 0:
            sot_for.append(sot)
            shots_for.append(shots)
            sot_ag_l.append(sot_ag)
            shots_ag_l.append(shots_ag)
        res = "W" if gf > ga else ("D" if gf == ga else "L")
        form_all.append(res)
        (form_home if is_home else form_away).append(res)
        pts += 3 if res == "W" else (1 if res == "D" else 0)

    n_played = len(past)
    last_date  = max(match_dates)
    days_since = max(0, (before_date - last_date).days)
    fatigue    = sum(1 for d in match_dates if (before_date - d).days <= 10)

    sot_avg    = round(sum(sot_for)    / len(sot_for),    4) if sot_for    else 0.0
    shots_avg  = round(sum(shots_for)  / len(shots_for),  4) if shots_for  else 0.0
    sot_ag_avg = round(sum(sot_ag_l)   / len(sot_ag_l),   4) if sot_ag_l   else 0.0
    sot_ratio  = round(sot_avg / max(shots_avg, 1.0), 4)

    return {
        "goals_for_avg":    round(sum(goals_for) / n_played, 4),
        "goals_ag_avg":     round(sum(goals_ag)  / n_played, 4),
        "win_rate":         round(sum(1 for c in form_all if c == "W") / n_played, 4),
        "clean_sheet_rate": round(sum(1 for g in goals_ag  if g == 0)  / n_played, 4),
        "scoring_rate":     round(sum(1 for g in goals_for if g > 0)   / n_played, 4),
        "form":             _form_score(form_all, 5),
        "form_long":        _form_score(form_all, 10),
        "form_home":        _form_score(form_home, 5),
        "form_away":        _form_score(form_away, 5),
        "pts_per_game":     round(pts / n_played, 4),
        "sot_avg":          sot_avg,
        "shots_avg":        shots_avg,
        "sot_ag_avg":       sot_ag_avg,
        "sot_ratio":        sot_ratio,
        "days_since_last":  days_since,
        "fatigue":          fatigue,
    }


def h2h_features(h2h_db: dict, home: str, away: str,
                 before_date, last_n: int = 10) -> dict:
    default = {
        "h2h_home_win_rate": 0.33, "h2h_draw_rate": 0.33,
        "h2h_away_win_rate": 0.33, "h2h_avg_goals": 2.5, "h2h_matches": 0,
    }
    key  = frozenset({home, away})
    hist = h2h_db.get(key, [])
    if not hist:
        return default

    dates = [x[0] for x in hist]
    idx   = bisect.bisect_left(dates, before_date)
    past  = hist[max(0, idx - last_n):idx]
    if not past:
        return default

    n = len(past)
    home_wins = draws = total_goals = 0
    for d, h_name, hg, ag in past:
        total_goals += hg + ag
        is_our_home = h_name == home
        if is_our_home:
            if hg > ag: home_wins += 1
            elif hg == ag: draws += 1
        else:
            if ag > hg: home_wins += 1
            elif hg == ag: draws += 1

    return {
        "h2h_home_win_rate": round(home_wins / n, 4),
        "h2h_draw_rate":     round(draws / n, 4),
        "h2h_away_win_rate": round((n - home_wins - draws) / n, 4),
        "h2h_avg_goals":     round(total_goals / n, 4),
        "h2h_matches":       n,
    }


def _get_rank(timeline: dict, team: str, date, n_teams: int) -> int:
    """Retourne le rang de l'équipe à la date donnée (avec fallback)."""
    day = timeline.get(date)
    if day and team in day:
        return day[team]
    past_dates = [d for d in timeline if d < date]
    if past_dates:
        return timeline[max(past_dates)].get(team, n_teams // 2)
    return n_teams // 2


def _extract_odds(row: pd.Series) -> tuple:
    """Tente d'extraire les cotes depuis plusieurs colonnes bookmaker."""
    for h_col, d_col, a_col in [
        ("B365H", "B365D", "B365A"),
        ("BbAvH", "BbAvD", "BbAvA"),
        ("PSH",   "PSD",   "PSA"),
        ("WHH",   "WHD",   "WHA"),
    ]:
        try:
            oh = float(row.get(h_col, 0) or 0)
            od = float(row.get(d_col, 0) or 0)
            oa = float(row.get(a_col, 0) or 0)
            if oh > 1.0 and od > 1.0 and oa > 1.0:
                return oh, od, oa
        except (TypeError, ValueError):
            continue
    return None, None, None


def build_row_features(row: pd.Series,
                       history: dict, dates_dict: dict,
                       h2h_db: dict,
                       standings_cache: dict,
                       n_teams: int) -> dict | None:
    home   = row["HomeTeam"]
    away   = row["AwayTeam"]
    date   = row["Date"]
    league = row["league"]
    season = row["season"]

    # Rolling stats
    hs  = team_rolling(history.get(home, []), dates_dict.get(home, []), date)
    ass = team_rolling(history.get(away, []), dates_dict.get(away, []), date)
    if hs is None or ass is None:
        return None

    # H2H
    h2h = h2h_features(h2h_db, home, away, date)

    # Attack / defense / Poisson
    avg          = AVG_GOALS / 2
    ha           = hs["goals_for_avg"]  / max(avg, 0.1)
    hd           = hs["goals_ag_avg"]   / max(avg, 0.1)
    aa           = ass["goals_for_avg"] / max(avg, 0.1)
    ad           = ass["goals_ag_avg"]  / max(avg, 0.1)
    home_lambda  = max(ha * ad * avg, 0.1)
    away_lambda  = max(aa * hd * avg, 0.1)

    # Bookmaker odds
    odd_h, odd_d, odd_a = _extract_odds(row)
    has_odds = 1.0 if odd_h else 0.0
    if odd_h:
        tot     = 1/odd_h + 1/odd_d + 1/odd_a
        impl_h  = round((1/odd_h) / tot, 4)
        impl_d  = round((1/odd_d) / tot, 4)
        impl_a  = round((1/odd_a) / tot, 4)
    else:
        impl_h, impl_d, impl_a = 0.33, 0.33, 0.33
        odd_h, odd_d, odd_a    = 2.5, 3.3, 2.5

    # Motivation / classement
    cache_key  = f"{league}_{season}"
    timeline   = standings_cache.get(cache_key, {})
    home_rank  = _get_rank(timeline, home, date, n_teams)
    away_rank  = _get_rank(timeline, away, date, n_teams)
    rel_thresh = n_teams - 3   # seuil zone de relégation

    return {
        # Forme
        "home_form":             hs["form"],
        "away_form":             ass["form"],
        "form_diff":             round(hs["form"]      - ass["form"],      4),
        "home_form_long":        hs["form_long"],
        "away_form_long":        ass["form_long"],
        "form_diff_long":        round(hs["form_long"] - ass["form_long"], 4),
        # Buts
        "home_goals_for_avg":    hs["goals_for_avg"],
        "home_goals_ag_avg":     hs["goals_ag_avg"],
        "away_goals_for_avg":    ass["goals_for_avg"],
        "away_goals_ag_avg":     ass["goals_ag_avg"],
        # Force
        "home_attack_str":       round(ha, 4),
        "home_defense_str":      round(hd, 4),
        "away_attack_str":       round(aa, 4),
        "away_defense_str":      round(ad, 4),
        # Win rate
        "home_win_rate":         hs["win_rate"],
        "away_win_rate":         ass["win_rate"],
        "win_rate_diff":         round(hs["win_rate"] - ass["win_rate"], 4),
        # Efficacité
        "home_clean_sheet_rate": hs["clean_sheet_rate"],
        "away_clean_sheet_rate": ass["clean_sheet_rate"],
        "home_scoring_rate":     hs["scoring_rate"],
        "away_scoring_rate":     ass["scoring_rate"],
        # Classement
        "home_pts_per_game":     hs["pts_per_game"],
        "away_pts_per_game":     ass["pts_per_game"],
        "pts_per_game_diff":     round(hs["pts_per_game"] - ass["pts_per_game"], 4),
        # Forme H/A
        "home_form_home":        hs["form_home"],
        "away_form_away":        ass["form_away"],
        "home_away_form_diff":   round(hs["form_home"] - ass["form_away"], 4),
        # Avantage
        "home_advantage":        1.0,
        # H2H
        **h2h,
        # Poisson
        "home_lambda":           round(home_lambda, 4),
        "away_lambda":           round(away_lambda, 4),
        "total_goals_exp":       round(home_lambda + away_lambda, 4),
        # Cotes
        "impl_home":             impl_h,
        "impl_draw":             impl_d,
        "impl_away":             impl_a,
        "odd_home":              round(odd_h, 3),
        "odd_draw":              round(odd_d, 3),
        "odd_away":              round(odd_a, 3),
        "has_odds":              has_odds,
        # Shots (proxy xG)
        "home_sot_avg":          hs["sot_avg"],
        "away_sot_avg":          ass["sot_avg"],
        "home_sot_ag_avg":       hs["sot_ag_avg"],
        "away_sot_ag_avg":       ass["sot_ag_avg"],
        "home_shots_avg":        hs["shots_avg"],
        "away_shots_avg":        ass["shots_avg"],
        "home_sot_ratio":        hs["sot_ratio"],
        "away_sot_ratio":        ass["sot_ratio"],
        # Fatigue
        "home_days_since_last":  hs["days_since_last"],
        "away_days_since_last":  ass["days_since_last"],
        "home_fatigue":          hs["fatigue"],
        "away_fatigue":          ass["fatigue"],
        # Motivation (positif = safe / en tête, négatif = en danger / loin)
        "home_relegation_gap":   round((rel_thresh - home_rank) / n_teams, 4),
        "away_relegation_gap":   round((rel_thresh - away_rank) / n_teams, 4),
        "home_title_gap":        round((home_rank - 1) / n_teams, 4),
        "away_title_gap":        round((away_rank - 1) / n_teams, 4),
    }


# ════════════════════════════════════════════════════════════
# SAUVEGARDE SHOTS LOOKUP (pour prédictions live)
# ════════════════════════════════════════════════════════════

def save_team_shots_lookup(all_df: pd.DataFrame):
    """
    Calcule les moyennes de tirs (HST/HS) de la saison la plus récente.
    Sauvegardé dans data/team_shots_current.json pour les prédictions live.
    """
    if "HST" not in all_df.columns or all_df["HST"].notna().sum() < 50:
        logger.info("Données shots non disponibles — lookup non généré.")
        return

    latest_season = SEASONS[0]
    lookup: dict  = {}

    for league_name in LEAGUES:
        df_s = all_df[
            (all_df["league"] == league_name) &
            (all_df["season"] == latest_season)
        ].dropna(subset=["HST", "HS", "AST", "AS"]).copy()
        if df_s.empty:
            continue

        team_shots: dict = {}
        all_teams = set(df_s["HomeTeam"]) | set(df_s["AwayTeam"])

        for team in all_teams:
            home_rows = df_s[df_s["HomeTeam"] == team]
            away_rows = df_s[df_s["AwayTeam"] == team]
            sot_for, shots_for, sot_ag = [], [], []

            for _, m in home_rows.iterrows():
                sot_for.append(float(m["HST"]))
                shots_for.append(float(m["HS"]))
                sot_ag.append(float(m["AST"]))
            for _, m in away_rows.iterrows():
                sot_for.append(float(m["AST"]))
                shots_for.append(float(m["AS"]))
                sot_ag.append(float(m["HST"]))

            if len(sot_for) >= 3:
                sh_avg = sum(shots_for) / len(shots_for)
                st_avg = sum(sot_for)   / len(sot_for)
                team_shots[team] = {
                    "sot_avg":    round(st_avg, 4),
                    "shots_avg":  round(sh_avg, 4),
                    "sot_ag_avg": round(sum(sot_ag) / len(sot_ag), 4),
                    "sot_ratio":  round(st_avg / max(sh_avg, 1.0), 4),
                    "n":          len(sot_for),
                }

        if team_shots:
            lookup[league_name] = team_shots

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TEAM_SHOTS_PATH, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)
    n_teams = sum(len(v) for v in lookup.values())
    logger.info(f"Shots lookup sauvegardé : {TEAM_SHOTS_PATH} ({n_teams} équipes)")


# ════════════════════════════════════════════════════════════
# OPTUNA — HYPERPARAMETER TUNING (D)
# ════════════════════════════════════════════════════════════

def _optuna_objective(trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eval_metric":      "mlogloss",
        "random_state":     42,
        "n_jobs":           -1,
        "use_label_encoder": False,
    }
    tss    = TimeSeriesSplit(n_splits=3)
    losses = []
    for tr_idx, val_idx in tss.split(X):
        m = XGBClassifier(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        losses.append(log_loss(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])))
    return float(np.mean(losses))


def optimize_hyperparams(X: pd.DataFrame, y: pd.Series, n_trials: int = 40) -> dict:
    """Optuna search — minimise log_loss sur TimeSeriesSplit(3). Sauvegarde dans best_params_football.json."""
    if not OPTUNA_OK:
        logger.warning("Optuna non installé — hyperparamètres par défaut.")
        return {}

    logger.info(f"Optuna hyperparameter tuning ({n_trials} trials)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda t: _optuna_objective(t, X, y),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )

    best = dict(study.best_params)
    best.update({"eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1})

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump({"params": best, "best_log_loss": round(study.best_value, 6)}, f, indent=2)

    logger.info(f"Best log_loss : {study.best_value:.4f}")
    logger.info(f"Best params   : {best}")
    return best


# ════════════════════════════════════════════════════════════
# SHAP — FEATURE IMPORTANCE (I)
# ════════════════════════════════════════════════════════════

def compute_shap(X: pd.DataFrame, y: pd.Series, xgb_params: dict):
    """
    Entraîne un XGBClassifier brut (sans calibration) pour SHAP.
    Logue les 10 features les plus importantes et sauvegarde un graphique.
    """
    if not SHAP_OK:
        logger.warning("shap/matplotlib non installés — analyse SHAP ignorée.")
        return

    logger.info("Calcul SHAP feature importance...")
    xgb_shap = XGBClassifier(**xgb_params)
    xgb_shap.fit(X, y)

    # Échantillon pour accélérer (max 2000 exemples)
    sample = X.sample(min(2000, len(X)), random_state=42)

    try:
        explainer  = shap.TreeExplainer(xgb_shap)
        shap_vals  = explainer.shap_values(sample)   # (n_samples, n_features, n_classes)
    except Exception as e:
        logger.warning(f"SHAP TreeExplainer failed: {e}")
        return

    # Importance moyenne absolue sur toutes les classes
    sv = np.array(shap_vals)
    if sv.ndim == 3:                        # (n_samples, n_features, n_classes)
        mean_abs = np.abs(sv).mean(axis=(0, 2))
    elif sv.ndim == 4:                      # list of (n_samples, n_features) → (n_cls, n, p)
        mean_abs = np.abs(sv).mean(axis=(0, 1))
    else:
        mean_abs = np.abs(sv).mean(axis=0)
    mean_abs = np.array(mean_abs, dtype=float).flatten()

    importance = sorted(zip(FEATURE_COLS, mean_abs), key=lambda x: x[1], reverse=True)

    logger.info("SHAP feature importance (top 10):")
    for feat, imp in importance[:10]:
        bar = "█" * int(imp * 80 / max(importance[0][1], 1e-9))
        logger.info(f"  {feat:30s} {imp:.4f}  {bar}")

    # Graphique barre
    shap_path = os.path.join(MODELS_DIR, "shap_football.png")
    try:
        top_n     = 15
        top_feats = [f for f, _ in importance[:top_n]]
        top_imps  = [v for _, v in importance[:top_n]]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(top_n), top_imps[::-1], color="#2196F3")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_feats[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Football model — Top 15 features (SHAP)")
        plt.tight_layout()
        fig.savefig(shap_path, dpi=100)
        plt.close(fig)
        logger.info(f"SHAP graphique sauvegardé : {shap_path}")
    except Exception as e:
        logger.warning(f"SHAP plot failed: {e}")


# ════════════════════════════════════════════════════════════
# ENSEMBLE XGB + LGB + STACKING (C)
# ════════════════════════════════════════════════════════════

def train_ensemble(X: pd.DataFrame, y: pd.Series, xgb_params: dict) -> EnsembleModel | None:
    """
    Entraîne XGB + LGB comme base-learners via OOF TimeSeriesSplit(5),
    puis un méta-modèle LogisticRegression sur les probas empilées.
    Gain typique : +2 à +4 pts d'accuracy vs XGB seul.
    """
    if not LGBM_OK:
        logger.warning("LightGBM non installé — ensemble non entraîné.")
        return None

    logger.info("Entraînement ensemble XGB + LGB + LR (stacking)...")
    n        = len(X)
    tss      = TimeSeriesSplit(n_splits=5)
    n_cls    = 3
    meta_X   = np.zeros((n, n_cls * 2))   # [XGB×3 + LGB×3]

    lgb_params = {
        "n_estimators":     xgb_params.get("n_estimators", 300),
        "max_depth":        xgb_params.get("max_depth", 5),
        "learning_rate":    xgb_params.get("learning_rate", 0.05),
        "subsample":        xgb_params.get("subsample", 0.8),
        "colsample_bytree": xgb_params.get("colsample_bytree", 0.8),
        "min_child_samples": 10,
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,
    }

    fold_accs = []
    for fold, (tr_idx, val_idx) in enumerate(tss.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        xgb_f = XGBClassifier(**xgb_params)
        lgb_f = LGBMClassifier(**lgb_params)
        xgb_f.fit(X_tr, y_tr)
        lgb_f.fit(X_tr, y_tr)

        meta_X[val_idx, :n_cls]    = xgb_f.predict_proba(X_val)
        meta_X[val_idx, n_cls:]    = lgb_f.predict_proba(X_val)

        p_avg = (xgb_f.predict_proba(X_val) + lgb_f.predict_proba(X_val)) / 2
        fold_accs.append(accuracy_score(y_val, p_avg.argmax(axis=1)))
        logger.info(f"  Fold {fold + 1}: avg-accuracy={fold_accs[-1]:.3f}")

    logger.info(f"Ensemble OOF accuracy (avg): {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")

    # Méta-modèle sur les OOF probas
    meta_model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    meta_model.fit(meta_X, y)

    # Base-learners finaux sur tout le dataset
    logger.info("Entraînement des base-learners finaux (toutes données)...")
    xgb_final = XGBClassifier(**xgb_params)
    lgb_final = LGBMClassifier(**lgb_params)
    xgb_final.fit(X, y)
    lgb_final.fit(X, y)

    ensemble = EnsembleModel(
        xgb_model=xgb_final,
        lgb_model=lgb_final,
        meta_model=meta_model,
        n_classes=n_cls,
    )
    ensemble.save(ENSEMBLE_PATH)
    logger.info(f"Ensemble sauvegardé : {ENSEMBLE_PATH}")
    return ensemble


# ════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    # 1. Télécharger les CSV
    frames = []
    for league_name, code in LEAGUES.items():
        for season in SEASONS:
            df = download_csv(code, season)
            if df is not None:
                df["league"]      = league_name
                df["league_code"] = code
                frames.append(df)

    if not frames:
        logger.error("Aucune donnée téléchargée.")
        sys.exit(1)

    all_df = (
        pd.concat(frames, ignore_index=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )
    logger.info(
        f"Total : {len(all_df)} matchs "
        f"({all_df['Date'].min().date()} → {all_df['Date'].max().date()})"
    )

    has_shots = "HST" in all_df.columns and all_df["HST"].notna().sum() > 100
    logger.info(f"Shots disponibles dans les CSV : {'oui' if has_shots else 'non (features shots = 0)'}")

    # 2. Précomputation O(n)
    logger.info("Précomputation des historiques équipes...")
    history, dates_dict, _ = _precompute_histories(all_df)
    h2h_db                 = _precompute_h2h(all_df)

    logger.info("Précomputation des classements temporels...")
    standings_cache: dict = {}
    for league_name, code in LEAGUES.items():
        n_teams = LEAGUE_N_TEAMS[code]
        for season in SEASONS:
            df_ls = all_df[
                (all_df["league"] == league_name) &
                (all_df["season"] == season)
            ]
            if df_ls.empty:
                continue
            key = f"{league_name}_{season}"
            standings_cache[key] = _build_standings_timeline(df_ls, n_teams)

    # 3. Construction du dataset
    logger.info("Construction des features...")
    LABEL = {"H": 0, "D": 1, "A": 2}
    rows_X, rows_y = [], []

    for i, row in all_df.iterrows():
        label = LABEL.get(str(row.get("FTR", "")).strip())
        if label is None:
            continue

        n_teams = LEAGUE_N_TEAMS.get(row.get("league_code", ""), 20)
        feats   = build_row_features(
            row, history, dates_dict, h2h_db, standings_cache, n_teams
        )
        if feats is None:
            continue

        rows_X.append({c: feats.get(c, 0.0) for c in FEATURE_COLS})
        rows_y.append(label)

        if (i + 1) % 2000 == 0:
            logger.info(f"  {i + 1}/{len(all_df)} matchs traités…")

    X = pd.DataFrame(rows_X, columns=FEATURE_COLS)
    y = pd.Series(rows_y)
    logger.info(
        f"Dataset : {len(X)} matchs | "
        f"H={int((y == 0).sum())} D={int((y == 1).sum())} A={int((y == 2).sum())}"
    )

    if len(X) < 500:
        logger.error("Pas assez de données (< 500). Vérifier les téléchargements.")
        sys.exit(1)

    # 4. Optuna hyperparameter tuning — D
    best_params = optimize_hyperparams(X, y, n_trials=40)
    xgb_params  = best_params if best_params else {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1,
    }

    # 5. Walk-forward validation — B
    logger.info("Walk-forward validation (TimeSeriesSplit n_splits=5)...")
    tss         = TimeSeriesSplit(n_splits=5)
    fold_accs   = []
    fold_lls    = []

    for fold, (tr_idx, val_idx) in enumerate(tss.split(X)):
        X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
        y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]

        base = XGBClassifier(**xgb_params)
        m    = CalibratedClassifierCV(base, cv=3, method="isotonic")
        m.fit(X_tr, y_tr)

        acc = accuracy_score(y_val, m.predict(X_val))
        ll  = log_loss(y_val, m.predict_proba(X_val))
        fold_accs.append(acc)
        fold_lls.append(ll)
        logger.info(
            f"  Fold {fold + 1}: Accuracy={acc:.3f} | LogLoss={ll:.4f} "
            f"| val={len(val_idx)} matchs"
        )

    logger.info(
        f"Walk-forward — Accuracy: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f} "
        f"| LogLoss: {np.mean(fold_lls):.4f}"
    )

    # 6. Modèle final sur tout le dataset
    logger.info("Entraînement final (toutes données)...")
    base_final  = XGBClassifier(**xgb_params)
    final_model = CalibratedClassifierCV(base_final, cv=5, method="isotonic")
    final_model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    logger.info(f"Modèle sauvegardé : {MODEL_PATH}")

    # 7. Ensemble XGB + LGB + stacking — C
    train_ensemble(X, y, xgb_params)

    # 8. SHAP feature importance — I
    compute_shap(X, y, xgb_params)

    # 9. Shots lookup pour prédictions live
    save_team_shots_lookup(all_df)

    final_acc = np.mean(fold_accs)
    logger.info(f"Accuracy walk-forward : {final_acc:.1%}")
    return final_acc


if __name__ == "__main__":
    train()
