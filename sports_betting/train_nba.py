"""
train_nba.py v2 — Entraîne XGBoost NBA via nba_api (nba.com/stats)
Gratuit, sans authentification, données complètes multi-saisons.

Installation : pip install nba_api
Usage        : cd sports_betting && python train_nba.py

Labels : 0=home win, 1=away win
  (requis par model.py _predict_xgb : proba[0]=prob_home, proba[1]=prob_away)
"""

import os
import sys
import time
import pickle
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from nba_api.stats.endpoints import leaguegamefinder
    NBA_API_OK = True
except ImportError:
    NBA_API_OK = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False

from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, classification_report

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

MODELS_DIR       = "models"
DATA_DIR         = "data"
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "best_params_nba.json")

# 5 saisons × ~1230 matchs = ~6150 matchs bruts
# Après exclusion MIN_GAMES_BEFORE : ~5400 exemples
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

MIN_GAMES_BEFORE = 10   # matchs minimum par équipe avant d'inclure un exemple
SLEEP_SECONDS    = 2.5  # pause entre appels nba.com (anti-rate-limit)

# Colonnes features — doit correspondre exactement à get_feature_columns("nba")
# dans feature_engineering.py
FEATURE_COLS = [
    "home_win_rate",
    "away_win_rate",
    "win_rate_diff",
    "home_pts_for_avg",
    "home_pts_ag_avg",
    "away_pts_for_avg",
    "away_pts_ag_avg",
    "home_pts_diff",
    "away_pts_diff",
    "pts_diff_gap",
    "home_off_eff",
    "away_off_eff",
    "home_advantage",
]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ── Récupération des données ──────────────────────────────────────────────────

def fetch_season(season: str) -> pd.DataFrame | None:
    """
    Récupère tous les matchs Regular Season d'une saison via nba.com/stats.
    Utilise un cache CSV pour éviter les re-téléchargements.
    """
    cache_path = os.path.join(DATA_DIR, f"nba_games_{season.replace('-', '')}.csv")

    if os.path.exists(cache_path):
        logger.info(f"Cache: {cache_path}")
        try:
            return pd.read_csv(cache_path)
        except Exception as e:
            logger.warning(f"Cache illisible: {e}")
            os.remove(cache_path)

    logger.info(f"Fetching nba.com/stats — saison {season} ...")
    try:
        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Regular Season",
            league_id_nullable="00",
            timeout=60,
        )
        df = gf.get_data_frames()[0]
        df.to_csv(cache_path, index=False)
        logger.info(f"  → {len(df)} lignes (2 par match) sauvegardées")
        time.sleep(SLEEP_SECONDS)
        return df
    except Exception as e:
        logger.error(f"Erreur saison {season}: {e}")
        return None


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Transforme le format nba_api (1 ligne par équipe par match)
    en 1 ligne par match avec home_team, away_team, scores.

    MATCHUP "BOS vs. MIA" → BOS est à domicile
    MATCHUP "BOS @ MIA"   → BOS est en déplacement (MIA à domicile)
    """
    games: dict = {}

    for _, row in df.iterrows():
        gid     = str(row.get("GAME_ID", ""))
        matchup = str(row.get("MATCHUP", ""))
        pts     = float(row.get("PTS", 0) or 0)
        wl      = str(row.get("WL", "")).upper()
        date    = str(row.get("GAME_DATE", ""))
        team    = str(row.get("TEAM_ABBREVIATION", ""))

        if not gid or not matchup:
            continue

        is_home = "vs." in matchup

        if gid not in games:
            games[gid] = {"date": date, "home": None, "away": None}

        entry = {"team": team, "pts": pts, "win": wl == "W"}

        if is_home:
            games[gid]["home"] = entry
        else:
            games[gid]["away"] = entry

    rows = []
    for gid, g in games.items():
        if g["home"] is None or g["away"] is None:
            continue
        # Ignorer les matchs sans score (abandon, report)
        if g["home"]["pts"] == 0 and g["away"]["pts"] == 0:
            continue

        rows.append({
            "season":     season,
            "date":       g["date"],
            "home_team":  g["home"]["team"],
            "away_team":  g["away"]["team"],
            "home_score": g["home"]["pts"],
            "away_score": g["away"]["pts"],
            # 0=home win, 1=away win (ordre attendu par model.py _predict_xgb)
            "label":      0 if g["home"]["win"] else 1,
        })

    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    logger.info(f"  Saison {season}: {len(result)} matchs parsés")
    return result


# ── Features rolling ──────────────────────────────────────────────────────────

def build_rolling_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque match (dans l'ordre chronologique), calcule les stats
    rolling des deux équipes à partir de tous leurs matchs précédents.
    L'historique est réinitialisé entre les saisons (appel séparé par saison).
    """
    team_history: dict = defaultdict(lambda: {
        "wins": 0, "games": 0, "pts_for": [], "pts_ag": []
    })

    rows = []
    for _, row in games_df.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        hh = team_history[h]
        ah = team_history[a]

        # Inclure seulement si les deux équipes ont assez de matchs
        if hh["games"] >= MIN_GAMES_BEFORE and ah["games"] >= MIN_GAMES_BEFORE:
            feat = _compute_features(hh, ah)
            feat["label"] = int(row["label"])
            rows.append(feat)

        # Mettre à jour l'historique APRÈS les features (pas de data leakage)
        _update_history(team_history, h, row["home_score"], row["away_score"])
        _update_history(team_history, a, row["away_score"], row["home_score"])

    return pd.DataFrame(rows)


def _compute_features(h: dict, a: dict) -> dict:
    """Calcule les 13 features NBA à partir des historiques des deux équipes."""
    def avg(lst: list) -> float:
        return float(np.mean(lst)) if lst else 100.0

    h_wr = h["wins"] / h["games"]
    a_wr = a["wins"] / a["games"]
    h_pf = avg(h["pts_for"])
    h_pa = avg(h["pts_ag"])
    a_pf = avg(a["pts_for"])
    a_pa = avg(a["pts_ag"])
    h_pd = h_pf - h_pa
    a_pd = a_pf - a_pa

    return {
        "home_win_rate":    round(h_wr, 4),
        "away_win_rate":    round(a_wr, 4),
        "win_rate_diff":    round(h_wr - a_wr, 4),
        "home_pts_for_avg": round(h_pf, 2),
        "home_pts_ag_avg":  round(h_pa, 2),
        "away_pts_for_avg": round(a_pf, 2),
        "away_pts_ag_avg":  round(a_pa, 2),
        "home_pts_diff":    round(h_pd, 2),
        "away_pts_diff":    round(a_pd, 2),
        "pts_diff_gap":     round(h_pd - a_pd, 2),
        "home_off_eff":     round(h_pf / max(a_pa, 1.0), 4),
        "away_off_eff":     round(a_pf / max(h_pa, 1.0), 4),
        "home_advantage":   1.0,
    }


def _update_history(history: dict, team: str, scored: float, allowed: float):
    """Met à jour l'historique d'une équipe après un match."""
    h = history[team]
    h["games"] += 1
    h["pts_for"].append(scored)
    h["pts_ag"].append(allowed)
    if scored > allowed:
        h["wins"] += 1


# ── Dataset complet ───────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    """Télécharge toutes les saisons et construit le dataset d'entraînement."""
    all_features = []

    for season in SEASONS:
        raw = fetch_season(season)
        if raw is None:
            logger.warning(f"Saison {season} ignorée.")
            continue

        games = parse_season(raw, season)
        if games.empty:
            logger.warning(f"Saison {season}: aucun match parsé.")
            continue

        # Historique reset par saison (roster changes)
        feats = build_rolling_features(games)
        logger.info(
            f"  Saison {season}: {len(feats)} exemples de training "
            f"({len(games)} matchs, {MIN_GAMES_BEFORE} exclus/équipe au départ)"
        )
        all_features.append(feats)

    if not all_features:
        return pd.DataFrame()

    dataset = pd.concat(all_features, ignore_index=True)
    logger.info(f"\nDataset total : {len(dataset)} exemples")
    return dataset


# ── Optuna — D ────────────────────────────────────────────────────────────────

def _nba_objective(trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eval_metric":      "logloss",
        "random_state":     42,
        "n_jobs":           -1,
    }
    tss = TimeSeriesSplit(n_splits=3)
    losses = []
    for tr_idx, val_idx in tss.split(X):
        m = XGBClassifier(**params) if XGB_AVAILABLE else GradientBoostingClassifier(
            n_estimators=100, max_depth=params["max_depth"],
            learning_rate=params["learning_rate"], random_state=42,
        )
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        losses.append(log_loss(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])))
    return float(np.mean(losses))


def optimize_hyperparams_nba(X: pd.DataFrame, y: pd.Series, n_trials: int = 40) -> dict:
    """Optuna search NBA — minimise log_loss sur TimeSeriesSplit(3)."""
    if not OPTUNA_OK or not XGB_AVAILABLE:
        logger.warning("Optuna/XGB non disponible — hyperparamètres par défaut.")
        return {}

    logger.info(f"Optuna hyperparameter tuning NBA ({n_trials} trials)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda t: _nba_objective(t, X, y),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )

    best = dict(study.best_params)
    best.update({"eval_metric": "logloss", "random_state": 42, "n_jobs": -1})

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump({"params": best, "best_log_loss": round(study.best_value, 6)}, f, indent=2)

    logger.info(f"Best log_loss : {study.best_value:.4f}")
    logger.info(f"Best params   : {best}")
    return best


# ── Entraînement ──────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> float:
    """Entraîne XGBoost et sauvegarde le modèle + colonnes features."""
    X = df[FEATURE_COLS]
    y = df["label"]

    baseline = (y == 0).mean()  # taux victoire domicile
    logger.info(f"Baseline (always home wins): {baseline:.1%}")
    logger.info(
        f"Distribution labels: {dict(y.value_counts().sort_index())} "
        f"(0=home win, 1=away win)"
    )

    # Optuna tuning — D
    best_params = optimize_hyperparams_nba(X, y, n_trials=40)
    xgb_params  = best_params if best_params else {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
    }

    # Walk-forward validation — B
    logger.info("Walk-forward validation (TimeSeriesSplit n_splits=5)...")
    tss       = TimeSeriesSplit(n_splits=5)
    fold_accs = []
    fold_lls  = []

    for fold, (tr_idx, val_idx) in enumerate(tss.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        if XGB_AVAILABLE:
            m = XGBClassifier(**xgb_params)
        else:
            m = GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=42,
            )
        m.fit(X_tr, y_tr)
        acc_f = accuracy_score(y_val, m.predict(X_val))
        ll_f  = log_loss(y_val, m.predict_proba(X_val))
        fold_accs.append(acc_f)
        fold_lls.append(ll_f)
        logger.info(f"  Fold {fold + 1}: Accuracy={acc_f:.3f} | LogLoss={ll_f:.4f} | val={len(val_idx)}")

    acc  = float(np.mean(fold_accs))
    loss = float(np.mean(fold_lls))
    logger.info(f"\n{'─'*50}")
    logger.info(f"Walk-forward NBA — Accuracy: {acc:.3f} ± {np.std(fold_accs):.3f}  (baseline: {baseline:.3f})")
    logger.info(f"LogLoss moyen    : {loss:.4f}")
    logger.info(f"Gain vs baseline : {acc - baseline:+.3f}")

    # Modèle final sur toutes les données
    if XGB_AVAILABLE:
        model = XGBClassifier(**xgb_params)
        logger.info("Entraînement final XGBoost (toutes données)...")
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42,
        )
        logger.info("Entraînement final GradientBoosting (toutes données)...")

    model.fit(X, y)

    # Rapport classification sur dernier fold (approximation)
    last_tr, last_val = list(tss.split(X))[-1]
    y_pred_last = model.predict(X.iloc[last_val])
    logger.info(f"\n{classification_report(X.iloc[last_val].index[:0].tolist() or y.iloc[last_val], y_pred_last, target_names=['home_win', 'away_win'], zero_division=0)}")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        imps = sorted(
            zip(FEATURE_COLS, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        logger.info("Feature importances:")
        for feat, imp in imps:
            bar = "█" * int(imp * 60)
            logger.info(f"  {feat:25s} {imp:.4f}  {bar}")

    # Sauvegarde
    model_path = os.path.join(MODELS_DIR, "nba_xgb_model.pkl")
    cols_path  = os.path.join(MODELS_DIR, "nba_feature_cols.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(cols_path, "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    logger.info(f"\nModèle sauvegardé    : {model_path}")
    logger.info(f"Features sauvegardées: {cols_path}")

    return acc


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not NBA_API_OK:
        logger.error("nba_api non installé. Lancez: pip install nba_api")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  NBA XGBoost Training — nba.com/stats (via nba_api)")
    logger.info(f"  Saisons : {', '.join(SEASONS)}")
    logger.info(f"  MIN_GAMES_BEFORE : {MIN_GAMES_BEFORE}")
    logger.info("=" * 60)

    dataset = build_dataset()

    if dataset.empty:
        logger.error("Dataset vide. Vérifiez la connectivité vers nba.com.")
        sys.exit(1)

    if len(dataset) < 500:
        logger.warning(
            f"Seulement {len(dataset)} exemples — résultats peu fiables."
        )

    acc = train(dataset)

    if acc > 0.62:
        logger.info(f"\n✓ Bon modèle ({acc:.1%}) — supérieur à la baseline.")
    elif acc > 0.58:
        logger.info(f"\n✓ Modèle acceptable ({acc:.1%}) — léger gain vs baseline.")
    else:
        logger.warning(
            f"\n⚠ Modèle faible ({acc:.1%}) — proche de la baseline. "
            "Supprimez nba_xgb_model.pkl pour revenir au prior NBA (~75%)."
        )
