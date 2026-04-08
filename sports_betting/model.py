# ============================================================
# model.py — XGBoost + Distribution de Poisson + Value Bets
# ============================================================

import os
import logging
import pickle
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.special import factorial

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False
    logger.warning("XGBoost non disponible, fallback sur GradientBoosting.")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from config import MODELS_DIR, VALUE_BET_EDGE, CONFIDENCE_THRESHOLD


os.makedirs(MODELS_DIR, exist_ok=True)

# Labels : 0=Home, 1=Draw, 2=Away
RESULT_LABELS = {0: "H", 1: "D", 2: "A"}
RESULT_NAMES  = {"H": "Victoire domicile", "D": "Match nul", "A": "Victoire extérieur"}


# ════════════════════════════════════════════════════════════
# MODÈLE XGBOOST — CLASSIFICATION 1X2
# ════════════════════════════════════════════════════════════

class BettingModel:
    """
    Modèle de prédiction 1X2.
    Peut être entraîné sur des données historiques
    ou utilisé en mode "prior" (probabilités calculées).
    """

    def __init__(self, sport: str = "football"):
        self.sport     = sport
        self.model     = None
        self.is_trained = False
        self.model_path = os.path.join(MODELS_DIR, f"{sport}_xgb_model.pkl")
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            self.is_trained = True
            logger.info(f"Model loaded: {self.model_path}")

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Entraîne le modèle XGBoost sur des données historiques."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if XGB_AVAILABLE:
            self.model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            )

        self.model.fit(X_train, y_train)

        # Évaluation
        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        acc  = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)
        logger.info(f"[{self.sport}] Train done — Accuracy: {acc:.3f} | LogLoss: {loss:.4f}")

        # Sauvegarde
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        self.is_trained = True

        return {"accuracy": acc, "log_loss": loss}

    def predict_proba(self, features: dict) -> dict:
        """
        Prédit les probabilités 1X2 pour un match.
        Si le modèle n'est pas entraîné, utilise les priors Poisson (foot)
        ou statistiques directes (NBA).
        """
        if self.is_trained and self.model is not None:
            return self._predict_xgb(features)
        else:
            if self.sport == "football":
                return self._predict_poisson_prior(features)
            else:
                return self._predict_nba_prior(features)

    def _predict_xgb(self, features: dict) -> dict:
        """Prédiction via XGBoost."""
        from feature_engineering import get_feature_columns
        cols = get_feature_columns(self.sport)
        X = pd.DataFrame([{c: features.get(c, 0.0) for c in cols}])
        proba = self.model.predict_proba(X)[0]

        # XGBoost renvoie [H, D, A] si classes=[0,1,2]
        return {
            "prob_home": round(float(proba[0]), 4),
            "prob_draw": round(float(proba[1]), 4) if self.sport == "football" else 0.0,
            "prob_away": round(float(proba[2 if self.sport == "football" else 1]), 4),
            "method":    "xgboost"
        }

    def _predict_poisson_prior(self, features: dict) -> dict:
        """
        Calcule P(H), P(D), P(A) via distribution de Poisson bivariée.
        Utilisée avant entraînement XGBoost ou en fallback.
        """
        lam_h = features.get("home_lambda", 1.4)
        lam_a = features.get("away_lambda", 1.1)

        max_goals = 8
        prob_home = prob_draw = prob_away = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
                if i > j:
                    prob_home += p
                elif i == j:
                    prob_draw += p
                else:
                    prob_away += p

        total = prob_home + prob_draw + prob_away
        return {
            "prob_home": round(prob_home / total, 4),
            "prob_draw": round(prob_draw / total, 4),
            "prob_away": round(prob_away / total, 4),
            "method":    "poisson"
        }

    def _predict_nba_prior(self, features: dict) -> dict:
        """Prior NBA basé sur win_rate et différentiel de points."""
        home_wr   = features.get("home_win_rate", 0.5)
        away_wr   = features.get("away_win_rate", 0.5)
        pts_gap   = features.get("pts_diff_gap", 0.0)
        home_adv  = 0.04  # avantage domicile NBA ~4%

        # Logit simple
        raw_home = home_wr - away_wr + home_adv + (pts_gap / 100)
        prob_home = 1 / (1 + np.exp(-raw_home * 5))
        prob_home = float(np.clip(prob_home, 0.05, 0.95))

        return {
            "prob_home": round(prob_home, 4),
            "prob_draw": 0.0,
            "prob_away": round(1 - prob_home, 4),
            "method":    "prior_nba"
        }


# ════════════════════════════════════════════════════════════
# VALUE BET DETECTOR
# ════════════════════════════════════════════════════════════

def detect_value_bets(proba: dict, odds_row: dict, sport: str = "football") -> list:
    """
    Compare les probabilités du modèle aux probabilités implicites des bookmakers.
    Retourne la liste des value bets détectés.

    Args:
        proba     : {"prob_home": 0.55, "prob_draw": 0.25, "prob_away": 0.20}
        odds_row  : {"odd_home": 1.8, "odd_draw": 3.5, "odd_away": 4.2,
                     "impl_home": 0.50, "impl_draw": 0.22, "impl_away": 0.28}
    Returns:
        list of dict : chaque value bet détecté
    """
    value_bets = []

    candidates = [
        ("H", proba.get("prob_home", 0), odds_row.get("impl_home", 0), odds_row.get("odd_home")),
        ("A", proba.get("prob_away", 0), odds_row.get("impl_away", 0), odds_row.get("odd_away")),
    ]
    if sport == "football":
        candidates.append(
            ("D", proba.get("prob_draw", 0), odds_row.get("impl_draw", 0), odds_row.get("odd_draw"))
        )

    for result, p_model, p_implied, odd in candidates:
        if odd is None or p_implied == 0:
            continue
        edge = p_model - p_implied
        if edge >= VALUE_BET_EDGE and p_model >= CONFIDENCE_THRESHOLD:
            ev = (p_model * (odd - 1)) - (1 - p_model)
            value_bets.append({
                "result":     result,
                "result_name": RESULT_NAMES.get(result, result),
                "p_model":    round(p_model, 4),
                "p_implied":  round(p_implied, 4),
                "edge":       round(edge, 4),
                "odd":        round(odd, 3),
                "expected_value": round(ev, 4),
                "is_value":   True,
            })

    # Trier par edge décroissant
    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets


# ════════════════════════════════════════════════════════════
# SIGNAL FINAL
# ════════════════════════════════════════════════════════════

def build_prediction_signal(
    features: dict,
    odds_row: dict,
    model: BettingModel,
    sport: str = "football",
    match_info: dict = {}
) -> dict:
    """
    Pipeline complet : features → proba → value bets → signal.
    Retourne un dict prêt pour la DB et le dashboard.
    """
    proba = model.predict_proba(features)

    # Résultat le plus probable
    if sport == "football":
        best_result = max(
            [("H", proba["prob_home"]), ("D", proba["prob_draw"]), ("A", proba["prob_away"])],
            key=lambda x: x[1]
        )
    else:
        best_result = ("H", proba["prob_home"]) if proba["prob_home"] > 0.5 else ("A", proba["prob_away"])

    pred_result, confidence = best_result

    # Détection value bets
    value_bets = detect_value_bets(proba, odds_row, sport)

    signal = {
        "sport":       sport,
        "league":      match_info.get("league", ""),
        "home_team":   match_info.get("home_team", ""),
        "away_team":   match_info.get("away_team", ""),
        "match_date":  match_info.get("date", ""),
        "pred_result": pred_result,
        "pred_name":   RESULT_NAMES.get(pred_result, pred_result),
        "prob_home":   proba["prob_home"],
        "prob_draw":   proba.get("prob_draw", 0),
        "prob_away":   proba["prob_away"],
        "confidence":  round(confidence, 4),
        "method":      proba.get("method", ""),
        "value_bets":  value_bets,
        "is_value_bet": len(value_bets) > 0,
        "top_edge":    value_bets[0]["edge"] if value_bets else 0.0,
        "top_odd":     value_bets[0]["odd"]  if value_bets else None,
        "odd_used":    value_bets[0]["odd"]  if value_bets else None,
    }

    return signal