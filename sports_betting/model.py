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
from config import (
    MODELS_DIR, CONFIDENCE_THRESHOLD,
    MIN_ODD_ALLOWED, MAX_ODD_ALLOWED, MAX_EDGE_SANITY, MIN_EV_REQUIRED,
    VALUE_BET_EDGE_HOME, VALUE_BET_EDGE_AWAY, VALUE_BET_EDGE_DRAW,
)

try:
    from ensemble_model import EnsembleModel
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False


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
        self.sport         = sport
        self.model         = None
        self.is_trained    = False
        self.feature_cols  = None
        self.model_path    = os.path.join(MODELS_DIR, f"{sport}_xgb_model.pkl")
        self.ensemble_path = os.path.join(MODELS_DIR, f"{sport}_ensemble_model.pkl")
        self._load_if_exists()

    def _load_if_exists(self):
        # Prefer ensemble over single XGB model when available
        if ENSEMBLE_AVAILABLE and os.path.exists(self.ensemble_path):
            try:
                self.model = EnsembleModel.load(self.ensemble_path)
                self.is_trained = True
                logger.info(f"Ensemble loaded: {self.ensemble_path}")
                return
            except Exception as e:
                logger.warning(f"Ensemble load failed ({e}), falling back to XGB.")

        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            self.is_trained = True
            logger.info(f"Model loaded: {self.model_path}")
            try:
                self.feature_cols = list(self.model.feature_names_in_)
                logger.info(f"Features: {len(self.feature_cols)}")
            except AttributeError:
                self.feature_cols = None

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
        # Utilise les features du modèle entraîné (évite mismatch shape)
        cols = self.feature_cols if self.feature_cols else get_feature_columns(self.sport)
        X = pd.DataFrame([{c: features.get(c, 0.0) for c in cols}])
        proba = self.model.predict_proba(X)[0]

        if self.sport == "ou_football":
            # Binaire : classes=[0=Under, 1=Over]
            return {
                "prob_over":  round(float(proba[1]), 4),
                "prob_under": round(float(proba[0]), 4),
                "prob_home":  round(float(proba[1]), 4),   # alias pour compatibilité
                "prob_draw":  0.0,
                "prob_away":  round(float(proba[0]), 4),
                "method":     "xgboost_ou"
            }

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

    _EDGE_MIN = {"H": VALUE_BET_EDGE_HOME, "A": VALUE_BET_EDGE_AWAY, "D": VALUE_BET_EDGE_DRAW}

    for result, p_model, p_implied, odd in candidates:
        if odd is None or p_implied == 0:
            continue

        # ── Sanity check 1 : cote hors plage acceptable ──────
        if not (MIN_ODD_ALLOWED <= odd <= MAX_ODD_ALLOWED):
            continue

        edge = p_model - p_implied

        # ── Sanity check 2 : edge insuffisant ou suspect ──────
        if edge < _EDGE_MIN.get(result, VALUE_BET_EDGE_HOME):
            continue
        if edge > MAX_EDGE_SANITY:
            continue  # >30% d'avantage = probablement erreur de modèle

        # ── Sanity check 3 : confiance et EV ─────────────────
        if p_model < CONFIDENCE_THRESHOLD:
            continue

        ev = (p_model * (odd - 1)) - (1 - p_model)
        if ev < MIN_EV_REQUIRED:
            continue  # EV trop faible même si edge positif

        value_bets.append({
            "result":         result,
            "result_name":    RESULT_NAMES.get(result, result),
            "p_model":        round(p_model, 4),
            "p_implied":      round(p_implied, 4),
            "edge":           round(edge, 4),
            "odd":            round(odd, 3),
            "expected_value": round(ev, 4),
            "is_value":       True,
        })

    # Trier par edge décroissant
    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets


# ════════════════════════════════════════════════════════════
# OVER/UNDER VALUE BET + SIGNAL
# ════════════════════════════════════════════════════════════

def detect_ou_value_bet(prob_over: float, ou_odds_row: dict) -> dict | None:
    """
    Détecte un value bet Over/Under 2.5.
    Seuils : edge >= 7% (même que Away — variance comparable).
    """
    if not ou_odds_row:
        return None

    OU_EDGE_MIN = 0.07

    for side, p_model, impl_key, odd_key in (
        ("O", prob_over,       "impl_over",  "odd_over"),
        ("U", 1 - prob_over,   "impl_under", "odd_under"),
    ):
        odd       = ou_odds_row.get(odd_key)
        p_implied = ou_odds_row.get(impl_key, 0)
        if odd is None or p_implied == 0:
            continue
        if not (MIN_ODD_ALLOWED <= odd <= MAX_ODD_ALLOWED):
            continue
        edge = p_model - p_implied
        if edge < OU_EDGE_MIN or edge > MAX_EDGE_SANITY:
            continue
        if p_model < CONFIDENCE_THRESHOLD:
            continue
        ev = p_model * (odd - 1) - (1 - p_model)
        if ev < MIN_EV_REQUIRED:
            continue
        return {
            "result":         side,
            "result_name":    "Over 2.5" if side == "O" else "Under 2.5",
            "p_model":        round(p_model, 4),
            "p_implied":      round(p_implied, 4),
            "edge":           round(edge, 4),
            "odd":            round(odd, 3),
            "expected_value": round(ev, 4),
            "is_value":       True,
        }
    return None


def build_ou_signal(features: dict, ou_odds_row: dict,
                    ou_model: "BettingModel", match_info: dict) -> dict | None:
    """Signal Over/Under pour un match de foot. Retourne None si pas de value."""
    if not ou_model.is_trained:
        return None

    proba     = ou_model.predict_proba(features)
    prob_over = proba.get("prob_over", proba.get("prob_home", 0))
    vb        = detect_ou_value_bet(prob_over, ou_odds_row)
    if vb is None:
        return None

    return {
        "sport":       "football",
        "league":      match_info.get("league", ""),
        "home_team":   match_info.get("home_team", ""),
        "away_team":   match_info.get("away_team", ""),
        "match_date":  match_info.get("date", ""),
        "pred_result": vb["result"],
        "pred_name":   vb["result_name"],
        "prob_home":   prob_over,
        "prob_draw":   0.0,
        "prob_away":   round(1 - prob_over, 4),
        "confidence":  round(vb["p_model"], 4),
        "method":      proba.get("method", "xgboost_ou"),
        "value_bets":  [vb],
        "is_value_bet": True,
        "edge":        vb["edge"],
        "odd_used":    vb["odd"],
        "market":      "OU25",
    }


# ════════════════════════════════════════════════════════════
# ASIAN HANDICAP -0.5 (K)
# ════════════════════════════════════════════════════════════

def detect_ah_value_bet(prob_home: float, prob_away: float,
                        prob_draw: float, ah_odds_row: dict) -> dict | None:
    """
    Asian Handicap -0.5 : élimine le nul → binaire H/A.
    P(AH home win) = P(home win) / (P(home win) + P(away win))
    Seuil edge >= 6% (marché plus efficient que 1X2 mais moins que OU).
    """
    if not ah_odds_row:
        return None

    AH_EDGE_MIN = 0.06
    total_no_draw = prob_home + prob_away
    if total_no_draw < 0.01:
        return None

    p_ah_home = prob_home / total_no_draw
    p_ah_away = prob_away / total_no_draw

    for side, p_model, odd_key in (
        ("AH_H", p_ah_home, "odd_ah_home"),
        ("AH_A", p_ah_away, "odd_ah_away"),
    ):
        odd = ah_odds_row.get(odd_key)
        if not odd or odd <= 1.0:
            continue
        p_implied = round(1 / odd, 4)
        edge      = round(p_model - p_implied, 4)
        ev        = round(p_model * odd - 1, 4)

        if edge >= AH_EDGE_MIN and ev > 0:
            label = "Asian Handicap -0.5 Domicile" if side == "AH_H" else "Asian Handicap -0.5 Extérieur"
            return {
                "result":         side,
                "result_name":    label,
                "p_model":        round(p_model, 4),
                "p_implied":      p_implied,
                "edge":           edge,
                "odd":            round(odd, 3),
                "expected_value": ev,
                "is_value":       True,
            }
    return None


# ════════════════════════════════════════════════════════════
# BTTS VALUE BET + SIGNAL (F)
# ════════════════════════════════════════════════════════════

def detect_btts_value_bet(prob_btts: float, btts_odds_row: dict) -> dict | None:
    """
    Détecte un value bet BTTS / No-BTTS.
    Seuil edge >= 7% (même politique que OU).
    """
    if not btts_odds_row:
        return None

    BTTS_EDGE_MIN = 0.07

    for side, p_model, odd_key in (
        ("BTTS",    prob_btts,       "odd_btts"),
        ("No BTTS", 1 - prob_btts,   "odd_no_btts"),
    ):
        odd = btts_odds_row.get(odd_key)
        if not odd or odd <= 1.0:
            continue
        p_implied = round(1 / odd, 4)
        edge      = round(p_model - p_implied, 4)
        ev        = round(p_model * odd - 1, 4)

        if edge >= BTTS_EDGE_MIN and ev > 0:
            return {
                "result":      side,
                "result_name": "Les deux équipes marquent" if side == "BTTS" else "Au moins une équipe ne marque pas",
                "p_model":     round(p_model, 4),
                "p_implied":   p_implied,
                "edge":        edge,
                "odd":         round(odd, 3),
                "expected_value": ev,
                "is_value":    True,
            }
    return None


def build_btts_signal(features: dict, btts_odds_row: dict,
                      btts_model: "BettingModel", match_info: dict) -> dict | None:
    """Signal BTTS pour un match de foot. Retourne None si pas de value."""
    if not btts_model.is_trained:
        return None

    proba     = btts_model.predict_proba(features)
    prob_btts = proba.get("prob_over", proba.get("prob_home", 0))
    vb        = detect_btts_value_bet(prob_btts, btts_odds_row)
    if vb is None:
        return None

    return {
        "sport":        "football",
        "league":       match_info.get("league", ""),
        "home_team":    match_info.get("home_team", ""),
        "away_team":    match_info.get("away_team", ""),
        "match_date":   match_info.get("date", ""),
        "pred_result":  vb["result"],
        "pred_name":    vb["result_name"],
        "prob_home":    prob_btts,
        "prob_draw":    0.0,
        "prob_away":    round(1 - prob_btts, 4),
        "confidence":   round(vb["p_model"], 4),
        "method":       proba.get("method", "xgboost_btts"),
        "value_bets":   [vb],
        "is_value_bet": True,
        "edge":         vb["edge"],
        "odd_used":     vb["odd"],
        "market":       "BTTS",
    }


# ════════════════════════════════════════════════════════════
# ARBITRAGE MULTI-BOOKMAKERS — M
# ════════════════════════════════════════════════════════════

def detect_arbitrage(odds_by_bookmaker: dict) -> dict | None:
    """
    Détecte une opportunité d'arbitrage pur entre plusieurs bookmakers.

    odds_by_bookmaker : {
        "bookmaker_A": {"H": 2.10, "D": 3.40, "A": 3.20},
        "bookmaker_B": {"H": 2.05, "D": 3.50, "A": 3.30},
        ...
    }

    Méthode : prend la meilleure cote disponible par outcome,
    calcule la somme des inverses (1/odd). Si < 1.0 → arb détecté.

    Returns:
        dict avec best_odds, arb_pct (profit garanti en %), stakes_pct
        ou None si pas d'arb.
    """
    if not odds_by_bookmaker:
        return None

    outcomes = ["H", "D", "A"]
    best: dict[str, tuple[float, str]] = {}   # outcome → (meilleure cote, bookmaker)

    for bk, odds in odds_by_bookmaker.items():
        for outcome in outcomes:
            odd = odds.get(outcome, 0)
            if odd > 0:
                if outcome not in best or odd > best[outcome][0]:
                    best[outcome] = (odd, bk)

    # Seulement si on a les 3 outcomes
    if len(best) < 3:
        return None

    arb_sum = sum(1 / best[o][0] for o in outcomes)
    if arb_sum >= 1.0:
        return None   # Pas d'arbitrage

    arb_pct = round((1 - arb_sum) * 100, 3)   # % de profit garanti

    # Répartition des mises optimale (pour 100 unités investies)
    stakes_pct = {
        o: round(100 / (best[o][0] * arb_sum), 2)
        for o in outcomes
    }

    return {
        "arb_pct":    arb_pct,
        "arb_sum":    round(arb_sum, 4),
        "best_odds":  {o: {"odd": best[o][0], "bookmaker": best[o][1]} for o in outcomes},
        "stakes_pct": stakes_pct,   # % à miser sur chaque outcome pour garantir le profit
    }


def detect_implied_value(model_proba: dict, odds_by_bookmaker: dict,
                         min_edge: float = 0.03) -> list[dict]:
    """
    Compare les probas du modèle aux meilleures cotes disponibles
    parmi plusieurs bookmakers → détecte les inefficiences de marché.

    model_proba      : {"H": 0.45, "D": 0.28, "A": 0.27}
    odds_by_bookmaker: {"bk1": {"H": 2.5, ...}, "bk2": {...}}

    Returns liste de value bets triés par edge décroissant.
    """
    if not odds_by_bookmaker or not model_proba:
        return []

    outcomes = [o for o in ["H", "D", "A"] if o in model_proba]
    value_bets = []

    for outcome in outcomes:
        p_model = model_proba[outcome]
        # Meilleure cote disponible pour cet outcome
        best_odd = 0.0
        best_bk  = ""
        for bk, odds in odds_by_bookmaker.items():
            odd = odds.get(outcome, 0)
            if odd > best_odd:
                best_odd = odd
                best_bk  = bk

        if best_odd <= 1.0:
            continue

        p_implied = 1.0 / best_odd
        edge      = p_model - p_implied
        ev        = p_model * (best_odd - 1) - (1 - p_model)

        if edge >= min_edge and ev > 0:
            value_bets.append({
                "outcome":    outcome,
                "p_model":    round(p_model, 4),
                "p_implied":  round(p_implied, 4),
                "best_odd":   round(best_odd, 2),
                "bookmaker":  best_bk,
                "edge":       round(edge, 4),
                "ev":         round(ev, 4),
            })

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
    # Fusionner les cotes bookmaker dans les features
    # (le modèle football est entraîné avec impl_home/draw/away, odd_*, has_odds)
    enriched = dict(features)
    if odds_row:
        odd_h = odds_row.get("odd_home")
        odd_d = odds_row.get("odd_draw")
        odd_a = odds_row.get("odd_away")
        enriched.update({
            "impl_home":  odds_row.get("impl_home", 0.33),
            "impl_draw":  odds_row.get("impl_draw", 0.33),
            "impl_away":  odds_row.get("impl_away", 0.33),
            "odd_home":   odd_h or 2.5,
            "odd_draw":   odd_d or 3.3,
            "odd_away":   odd_a or 2.5,
            "has_odds":   1.0 if (odd_h and odd_d and odd_a) else 0.0,
        })

    proba = model.predict_proba(enriched)

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
        "edge":        value_bets[0]["edge"] if value_bets else 0.0,
        "top_odd":     value_bets[0]["odd"]  if value_bets else None,
        "odd_used":    value_bets[0]["odd"]  if value_bets else None,
    }

    return signal