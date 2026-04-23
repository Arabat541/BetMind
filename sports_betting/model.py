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
        """Prédiction via XGBoost (ou Ensemble avec variance — U)."""
        from feature_engineering import get_feature_columns
        cols = self.feature_cols if self.feature_cols else get_feature_columns(self.sport)
        X = pd.DataFrame([{c: features.get(c, 0.0) for c in cols}])

        # Variance ensemble — U
        pred_variance = None
        if ENSEMBLE_AVAILABLE and hasattr(self.model, "predict_proba_with_variance"):
            try:
                proba_arr, var_arr = self.model.predict_proba_with_variance(X)
                proba = proba_arr[0]
                pred_variance = round(float(var_arr[0]), 4)
            except Exception:
                proba = self.model.predict_proba(X)[0]
        else:
            proba = self.model.predict_proba(X)[0]

        if self.sport == "ou_football":
            return {
                "prob_over":      round(float(proba[1]), 4),
                "prob_under":     round(float(proba[0]), 4),
                "prob_home":      round(float(proba[1]), 4),
                "prob_draw":      0.0,
                "prob_away":      round(float(proba[0]), 4),
                "pred_variance":  pred_variance,
                "method":         "xgboost_ou"
            }

        return {
            "prob_home":     round(float(proba[0]), 4),
            "prob_draw":     round(float(proba[1]), 4) if self.sport == "football" else 0.0,
            "prob_away":     round(float(proba[2 if self.sport == "football" else 1]), 4),
            "pred_variance": pred_variance,
            "method":        "xgboost"
        }

    def _predict_poisson_prior(self, features: dict) -> dict:
        """
        Calcule P(H), P(D), P(A) via Dixon-Coles (Poisson corrigé scores faibles).
        """
        from dixon_coles import dc_1x2, load_rho
        lam_h = features.get("home_lambda", 1.4)
        lam_a = features.get("away_lambda", 1.1)
        rho   = load_rho("global")
        probs = dc_1x2(lam_h, lam_a, rho)
        return {**probs, "method": "dixon_coles"}

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
# VIG REMOVAL — SHIN METHOD — T
# ════════════════════════════════════════════════════════════

def shin_probabilities(*odds: float) -> list[float]:
    """
    Méthode de Shin : supprime la marge bookmaker et corrige le biais
    favori/outsider pour obtenir des probabilités "vraies".

    Formule :
        p_shin_i = (sqrt(z² + 4(1-z) · p_raw_i²) - z) / (2(1-z))
    où z est résolu numériquement tel que sum(p_shin_i) = 1.

    Args:
        *odds : cotes décimales (ex: 1.85, 3.40, 4.20)
    Returns:
        liste de probas corrigées (sum = 1), même ordre que les cotes.
    """
    odds = [float(o) for o in odds if o and float(o) > 1.0]
    if not odds:
        n = len(odds) or 1
        return [1.0 / n] * n

    p_raw = [1.0 / o for o in odds]
    vig   = sum(p_raw) - 1.0

    if vig <= 0:
        # Déjà sans marge — normalisation simple
        s = sum(p_raw)
        return [p / s for p in p_raw]

    # Résolution numérique de z via dichotomie (converge en ~30 itérations)
    lo, hi = 0.0, 0.5
    for _ in range(60):
        z   = (lo + hi) / 2.0
        ps  = [(np.sqrt(z**2 + 4.0 * (1.0 - z) * p**2) - z) / (2.0 * (1.0 - z))
               for p in p_raw]
        s   = sum(ps)
        if abs(s - 1.0) < 1e-10:
            break
        if s > 1.0:
            lo = z
        else:
            hi = z

    # Renormalise pour garantir sum = 1 (erreurs numériques)
    s = sum(ps)
    return [round(p / s, 6) for p in ps]


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

    odd_h = odds_row.get("odd_home")
    odd_d = odds_row.get("odd_draw")
    odd_a = odds_row.get("odd_away")

    # Shin method : supprime le vig pour avoir de vraies probas implicites — T
    if odd_h and odd_a:
        if sport == "football" and odd_d:
            shin_h, shin_d, shin_a = shin_probabilities(odd_h, odd_d, odd_a)
        else:
            shin_h, shin_a = shin_probabilities(odd_h, odd_a)
            shin_d = 0.0
    else:
        shin_h = odds_row.get("impl_home", 0.33)
        shin_d = odds_row.get("impl_draw", 0.33)
        shin_a = odds_row.get("impl_away", 0.33)

    candidates = [
        ("H", proba.get("prob_home", 0), shin_h, odd_h),
        ("A", proba.get("prob_away", 0), shin_a, odd_a),
    ]
    if sport == "football":
        candidates.append(
            ("D", proba.get("prob_draw", 0), shin_d, odd_d)
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

    # AI — Confirmation Dixon-Coles : aligner XGBoost et DC avant de signaler
    try:
        from dixon_coles import dc_over_under, load_rho
        lam_h = features.get("home_lambda", 1.4)
        lam_a = features.get("away_lambda", 1.1)
        rho   = load_rho("global")
        dc_ou = dc_over_under(lam_h, lam_a, rho)
        dc_over = dc_ou["prob_over"]
        # Moyenne pondérée : XGBoost 70% + DC 30%
        prob_over = round(0.70 * prob_over + 0.30 * dc_over, 4)
    except Exception:
        pass

    vb = detect_ou_value_bet(prob_over, ou_odds_row)
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
        "method":      proba.get("method", "xgboost_ou") + "+dc",
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
# AL — CORRECT SCORE (Dixon-Coles bivarié)
# ════════════════════════════════════════════════════════════

CS_EDGE_MIN   = 0.12   # edge minimum pour Correct Score (marché à haute variance)
CS_MIN_PROB   = 0.08   # probabilité modèle minimum pour alerter

def build_correct_score_signal(features: dict, cs_odds: dict,
                               match_info: dict) -> dict | None:
    """
    Détecte des value bets sur le marché Correct Score.

    cs_odds : dict {"1-0": 7.50, "2-1": 9.00, ...} — cotes bookmaker par score.
    Retourne le meilleur value bet ou None.
    """
    if not cs_odds:
        return None

    try:
        from dixon_coles import dc_correct_scores, load_rho
    except ImportError:
        return None

    lam_h = features.get("home_lambda", 1.4)
    lam_a = features.get("away_lambda", 1.1)
    rho   = load_rho("global")

    model_scores = dc_correct_scores(lam_h, lam_a, rho, top_n=25)
    model_dict   = {s["score"]: s["prob"] for s in model_scores}

    best_vb = None
    for score_key, odd in cs_odds.items():
        if not isinstance(odd, (int, float)) or odd <= 1:
            continue
        p_model   = model_dict.get(score_key, 0.0)
        p_implied = 1.0 / odd
        edge      = p_model - p_implied
        ev        = p_model * (odd - 1) - (1 - p_model)

        if p_model < CS_MIN_PROB:
            continue
        if edge < CS_EDGE_MIN:
            continue
        if ev <= 0:
            continue

        vb = {
            "score":     score_key,
            "p_model":   round(p_model,   4),
            "p_implied": round(p_implied, 4),
            "edge":      round(edge,      4),
            "odd":       round(odd,       3),
            "ev":        round(ev,        4),
        }
        if best_vb is None or vb["edge"] > best_vb["edge"]:
            best_vb = vb

    if best_vb is None:
        return None

    # Top 5 scores modèle pour affichage dans l'alerte
    top5 = model_scores[:5]

    return {
        "sport":        "football",
        "league":       match_info.get("league", ""),
        "home_team":    match_info.get("home_team", ""),
        "away_team":    match_info.get("away_team", ""),
        "match_date":   match_info.get("date", ""),
        "pred_result":  f"CS_{best_vb['score']}",
        "pred_name":    f"Score exact {best_vb['score']}",
        "prob_home":    best_vb["p_model"],
        "prob_draw":    0.0,
        "prob_away":    0.0,
        "confidence":   best_vb["p_model"],
        "method":       "dixon_coles_cs",
        "is_value_bet": True,
        "edge":         best_vb["edge"],
        "odd_used":     best_vb["odd"],
        "market":       "CS",
        "value_bets":   [best_vb],
        "top_scores":   top5,
    }


def fetch_correct_score_odds(league_key: str) -> dict:
    """
    Récupère les cotes Correct Score depuis The Odds API.
    Retourne dict {home_team|away_team: {"1-0": 7.5, ...}}.
    """
    try:
        from data_fetcher import _odds_get
        data = _odds_get(f"sports/{league_key}/odds", {
            "regions": "eu",
            "markets": "h2h_lay,scorers",
            "oddsFormat": "decimal",
        }, ttl=180)
        if not data or not isinstance(data, list):
            return {}

        result = {}
        for event in data:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            key  = f"{home}|{away}"
            cs_odds = {}
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") == "correct_score":
                        for outcome in market.get("outcomes", []):
                            name = outcome.get("name", "")
                            price = outcome.get("price", 0)
                            # Format "2-1" ou "Home 2-1" → normalise
                            parts = name.replace("Home ", "").replace("Away ", "").strip()
                            if "-" in parts:
                                cs_odds[parts] = float(price)
                        break
                if cs_odds:
                    break
            if cs_odds:
                result[key] = cs_odds
        return result
    except Exception as e:
        logger.debug(f"fetch_correct_score_odds: {e}")
        return {}


# ════════════════════════════════════════════════════════════
# AM — REVERSE LINE MOVEMENT
# ════════════════════════════════════════════════════════════

RLM_MIN_LINE_MOVE = 0.04   # la cote doit bouger d'au moins 4% depuis l'ouverture
RLM_MIN_PUBLIC    = 0.55   # au moins 55% des mises publiques du mauvais côté

def detect_rlm(odds_opening: dict, odds_current: dict,
               public_bets_pct: dict | None = None) -> list[dict]:
    """
    Détecte un Reverse Line Movement sur les cotes 1X2.

    odds_opening  : {"H": 2.10, "D": 3.40, "A": 3.20} — cotes à l'ouverture
    odds_current  : {"H": 2.20, "D": 3.35, "A": 3.10} — cotes actuelles
    public_bets_pct : {"H": 0.60, "D": 0.20, "A": 0.20} — % billets public
                      Si None, le critère public est ignoré.

    Retourne une liste de signals RLM :
    [{"outcome": "H", "odd_open": 2.10, "odd_current": 2.20,
      "line_move_pct": 0.048, "public_pct": 0.60, "rlm_score": 0.85}, ...]

    Logique :
      Si la cote MONTE (odd_current > odd_open) d'au moins RLM_MIN_LINE_MOVE
      et que le public mise majoritairement SUR ce côté (public_pct >= RLM_MIN_PUBLIC),
      c'est un RLM : les books défavorisent ce outcome malgré le public.
      → Les sharps ont misé sur l'autre côté et les books se couvrent.
    """
    signals = []
    for outcome in ("H", "D", "A"):
        open_ = odds_opening.get(outcome)
        curr  = odds_current.get(outcome)
        if not open_ or not curr or open_ <= 1 or curr <= 1:
            continue

        line_move = (curr - open_) / open_   # positif = cote monte
        if line_move < RLM_MIN_LINE_MOVE:
            continue

        pub = (public_bets_pct or {}).get(outcome, 0.0)
        if public_bets_pct and pub < RLM_MIN_PUBLIC:
            continue

        # RLM score : produit normalisé des deux signaux
        move_score = min(line_move / 0.10, 1.0)    # 10% move → score max
        pub_score  = min((pub - 0.5) / 0.3, 1.0) if public_bets_pct else 0.5
        rlm_score  = round((move_score + pub_score) / 2, 3)

        signals.append({
            "outcome":       outcome,
            "odd_open":      round(open_, 3),
            "odd_current":   round(curr,  3),
            "line_move_pct": round(line_move, 4),
            "public_pct":    round(pub, 3),
            "rlm_score":     rlm_score,
        })

    signals.sort(key=lambda x: x["rlm_score"], reverse=True)
    return signals


def fetch_opening_odds(league_key: str) -> dict:
    """
    Récupère les cotes d'ouverture via The Odds API (bookmaker pinnacle ou
    premier dispo avec historique). Retourne dict {home|away: {"H":…, "D":…, "A":…}}.
    """
    try:
        from data_fetcher import _odds_get
        data = _odds_get(f"sports/{league_key}/odds", {
            "regions":    "eu",
            "markets":    "h2h",
            "oddsFormat": "decimal",
            "bookmakers": "pinnacle,betfair_ex_eu,unibet",
        }, ttl=300)
        if not data or not isinstance(data, list):
            return {}

        result = {}
        for event in data:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            key  = f"{home}|{away}"
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = market.get("outcomes", [])
                    h_odd = next((o["price"] for o in outcomes if o["name"] == home), None)
                    d_odd = next((o["price"] for o in outcomes if o["name"] == "Draw"), None)
                    a_odd = next((o["price"] for o in outcomes if o["name"] == away), None)
                    if h_odd and a_odd:
                        result[key] = {
                            "H": float(h_odd),
                            "D": float(d_odd) if d_odd else None,
                            "A": float(a_odd),
                        }
                        break
                if key in result:
                    break
        return result
    except Exception as e:
        logger.debug(f"fetch_opening_odds: {e}")
        return {}


# ════════════════════════════════════════════════════════════
# AQ — BOTH HALVES GOALS (BHG)
# ════════════════════════════════════════════════════════════
# Les deux équipes marquent dans CHAQUE mi-temps (not just overall BTTS).
# Probabilité dérivée du modèle de Poisson bivarié Dixon-Coles.
# On suppose indépendance des deux mi-temps et distribution symétrique
# (chaque équipe marque ~50% de ses buts en chaque mi-temps).

BHG_EDGE_MIN = 0.10   # edge minimum — marché moins liquide que BTTS
BHG_MIN_PROB = 0.15   # prob modèle minimum pour alerter

def _bhg_probability(home_lambda: float, away_lambda: float,
                     home_half_ratio: float = 0.5,
                     away_half_ratio: float = 0.5) -> float:
    """
    Probabilité que les deux équipes marquent dans chaque mi-temps.

    On modélise chaque mi-temps comme Poisson indépendant :
      - λ_home_H1 = home_lambda * home_half_ratio
      - λ_away_H1 = away_lambda * away_half_ratio

    P(BHG) = P(home scores in H1) * P(away scores in H1)
           * P(home scores in H2) * P(away scores in H2)

    P(équipe marque dans une mi-temps) = 1 - P(0 but) = 1 - e^{-λ}
    """
    lh1 = home_lambda * home_half_ratio
    la1 = away_lambda * away_half_ratio
    lh2 = home_lambda * (1 - home_half_ratio)
    la2 = away_lambda * (1 - away_half_ratio)

    p_h1 = 1 - np.exp(-lh1)
    p_a1 = 1 - np.exp(-la1)
    p_h2 = 1 - np.exp(-lh2)
    p_a2 = 1 - np.exp(-la2)

    return round(p_h1 * p_a1 * p_h2 * p_a2, 4)


def build_bhg_signal(features: dict, bhg_odds_row: dict,
                     match_info: dict) -> dict | None:
    """
    Détecte un value bet sur le marché Both Halves Goals.

    bhg_odds_row : dict {"odd_bhg": float, "odd_no_bhg": float}
    Retourne le signal value bet ou None.
    """
    if not bhg_odds_row:
        return None

    odd_bhg    = bhg_odds_row.get("odd_bhg")
    odd_no_bhg = bhg_odds_row.get("odd_no_bhg")

    if not odd_bhg:
        return None

    # Probabilité modèle via Poisson
    home_lam = features.get("home_lambda", 1.2)
    away_lam = features.get("away_lambda", 1.0)

    # Ratios mi-temps ajustés si on a des stats HT disponibles
    home_ratio = features.get("home_ht_goal_ratio", 0.5)
    away_ratio = features.get("away_ht_goal_ratio", 0.5)

    prob_bhg = _bhg_probability(home_lam, away_lam, home_ratio, away_ratio)

    result = None
    for side, prob, odd in [
        ("BHG",    prob_bhg,        odd_bhg),
        ("No BHG", 1 - prob_bhg,    odd_no_bhg),
    ]:
        if odd is None or odd <= 1.0:
            continue
        try:
            odd = float(odd)
        except (TypeError, ValueError):
            continue
        impl = round(1 / odd, 4)
        edge = round(prob - impl, 4)
        ev   = round(prob * odd - 1, 4)
        if edge >= BHG_EDGE_MIN and ev > 0 and prob >= BHG_MIN_PROB:
            result = {
                "side":        side,
                "prob_model":  prob,
                "prob_impl":   impl,
                "edge":        edge,
                "ev":          ev,
                "odd":         odd,
                "result_name": ("Les deux équipes marquent dans chaque mi-temps"
                                if side == "BHG"
                                else "Pas de BHG dans les deux mi-temps"),
                **match_info,
            }

    return result


def fetch_bhg_odds(league_key: str) -> dict:
    """
    Récupère les cotes Both Halves Goals via The Odds API.
    Marché : "both_halves_goals" (si disponible) ou "btts" (proxy).
    Retourne {event_id: {"odd_bhg": float, "odd_no_bhg": float}}.
    """
    from config import THE_ODDS_API_KEY, ODDS_API_BASE
    if not THE_ODDS_API_KEY:
        return {}

    try:
        import requests
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{league_key}/odds",
            params={
                "apiKey":   THE_ODDS_API_KEY,
                "markets":  "both_halves_goals",
                "regions":  "eu",
                "oddsFormat": "decimal",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return {}

        out = {}
        for event in resp.json():
            eid = event.get("id", "")
            for book in event.get("bookmakers", []):
                for mkt in book.get("markets", []):
                    if mkt.get("key") not in ("both_halves_goals", "both_teams_to_score"):
                        continue
                    o = {}
                    for out_come in mkt.get("outcomes", []):
                        nm = out_come.get("name", "").lower()
                        price = out_come.get("price")
                        if nm in ("yes", "bhg"):
                            o["odd_bhg"] = price
                        elif nm in ("no", "no bhg"):
                            o["odd_no_bhg"] = price
                    if o:
                        out[eid] = o
                        break
                if eid in out:
                    break
        return out
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════
# AS — LINE SHOPPING MULTI-BOOKMAKERS
# ════════════════════════════════════════════════════════════

def get_best_odds(league_key: str, home_team: str, away_team: str) -> dict:
    """
    Récupère les meilleures cotes 1X2 disponibles sur 5+ bookmakers via The Odds API.

    Retourne:
    {
      "best_odd_home":  float,  "best_book_home":  str,
      "best_odd_draw":  float,  "best_book_draw":  str,
      "best_odd_away":  float,  "best_book_away":  str,
      "bookmakers_checked": int,
    }
    ou dict vide si erreur.
    """
    from config import THE_ODDS_API_KEY, ODDS_API_BASE
    if not THE_ODDS_API_KEY:
        return {}

    try:
        import requests
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{league_key}/odds",
            params={
                "apiKey":     THE_ODDS_API_KEY,
                "markets":    "h2h",
                "regions":    "eu,uk,us",
                "oddsFormat": "decimal",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return {}

        home_lower = home_team.lower().split()[0]
        away_lower = away_team.lower().split()[0]

        target_event = None
        for event in resp.json():
            h = event.get("home_team", "").lower()
            a = event.get("away_team", "").lower()
            if home_lower in h and away_lower in a:
                target_event = event
                break

        if not target_event:
            return {}

        best = {"H": (0.0, ""), "D": (0.0, ""), "A": (0.0, "")}
        n_books = 0

        for book in target_event.get("bookmakers", []):
            bk_name = book.get("title", "")
            for mkt in book.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                n_books += 1
                for o in mkt.get("outcomes", []):
                    nm = o.get("name", "")
                    p  = float(o.get("price", 0))
                    if nm == target_event.get("home_team") and p > best["H"][0]:
                        best["H"] = (p, bk_name)
                    elif nm == target_event.get("away_team") and p > best["A"][0]:
                        best["A"] = (p, bk_name)
                    elif nm == "Draw" and p > best["D"][0]:
                        best["D"] = (p, bk_name)

        return {
            "best_odd_home":  best["H"][0],
            "best_book_home": best["H"][1],
            "best_odd_draw":  best["D"][0],
            "best_book_draw": best["D"][1],
            "best_odd_away":  best["A"][0],
            "best_book_away": best["A"][1],
            "bookmakers_checked": n_books,
        }
    except Exception:
        return {}


def enrich_signal_with_line_shop(signal: dict, league_key: str) -> dict:
    """
    Enrichit un signal avec les meilleures cotes disponibles (AS).
    Ajoute best_odd_available, best_book, line_shop_gain.
    """
    home = signal.get("home_team", "")
    away = signal.get("away_team", "")
    best = get_best_odds(league_key, home, away)
    if not best:
        return signal

    result = signal.get("predicted_result", signal.get("side", ""))
    outcome_map = {"H": "home", "D": "draw", "A": "away"}
    key = outcome_map.get(result, "home")

    best_odd = best.get(f"best_odd_{key}", 0.0)
    best_bk  = best.get(f"best_book_{key}", "")
    used_odd = signal.get("odd_used", signal.get("odd", 0.0))

    signal["best_odd_available"] = best_odd
    signal["best_book"]          = best_bk
    signal["line_shop_gain"]     = round(best_odd - used_odd, 3) if best_odd and used_odd else 0.0
    signal["bookmakers_checked"] = best.get("bookmakers_checked", 0)

    if best_odd > used_odd + 0.05:
        logger.info(
            f"  [AS] Line shop: {home} vs {away} — use {best_bk} @ {best_odd:.2f} "
            f"vs current {used_odd:.2f} (+{best_odd - used_odd:.2f})"
        )
    return signal


# ════════════════════════════════════════════════════════════
# AR — DUTCHING MULTI-OUTCOMES
# ════════════════════════════════════════════════════════════
# Répartir la mise sur plusieurs outcomes pour profit garanti
# quand la somme des probabilités implicites < 1 (sur-cotes réelles).

def dutch_stakes(outcomes: list[dict], total_stake: float) -> list[dict]:
    """
    Calcule les mises Dutching pour garantir un profit identique
    quel que soit le résultat parmi les outcomes sélectionnés.

    outcomes : [{"name": str, "odd": float, "prob_model": float}, ...]
    total_stake : mise totale en unités monétaires.

    Retourne outcomes enrichis avec "dutch_stake" et "dutch_return".
    Retourne [] si pas de profit garanti possible.

    Formule Dutch :
      stake_i = total_stake * (1/odd_i) / sum(1/odd_j)
      guaranteed_return = total_stake / sum(1/odd_j)
      profit si guaranteed_return > total_stake (sum impl < 1)
    """
    if not outcomes:
        return []

    odds = [float(o["odd"]) for o in outcomes]
    if any(od <= 1.0 for od in odds):
        return []

    impl_sum = sum(1 / od for od in odds)
    if impl_sum >= 1.0:
        return []  # pas d'opportunité Dutch (somme prob implicites ≥ 1)

    guaranteed_return = total_stake / impl_sum
    profit = guaranteed_return - total_stake

    result = []
    for o, od in zip(outcomes, odds):
        stake_i = round(total_stake * (1 / od) / impl_sum, 2)
        result.append({
            **o,
            "dutch_stake":  stake_i,
            "dutch_return": round(guaranteed_return, 2),
            "dutch_profit": round(profit, 2),
            "dutch_roi":    round(profit / total_stake, 4) if total_stake > 0 else 0.0,
        })
    return result


def detect_dutching_opportunity(odds_row: dict,
                                proba: dict,
                                match_info: dict,
                                total_stake: float = 100.0,
                                min_profit_pct: float = 0.02) -> dict | None:
    """
    Détecte une opportunité Dutching sur les 3 issues 1X2.

    Cherche les combinaisons Home+Away, Home+Draw, Draw+Away
    et la triple Home+Draw+Away.

    Retourne la meilleure opportunité ou None.
    """
    if not odds_row:
        return None

    odd_h = odds_row.get("odd_home", 0)
    odd_d = odds_row.get("odd_draw", 0)
    odd_a = odds_row.get("odd_away", 0)

    candidates = [
        [{"name": "H", "odd": odd_h, "label": "Domicile"},
         {"name": "A", "odd": odd_a, "label": "Extérieur"}],
        [{"name": "H", "odd": odd_h, "label": "Domicile"},
         {"name": "D", "odd": odd_d, "label": "Nul"}],
        [{"name": "D", "odd": odd_d, "label": "Nul"},
         {"name": "A", "odd": odd_a, "label": "Extérieur"}],
        [{"name": "H", "odd": odd_h, "label": "Domicile"},
         {"name": "D", "odd": odd_d, "label": "Nul"},
         {"name": "A", "odd": odd_a, "label": "Extérieur"}],
    ]

    best = None
    for combo in candidates:
        if any(o["odd"] <= 1.0 for o in combo):
            continue
        result = dutch_stakes(combo, total_stake)
        if not result:
            continue
        roi = result[0]["dutch_roi"]
        if roi < min_profit_pct:
            continue
        if best is None or roi > best["dutch_roi"]:
            best = {
                "outcomes":    result,
                "dutch_roi":   roi,
                "dutch_profit": result[0]["dutch_profit"],
                "total_stake": total_stake,
                "impl_sum":    round(sum(1 / o["odd"] for o in combo), 4),
                **match_info,
            }
    return best


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
            "impl_home":    odds_row.get("impl_home", 0.33),
            "impl_draw":    odds_row.get("impl_draw", 0.33),
            "impl_away":    odds_row.get("impl_away", 0.33),
            "odd_home":     odd_h or 2.5,
            "odd_draw":     odd_d or 3.3,
            "odd_away":     odd_a or 2.5,
            "has_odds":     1.0 if (odd_h and odd_d and odd_a) else 0.0,
            # Closing odds proxy (AB) — à l'inférence, cotes actuelles = proxy fermeture
            "impl_cl_home": odds_row.get("impl_home", 0.33),
            "impl_cl_draw": odds_row.get("impl_draw", 0.33),
            "impl_cl_away": odds_row.get("impl_away", 0.33),
            "cl_move_home": 0.0,
            "cl_move_draw": 0.0,
            "cl_move_away": 0.0,
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
        "confidence":     round(confidence, 4),
        "method":         proba.get("method", ""),
        "value_bets":     value_bets,
        "is_value_bet":   len(value_bets) > 0,
        "edge":           value_bets[0]["edge"] if value_bets else 0.0,
        "top_odd":        value_bets[0]["odd"]  if value_bets else None,
        "odd_used":       value_bets[0]["odd"]  if value_bets else None,
        # Variance ensemble — U : std max entre XGB et LGB (None si pas d'ensemble)
        "pred_variance":  proba.get("pred_variance"),
    }

    # Filtre variance — U : si désaccord fort entre base-learners, retire le signal value
    variance = proba.get("pred_variance")
    if variance is not None and variance > 0.08 and signal["is_value_bet"]:
        logger.info(
            f"  [Variance filter] Désaccord XGB/LGB trop élevé "
            f"(std={variance:.3f} > 0.08) — value bet retiré."
        )
        signal["value_bets"]  = []
        signal["is_value_bet"] = False
        signal["edge"]         = 0.0

    return signal