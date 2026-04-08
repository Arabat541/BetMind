# ============================================================
# feature_engineering.py — Construction des features ML
# ============================================================

import logging
import numpy as np
import pandas as pd
from data_fetcher import (
    fetch_team_stats, fetch_h2h,
    fetch_nba_team_stats
)
from config import FORM_WINDOW, MIN_MATCHES_MODEL

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# FOOTBALL FEATURES
# ════════════════════════════════════════════════════════════

def build_football_features(home_id: int, away_id: int,
                             league_code: str = "", home_name: str = "",
                             away_name: str = "") -> dict | None:
    """
    Construit le vecteur de features pour un match de foot.
    Retourne None si les données sont insuffisantes.
    """
    logger.info(f"Building football features: {home_name} vs {away_name}")

    home_stats = fetch_team_stats(home_id, league_code)
    away_stats = fetch_team_stats(away_id, league_code)
    h2h_df     = fetch_h2h(home_id, away_id, last_n=10)

    if not home_stats or not away_stats:
        logger.warning(f"Insufficient stats for {home_name} or {away_name}")
        return None

    if home_stats.get("played", 0) < MIN_MATCHES_MODEL:
        logger.warning(f"Not enough matches for {home_name}")
        return None

    # ── H2H features ────────────────────────────────────────
    h2h_features = _compute_h2h_features(h2h_df, home_id)

    # ── Attack / Defense strength ────────────────────────────
    avg_goals_league = 2.5  # moyenne ligue (approximation)
    home_attack  = home_stats["goals_for_avg"] / max(avg_goals_league / 2, 0.1)
    home_defense = home_stats["goals_ag_avg"]  / max(avg_goals_league / 2, 0.1)
    away_attack  = away_stats["goals_for_avg"] / max(avg_goals_league / 2, 0.1)
    away_defense = away_stats["goals_ag_avg"]  / max(avg_goals_league / 2, 0.1)

    features = {
        # Forme récente (0-1)
        "home_form":            home_stats["form_score"],
        "away_form":            away_stats["form_score"],
        "form_diff":            home_stats["form_score"] - away_stats["form_score"],

        # Stats offensives / défensives
        "home_goals_for_avg":   home_stats["goals_for_avg"],
        "home_goals_ag_avg":    home_stats["goals_ag_avg"],
        "away_goals_for_avg":   away_stats["goals_for_avg"],
        "away_goals_ag_avg":    away_stats["goals_ag_avg"],

        # Ratios attack/defense
        "home_attack_str":      round(home_attack, 4),
        "home_defense_str":     round(home_defense, 4),
        "away_attack_str":      round(away_attack, 4),
        "away_defense_str":     round(away_defense, 4),

        # Taux de victoire (saison entière)
        "home_win_rate":        _win_rate(home_stats),
        "away_win_rate":        _win_rate(away_stats),
        "win_rate_diff":        _win_rate(home_stats) - _win_rate(away_stats),

        # Clean sheets & efficacité
        "home_clean_sheet_rate": _clean_sheet_rate(home_stats),
        "away_clean_sheet_rate": _clean_sheet_rate(away_stats),
        "home_scoring_rate":     _scoring_rate(home_stats),
        "away_scoring_rate":     _scoring_rate(away_stats),

        # Avantage domicile (feature booléenne)
        "home_advantage":       1.0,

        # H2H
        **h2h_features,
    }

    # Poisson lambda — buts attendus
    home_lambda = home_attack * away_defense * (avg_goals_league / 2)
    away_lambda = away_attack * home_defense * (avg_goals_league / 2)
    features["home_lambda"] = round(max(home_lambda, 0.1), 4)
    features["away_lambda"] = round(max(away_lambda, 0.1), 4)
    features["total_goals_exp"] = round(home_lambda + away_lambda, 4)

    return features


def _win_rate(stats: dict) -> float:
    played = stats.get("played", 0)
    if played == 0:
        return 0.0
    return round(stats.get("wins", 0) / played, 4)


def _clean_sheet_rate(stats: dict) -> float:
    played = stats.get("played", 0)
    if played == 0:
        return 0.0
    return round(stats.get("clean_sheets", 0) / played, 4)


def _scoring_rate(stats: dict) -> float:
    played = stats.get("played", 0)
    if played == 0:
        return 0.0
    failed = stats.get("failed_to_score", 0)
    return round(1 - failed / played, 4)


def _compute_h2h_features(h2h_df: pd.DataFrame, home_id: int) -> dict:
    """Features issues des confrontations directes."""
    if h2h_df.empty:
        return {
            "h2h_home_win_rate":  0.33,
            "h2h_draw_rate":      0.33,
            "h2h_away_win_rate":  0.33,
            "h2h_avg_goals":      2.5,
            "h2h_matches":        0,
        }

    n = len(h2h_df)
    home_wins  = len(h2h_df[h2h_df.apply(
        lambda r: (r["home_id"] == home_id and r["result"] == "H") or
                  (r["away_id"] == home_id and r["result"] == "A"), axis=1)])
    draws      = len(h2h_df[h2h_df["result"] == "D"])
    away_wins  = n - home_wins - draws
    avg_goals  = (h2h_df["home_goals"] + h2h_df["away_goals"]).mean()

    return {
        "h2h_home_win_rate": round(home_wins / n, 4),
        "h2h_draw_rate":     round(draws / n, 4),
        "h2h_away_win_rate": round(away_wins / n, 4),
        "h2h_avg_goals":     round(avg_goals, 4),
        "h2h_matches":       n,
    }


# ════════════════════════════════════════════════════════════
# NBA FEATURES
# ════════════════════════════════════════════════════════════

def build_nba_features(home_id: int, away_id: int,
                        home_name: str = "", away_name: str = "") -> dict | None:
    """
    Construit le vecteur de features pour un match NBA.
    """
    logger.info(f"Building NBA features: {home_name} vs {away_name}")

    home_stats = fetch_nba_team_stats(home_id)
    away_stats = fetch_nba_team_stats(away_id)

    if not home_stats or not away_stats:
        logger.warning(f"Insufficient NBA stats for {home_name} or {away_name}")
        return None

    if home_stats.get("games", 0) < MIN_MATCHES_MODEL:
        return None

    features = {
        # Taux de victoire
        "home_win_rate":      home_stats["win_rate"],
        "away_win_rate":      away_stats["win_rate"],
        "win_rate_diff":      home_stats["win_rate"] - away_stats["win_rate"],

        # Points marqués / encaissés
        "home_pts_for_avg":   home_stats["pts_for_avg"],
        "home_pts_ag_avg":    home_stats["pts_ag_avg"],
        "away_pts_for_avg":   away_stats["pts_for_avg"],
        "away_pts_ag_avg":    away_stats["pts_ag_avg"],

        # Différentiel de points
        "home_pts_diff":      home_stats["pts_diff_avg"],
        "away_pts_diff":      away_stats["pts_diff_avg"],
        "pts_diff_gap":       home_stats["pts_diff_avg"] - away_stats["pts_diff_avg"],

        # Efficacité offensive / défensive relative
        "home_off_eff":       round(home_stats["pts_for_avg"] / max(away_stats["pts_ag_avg"], 1), 4),
        "away_off_eff":       round(away_stats["pts_for_avg"] / max(home_stats["pts_ag_avg"], 1), 4),

        # Avantage domicile
        "home_advantage":     1.0,
    }

    return features


# ════════════════════════════════════════════════════════════
# FEATURES → DataFrame (pour batch prediction)
# ════════════════════════════════════════════════════════════

def features_to_dataframe(features: dict) -> pd.DataFrame:
    """Convertit un dict de features en DataFrame 1 ligne pour le modèle."""
    return pd.DataFrame([features])


def get_feature_columns(sport: str = "football") -> list:
    """Retourne la liste ordonnée des colonnes features."""
    if sport == "football":
        return [
            "home_form", "away_form", "form_diff",
            "home_goals_for_avg", "home_goals_ag_avg",
            "away_goals_for_avg", "away_goals_ag_avg",
            "home_attack_str", "home_defense_str",
            "away_attack_str", "away_defense_str",
            "home_win_rate", "away_win_rate", "win_rate_diff",
            "home_clean_sheet_rate", "away_clean_sheet_rate",
            "home_scoring_rate", "away_scoring_rate",
            "home_advantage",
            "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
            "h2h_avg_goals", "h2h_matches",
            "home_lambda", "away_lambda", "total_goals_exp",
        ]
    elif sport == "nba":
        return [
            "home_win_rate", "away_win_rate", "win_rate_diff",
            "home_pts_for_avg", "home_pts_ag_avg",
            "away_pts_for_avg", "away_pts_ag_avg",
            "home_pts_diff", "away_pts_diff", "pts_diff_gap",
            "home_off_eff", "away_off_eff",
            "home_advantage",
        ]
    return []