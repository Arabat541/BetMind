# ============================================================
# feature_engineering.py — Construction des features ML
# ============================================================

import logging
import numpy as np
import pandas as pd
from data_fetcher import (
    fetch_team_stats, fetch_h2h, fetch_standings, get_team_standing,
    fetch_nba_team_stats, get_injury_stats, get_team_shots_stats,
    fetch_match_weather, get_team_elo,
)
from config import FORM_WINDOW, FORM_WINDOW_LONG, MIN_MATCHES_MODEL
from understat_fetcher import load_xg_history, get_team_xg_rolling
from transfermarkt_fetcher import load_squad_values, get_squad_value_features

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# FOOTBALL FEATURES
# ════════════════════════════════════════════════════════════

def build_football_features(home_id: int, away_id: int,
                             league_code: str = "", home_name: str = "",
                             away_name: str = "",
                             league_name: str = "") -> dict | None:
    """
    Construit le vecteur de features pour un match de foot.
    Retourne None si les données sont insuffisantes.
    """
    logger.info(f"Building football features: {home_name} vs {away_name}")

    home_stats   = fetch_team_stats(home_id, league_code)
    away_stats   = fetch_team_stats(away_id, league_code)
    h2h_df       = fetch_h2h(home_id, away_id, last_n=10)
    standings_df = fetch_standings(league_code) if league_code else pd.DataFrame()
    home_std     = get_team_standing(standings_df, home_id)
    away_std     = get_team_standing(standings_df, away_id)

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

    home_form_long = home_stats.get("form_score_long", home_stats["form_score"])
    away_form_long = away_stats.get("form_score_long", away_stats["form_score"])

    features = {
        # Forme récente 5 matchs (0-1)
        "home_form":            home_stats["form_score"],
        "away_form":            away_stats["form_score"],
        "form_diff":            home_stats["form_score"] - away_stats["form_score"],
        # Forme longue 10 matchs (0-1)
        "home_form_long":       home_form_long,
        "away_form_long":       away_form_long,
        "form_diff_long":       home_form_long - away_form_long,

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

        # Classement (points/match dans la saison)
        "home_pts_per_game":    home_std["pts_per_game"],
        "away_pts_per_game":    away_std["pts_per_game"],
        "pts_per_game_diff":    round(home_std["pts_per_game"] - away_std["pts_per_game"], 4),

        # Forme spécifique domicile / extérieur
        "home_form_home":       home_stats.get("home_form_score", home_stats["form_score"]),
        "away_form_away":       away_stats.get("away_form_score", away_stats["form_score"]),
        "home_away_form_diff":  home_stats.get("home_form_score", home_stats["form_score"])
                                - away_stats.get("away_form_score", away_stats["form_score"]),

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

    # ── Shots (proxy xG) — A ─────────────────────────────────
    home_sh = get_team_shots_stats(home_name, league_name)
    away_sh = get_team_shots_stats(away_name, league_name)
    features.update({
        "home_sot_avg":    home_sh["sot_avg"],
        "away_sot_avg":    away_sh["sot_avg"],
        "home_sot_ag_avg": home_sh["sot_ag_avg"],
        "away_sot_ag_avg": away_sh["sot_ag_avg"],
        "home_shots_avg":  home_sh["shots_avg"],
        "away_shots_avg":  away_sh["shots_avg"],
        "home_sot_ratio":  home_sh["sot_ratio"],
        "away_sot_ratio":  away_sh["sot_ratio"],
    })

    # ── Fatigue / calendrier — E ─────────────────────────────
    features.update({
        "home_days_since_last": home_stats.get("days_since_last_match", 7),
        "away_days_since_last": away_stats.get("days_since_last_match", 7),
        "home_fatigue":         home_stats.get("matches_last_10days", 1),
        "away_fatigue":         away_stats.get("matches_last_10days", 1),
    })

    # ── Motivation / enjeu — G ───────────────────────────────
    features.update({
        "home_relegation_gap": home_std.get("relegation_gap", 0.0),
        "away_relegation_gap": away_std.get("relegation_gap", 0.0),
        "home_title_gap":      home_std.get("title_gap", 0.5),
        "away_title_gap":      away_std.get("title_gap", 0.5),
    })

    # ── Météo — J ────────────────────────────────────────────
    weather = fetch_match_weather(home_name)
    features["rainy_match"] = float(weather["rainy_match"])

    # ── ELO dynamique — S ────────────────────────────────────
    home_elo = get_team_elo(home_name)
    away_elo = get_team_elo(away_name)
    features.update({
        "home_elo":  home_elo,
        "away_elo":  away_elo,
        "elo_diff":  round(home_elo - away_elo, 1),
    })

    # ── xG réel Understat — W ────────────────────────────────
    xg_hist  = load_xg_history()
    today    = __import__("datetime").date.today().isoformat()
    h_xg     = get_team_xg_rolling(xg_hist, home_name, before_date=today, window=8)
    a_xg     = get_team_xg_rolling(xg_hist, away_name, before_date=today, window=8)
    features.update({
        "home_xg_avg":  h_xg["xg_avg"],
        "away_xg_avg":  a_xg["xg_avg"],
        "home_xga_avg": h_xg["xga_avg"],
        "away_xga_avg": a_xg["xga_avg"],
        "xg_diff":  round(h_xg["xg_avg"]  - a_xg["xg_avg"],  4),
        "xga_diff": round(h_xg["xga_avg"] - a_xg["xga_avg"], 4),
    })

    # ── Valeur marchande effectifs — Y ───────────────────────
    sq_vals = load_squad_values()
    features.update(get_squad_value_features(sq_vals, home_name, away_name))

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
                        home_name: str = "", away_name: str = "",
                        injuries_dict: dict | None = None) -> dict | None:
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

    # ── Blessures (ESPN) ────────────────────────────────────
    home_inj = get_injury_stats(injuries_dict or {}, home_name)
    away_inj = get_injury_stats(injuries_dict or {}, away_name)
    features["home_injuries_out"] = home_inj["out"]
    features["home_injuries_dtd"] = home_inj["day_to_day"]
    features["away_injuries_out"] = away_inj["out"]
    features["away_injuries_dtd"] = away_inj["day_to_day"]
    # impact_diff > 0 = avantage domicile (l'adverse est plus touché)
    features["injury_diff"]       = round(away_inj["impact"] - home_inj["impact"], 4)

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
            "home_form_long", "away_form_long", "form_diff_long",
            "home_goals_for_avg", "home_goals_ag_avg",
            "away_goals_for_avg", "away_goals_ag_avg",
            "home_attack_str", "home_defense_str",
            "away_attack_str", "away_defense_str",
            "home_win_rate", "away_win_rate", "win_rate_diff",
            "home_clean_sheet_rate", "away_clean_sheet_rate",
            "home_scoring_rate", "away_scoring_rate",
            "home_pts_per_game", "away_pts_per_game", "pts_per_game_diff",
            "home_form_home", "away_form_away", "home_away_form_diff",
            "home_advantage",
            "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
            "h2h_avg_goals", "h2h_matches",
            "home_lambda", "away_lambda", "total_goals_exp",
            # Shots (proxy xG) — A
            "home_sot_avg", "away_sot_avg",
            "home_sot_ag_avg", "away_sot_ag_avg",
            "home_shots_avg", "away_shots_avg",
            "home_sot_ratio", "away_sot_ratio",
            # Fatigue — E
            "home_days_since_last", "away_days_since_last",
            "home_fatigue",         "away_fatigue",
            # Motivation — G
            "home_relegation_gap", "away_relegation_gap",
            "home_title_gap",      "away_title_gap",
            # Météo — J
            "rainy_match",
            # ELO dynamique — S
            "home_elo", "away_elo", "elo_diff",
            # Cotes de fermeture — AB
            "impl_cl_home", "impl_cl_draw", "impl_cl_away",
            "cl_move_home", "cl_move_draw", "cl_move_away",
            # xG réel Understat — W
            "home_xg_avg", "away_xg_avg",
            "home_xga_avg", "away_xga_avg",
            "xg_diff", "xga_diff",
            # Valeur marchande effectifs — Y
            "home_squad_value", "away_squad_value", "squad_value_ratio",
        ]
    elif sport == "ou_football":
        return [
            "home_goals_for_avg", "home_goals_ag_avg",
            "away_goals_for_avg", "away_goals_ag_avg",
            "home_attack_str",    "home_defense_str",
            "away_attack_str",    "away_defense_str",
            "home_scoring_rate",  "away_scoring_rate",
            "home_lambda",        "away_lambda",
            "total_goals_exp",    "h2h_avg_goals",
            # Shots (proxy xG) — A
            "home_sot_avg", "away_sot_avg",
            "home_sot_ag_avg", "away_sot_ag_avg",
            # Fatigue — E
            "home_days_since_last", "away_days_since_last",
            "home_fatigue",         "away_fatigue",
        ]
    elif sport == "btts_football":
        return [
            "home_goals_for_avg", "home_goals_ag_avg",
            "away_goals_for_avg", "away_goals_ag_avg",
            "home_attack_str",    "home_defense_str",
            "away_attack_str",    "away_defense_str",
            "home_scoring_rate",  "away_scoring_rate",
            "home_clean_sheet_rate", "away_clean_sheet_rate",
            "h2h_avg_goals",      "btts_h2h_rate",
            "home_sot_avg",       "away_sot_avg",
            "home_sot_ag_avg",    "away_sot_ag_avg",
            "home_days_since_last", "away_days_since_last",
            "home_fatigue",         "away_fatigue",
        ]
    elif sport == "nba":
        return [
            "home_win_rate", "away_win_rate", "win_rate_diff",
            "home_pts_for_avg", "home_pts_ag_avg",
            "away_pts_for_avg", "away_pts_ag_avg",
            "home_pts_diff", "away_pts_diff", "pts_diff_gap",
            "home_off_eff", "away_off_eff",
            "home_advantage",
            "home_injuries_out", "home_injuries_dtd",
            "away_injuries_out", "away_injuries_dtd",
            "injury_diff",
        ]
    return []