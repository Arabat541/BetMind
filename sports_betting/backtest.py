"""
backtest.py — Simulation historique de la stratégie BetMind
============================================================
Rejoue chaque match des CSV football-data.co.uk dans l'ordre chronologique.
Pour chaque match avec un signal de value bet : simule la mise Kelly
et enregistre le résultat réel.

Produit : ROI, win rate, max drawdown, Sharpe, breakdown par ligue.

Usage : cd sports_betting && python backtest.py [--season 2425]
"""

import io
import os
import sys
import logging
import pickle
import argparse

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODELS_DIR, INITIAL_BANKROLL,
    VALUE_BET_EDGE, CONFIDENCE_THRESHOLD,
    MIN_ODD_ALLOWED, MAX_ODD_ALLOWED, MAX_EDGE_SANITY, MIN_EV_REQUIRED,
    KELLY_FRACTION, MAX_BET_PCT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL    = "https://www.football-data.co.uk/mmz4281"
LEAGUES     = {"Premier League": "E0", "Ligue 1": "F1",
               "Bundesliga": "D1", "La Liga": "SP1", "Serie A": "I1"}
ALL_SEASONS = ["2425", "2324", "2223"]

FORM_WINDOW        = 5
MIN_MATCHES_BEFORE = 3
AVG_GOALS          = 2.5

ODD_HOME_COLS = ["B365H", "BbAvH", "PSH", "WHH"]
ODD_DRAW_COLS = ["B365D", "BbAvD", "PSD", "WHD"]
ODD_AWAY_COLS = ["B365A", "BbAvA", "PSA", "WHA"]


# ── Chargement modèles ────────────────────────────────────────────────────────

def load_model():
    model_path = os.path.join(MODELS_DIR, "football_xgb_model.pkl")
    cols_path  = os.path.join(MODELS_DIR, "football_feature_cols.pkl")
    if not os.path.exists(model_path):
        logger.error(f"Modèle introuvable : {model_path}")
        logger.error("Lance d'abord : python train_from_csv.py")
        sys.exit(1)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(cols_path, "rb") as f:
        cols = pickle.load(f)
    logger.info(f"Modèle chargé : {model_path} ({len(cols)} features)")
    return model, cols


# ── Données ───────────────────────────────────────────────────────────────────

def download_csv(league_code: str, season: str) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), encoding="latin-1", on_bad_lines="skip")
        df = df[df["FTR"].isin(["H", "D", "A"])].copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df["season"] = season
        df["league"] = [k for k, v in LEAGUES.items() if v == league_code][0]
        return df
    except Exception as e:
        logger.warning(f"Échec {league_code} {season}: {e}")
        return None


def get_odd(row, cols):
    for c in cols:
        v = row.get(c)
        if v is not None and not pd.isna(v) and float(v) > 1.0:
            return float(v)
    return None


def load_all_data(seasons: list) -> pd.DataFrame:
    frames = []
    for league_name, code in LEAGUES.items():
        for season in seasons:
            df = download_csv(code, season)
            if df is not None:
                frames.append(df)
                logger.info(f"  {league_name} {season}: {len(df)} matchs")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


# ── Features rolling ──────────────────────────────────────────────────────────

def team_stats_before(all_df, team, match_date, n=FORM_WINDOW):
    past = all_df[
        ((all_df["HomeTeam"] == team) | (all_df["AwayTeam"] == team)) &
        (all_df["Date"] < match_date)
    ].sort_values("Date", ascending=False).head(n)

    if len(past) < MIN_MATCHES_BEFORE:
        return None

    gf_list, ga_list = [], []
    wins = draws = losses = cs = fts = 0
    for _, m in past.iterrows():
        is_home = m["HomeTeam"] == team
        gf = int(m["FTHG"]) if is_home else int(m["FTAG"])
        ga = int(m["FTAG"]) if is_home else int(m["FTHG"])
        gf_list.append(gf)
        ga_list.append(ga)
        if gf > ga:    wins   += 1
        elif gf == ga: draws  += 1
        else:          losses += 1
        if ga == 0: cs  += 1
        if gf == 0: fts += 1

    n_played = len(past)
    return {
        "win_rate":         round(wins / n_played, 4),
        "form_score":       round((3*wins + draws) / (3*n_played), 4),
        "goals_for_avg":    round(np.mean(gf_list), 4),
        "goals_ag_avg":     round(np.mean(ga_list), 4),
        "clean_sheet_rate": round(cs / n_played, 4),
        "scoring_rate":     round(1 - fts / n_played, 4),
    }


def build_features(row, all_df, feature_cols):
    home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
    hs = team_stats_before(all_df, home, date)
    as_ = team_stats_before(all_df, away, date)
    if hs is None or as_ is None:
        return None

    ha = hs["goals_for_avg"] / max(AVG_GOALS / 2, 0.1)
    hd = hs["goals_ag_avg"]  / max(AVG_GOALS / 2, 0.1)
    aa = as_["goals_for_avg"] / max(AVG_GOALS / 2, 0.1)
    ad = as_["goals_ag_avg"]  / max(AVG_GOALS / 2, 0.1)
    lam_h = max(ha * ad * (AVG_GOALS / 2), 0.1)
    lam_a = max(aa * hd * (AVG_GOALS / 2), 0.1)

    odd_h = get_odd(row, ODD_HOME_COLS)
    odd_d = get_odd(row, ODD_DRAW_COLS)
    odd_a = get_odd(row, ODD_AWAY_COLS)

    if odd_h and odd_d and odd_a:
        margin = 1/odd_h + 1/odd_d + 1/odd_a
        impl_h = round((1/odd_h) / margin, 4)
        impl_d = round((1/odd_d) / margin, 4)
        impl_a = round((1/odd_a) / margin, 4)
        has_odds = 1.0
    else:
        impl_h = impl_d = impl_a = 0.33
        odd_h = odd_d = odd_a = None
        has_odds = 0.0

    raw = {
        "home_form":             hs["form_score"],
        "away_form":             as_["form_score"],
        "form_diff":             hs["form_score"] - as_["form_score"],
        "home_goals_for_avg":    hs["goals_for_avg"],
        "home_goals_ag_avg":     hs["goals_ag_avg"],
        "away_goals_for_avg":    as_["goals_for_avg"],
        "away_goals_ag_avg":     as_["goals_ag_avg"],
        "home_attack_str":       round(ha, 4),
        "home_defense_str":      round(hd, 4),
        "away_attack_str":       round(aa, 4),
        "away_defense_str":      round(ad, 4),
        "home_win_rate":         hs["win_rate"],
        "away_win_rate":         as_["win_rate"],
        "win_rate_diff":         hs["win_rate"] - as_["win_rate"],
        "home_clean_sheet_rate": hs["clean_sheet_rate"],
        "away_clean_sheet_rate": as_["clean_sheet_rate"],
        "home_scoring_rate":     hs["scoring_rate"],
        "away_scoring_rate":     as_["scoring_rate"],
        "home_lambda":           round(lam_h, 4),
        "away_lambda":           round(lam_a, 4),
        "total_goals_exp":       round(lam_h + lam_a, 4),
        "h2h_home_win_rate":     0.33,
        "h2h_draw_rate":         0.33,
        "h2h_away_win_rate":     0.33,
        "h2h_avg_goals":         2.5,
        "h2h_matches":           0,
        "home_advantage":        1.0,
        "impl_home":             impl_h,
        "impl_draw":             impl_d,
        "impl_away":             impl_a,
        "odd_home":              odd_h or 2.5,
        "odd_draw":              odd_d or 3.3,
        "odd_away":              odd_a or 2.5,
        "has_odds":              has_odds,
    }

    # Aligner sur les colonnes du modèle
    return {c: raw.get(c, 0.0) for c in feature_cols}


# ── Value bet detection ───────────────────────────────────────────────────────

def detect_value(proba: dict, odds_row: dict) -> dict | None:
    """Retourne le meilleur value bet ou None."""
    candidates = [
        ("H", proba[0], odds_row["impl_h"], odds_row["odd_h"]),
        ("D", proba[1], odds_row["impl_d"], odds_row["odd_d"]),
        ("A", proba[2], odds_row["impl_a"], odds_row["odd_a"]),
    ]
    best = None
    for result, p_model, p_impl, odd in candidates:
        if odd is None or p_impl == 0:
            continue
        if not (MIN_ODD_ALLOWED <= odd <= MAX_ODD_ALLOWED):
            continue
        edge = p_model - p_impl
        if edge < VALUE_BET_EDGE or edge > MAX_EDGE_SANITY:
            continue
        if p_model < CONFIDENCE_THRESHOLD:
            continue
        ev = p_model * (odd - 1) - (1 - p_model)
        if ev < MIN_EV_REQUIRED:
            continue
        if best is None or edge > best["edge"]:
            best = {"result": result, "p_model": p_model, "odd": odd, "edge": edge, "ev": ev}
    return best


def kelly_amount(p, odd, bankroll):
    b = odd - 1
    q = 1 - p
    k = (b * p - q) / b
    if k <= 0:
        return 0
    pct = min(k * KELLY_FRACTION, MAX_BET_PCT)
    return round(bankroll * pct)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_backtest(all_df: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    bets = []
    bankroll = INITIAL_BANKROLL
    total = len(all_df)

    logger.info(f"\nSimulation sur {total} matchs...")

    for i, (_, row) in enumerate(all_df.iterrows()):
        if i % 1000 == 0 and i > 0:
            logger.info(f"  {i}/{total} — bankroll: {bankroll:,.0f} FCFA")

        feats = build_features(row, all_df, feature_cols)
        if feats is None:
            continue

        X = pd.DataFrame([feats])
        proba = model.predict_proba(X)[0]

        odd_h = get_odd(row, ODD_HOME_COLS)
        odd_d = get_odd(row, ODD_DRAW_COLS)
        odd_a = get_odd(row, ODD_AWAY_COLS)

        if not all([odd_h, odd_d, odd_a]):
            continue

        margin = 1/odd_h + 1/odd_d + 1/odd_a
        odds_row = {
            "odd_h":  odd_h, "odd_d":  odd_d, "odd_a":  odd_a,
            "impl_h": (1/odd_h) / margin,
            "impl_d": (1/odd_d) / margin,
            "impl_a": (1/odd_a) / margin,
        }

        vb = detect_value(proba, odds_row)
        if vb is None:
            continue

        # Mise Kelly
        stake = kelly_amount(vb["p_model"], vb["odd"], bankroll)
        if stake <= 0:
            continue

        # Résultat réel
        actual = row["FTR"]  # H, D, A
        won = (actual == vb["result"])
        pnl = round(stake * (vb["odd"] - 1)) if won else -stake
        bankroll = max(bankroll + pnl, 0)

        bets.append({
            "date":     row["Date"],
            "season":   row["season"],
            "league":   row["league"],
            "home":     row["HomeTeam"],
            "away":     row["AwayTeam"],
            "bet_on":   vb["result"],
            "odd":      vb["odd"],
            "edge":     round(vb["edge"], 4),
            "ev":       round(vb["ev"], 4),
            "p_model":  round(vb["p_model"], 4),
            "stake":    stake,
            "actual":   actual,
            "won":      won,
            "pnl":      pnl,
            "bankroll": bankroll,
        })

    return pd.DataFrame(bets)


# ── Métriques ─────────────────────────────────────────────────────────────────

def compute_metrics(bets: pd.DataFrame, initial_bankroll: float) -> dict:
    if bets.empty:
        return {}

    total_staked = bets["stake"].sum()
    total_pnl    = bets["pnl"].sum()
    wins         = bets["won"].sum()
    losses       = len(bets) - wins

    roi          = total_pnl / total_staked * 100 if total_staked > 0 else 0
    win_rate     = wins / len(bets) * 100

    # Max drawdown
    peak = initial_bankroll
    max_dd = 0
    for br in bets["bankroll"]:
        if br > peak:
            peak = br
        dd = (peak - br) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_wins   = bets[bets["pnl"] > 0]["pnl"].sum()
    gross_losses = abs(bets[bets["pnl"] < 0]["pnl"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Sharpe (daily returns)
    daily = bets.groupby(bets["date"].dt.date)["pnl"].sum()
    sharpe = daily.mean() / daily.std() * (252 ** 0.5) if daily.std() > 0 else 0

    final_bankroll = bets["bankroll"].iloc[-1] if not bets.empty else initial_bankroll

    return {
        "total_bets":       len(bets),
        "wins":             int(wins),
        "losses":           int(losses),
        "win_rate":         round(win_rate, 1),
        "total_staked":     round(total_staked),
        "total_pnl":        round(total_pnl),
        "roi":              round(roi, 2),
        "max_drawdown":     round(max_dd, 1),
        "profit_factor":    round(profit_factor, 2),
        "sharpe":           round(sharpe, 2),
        "initial_bankroll": initial_bankroll,
        "final_bankroll":   round(final_bankroll),
    }


def print_report(metrics: dict, bets: pd.DataFrame):
    sep = "═" * 55
    logger.info(f"\n{sep}")
    logger.info(f"  BACKTEST FOOTBALL — RÉSULTATS")
    logger.info(sep)
    logger.info(f"  Paris simulés     : {metrics['total_bets']}")
    logger.info(f"  Gagnés / Perdus   : {metrics['wins']} / {metrics['losses']}  ({metrics['win_rate']:.1f}%)")
    logger.info(f"  Total misé        : {metrics['total_staked']:,.0f} FCFA")
    logger.info(f"  P&L total         : {metrics['total_pnl']:+,.0f} FCFA")
    logger.info(f"  ROI               : {metrics['roi']:+.2f}%")
    logger.info(f"  Max drawdown      : -{metrics['max_drawdown']:.1f}%")
    logger.info(f"  Profit factor     : {metrics['profit_factor']:.2f}")
    logger.info(f"  Sharpe (annualisé): {metrics['sharpe']:.2f}")
    logger.info(f"  Bankroll init     : {metrics['initial_bankroll']:,.0f} FCFA")
    logger.info(f"  Bankroll finale   : {metrics['final_bankroll']:,.0f} FCFA")
    logger.info(sep)

    # Par ligue
    if not bets.empty:
        logger.info("\n  Par ligue :")
        by_league = bets.groupby("league").agg(
            bets_=("pnl", "count"),
            wins_=("won", "sum"),
            pnl_=("pnl", "sum"),
            staked_=("stake", "sum"),
        )
        for lg, row in by_league.iterrows():
            wr  = row["wins_"] / row["bets_"] * 100
            roi = row["pnl_"] / row["staked_"] * 100 if row["staked_"] > 0 else 0
            logger.info(
                f"  {lg:<18} {row['bets_']:3.0f} paris | "
                f"WR {wr:.0f}% | P&L {row['pnl_']:+,.0f} | ROI {roi:+.1f}%"
            )

        # Par saison
        logger.info("\n  Par saison :")
        by_season = bets.groupby("season").agg(
            bets_=("pnl", "count"),
            wins_=("won", "sum"),
            pnl_=("pnl", "sum"),
            staked_=("stake", "sum"),
        )
        season_labels = {"2425": "2024-25", "2324": "2023-24", "2223": "2022-23"}
        for s, row in by_season.iterrows():
            wr  = row["wins_"] / row["bets_"] * 100
            roi = row["pnl_"] / row["staked_"] * 100 if row["staked_"] > 0 else 0
            logger.info(
                f"  {season_labels.get(s, s):<10} {row['bets_']:3.0f} paris | "
                f"WR {wr:.0f}% | P&L {row['pnl_']:+,.0f} | ROI {roi:+.1f}%"
            )

        # Top 5 meilleurs paris
        logger.info("\n  Top 5 meilleurs paris :")
        for _, r in bets.nlargest(5, "pnl").iterrows():
            logger.info(
                f"  +{r['pnl']:,.0f} FCFA | {r['home']} vs {r['away']} "
                f"({r['league']}, {r['date'].strftime('%Y-%m-%d')}) | "
                f"Mis sur {r['bet_on']} @ {r['odd']:.2f}"
            )

        # Évolution bankroll (mini graphe ASCII)
        logger.info("\n  Évolution bankroll (début → fin) :")
        samples = bets["bankroll"].values
        step = max(1, len(samples) // 30)
        pts  = samples[::step]
        mx   = max(pts)
        mn   = min(pts)
        rng  = mx - mn if mx != mn else 1
        bars = [int((p - mn) / rng * 10) for p in pts]
        chars = " ▁▂▃▄▅▆▇█"
        line  = "  " + "".join(chars[min(b, 8)] for b in bars)
        logger.info(f"{line}")
        logger.info(f"  {mn:,.0f} ─────────────────────────────── {mx:,.0f} FCFA")

    logger.info(sep)

    # Interprétation
    roi = metrics["roi"]
    if roi > 5:
        logger.info("  ✅ Stratégie profitable — ROI > 5%")
    elif roi > 0:
        logger.info("  ⚠️  Stratégie légèrement profitable — surveiller sur plus de données")
    else:
        logger.info("  ❌ Stratégie non profitable sur cette période")
    logger.info(f"  ⚠️  Note : backtest IN-SAMPLE (modèle entraîné sur ces données)")
    logger.info(f"      → Performance réelle attendue légèrement inférieure")
    logger.info(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BetMind Football Backtest")
    parser.add_argument(
        "--seasons", nargs="+", default=ALL_SEASONS,
        help="Saisons à tester (ex: 2324 2425)"
    )
    parser.add_argument(
        "--save", default="data/backtest_results.csv",
        help="Fichier CSV de sortie des paris simulés"
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("data", exist_ok=True)

    logger.info("=" * 55)
    logger.info("  BETMIND — BACKTEST FOOTBALL")
    logger.info(f"  Saisons : {', '.join(args.seasons)}")
    logger.info(f"  Bankroll initiale : {INITIAL_BANKROLL:,.0f} FCFA")
    logger.info("=" * 55)

    # Chargement modèle
    model, feature_cols = load_model()

    # Données historiques
    logger.info("\nTéléchargement des données...")
    all_df = load_all_data(args.seasons)
    if all_df.empty:
        logger.error("Aucune donnée disponible.")
        sys.exit(1)
    logger.info(f"Total : {len(all_df)} matchs chargés")

    # Simulation
    bets = run_backtest(all_df, model, feature_cols)

    if bets.empty:
        logger.warning("Aucun value bet détecté sur cette période.")
        logger.warning("Vérifiez les seuils dans config.py (VALUE_BET_EDGE, CONFIDENCE_THRESHOLD).")
        sys.exit(0)

    logger.info(f"\n{len(bets)} value bets simulés.")

    # Métriques et rapport
    metrics = compute_metrics(bets, INITIAL_BANKROLL)
    print_report(metrics, bets)

    # Sauvegarde CSV
    bets.to_csv(args.save, index=False)
    logger.info(f"\nRésultats sauvegardés : {args.save}")
