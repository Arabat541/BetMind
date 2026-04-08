# ============================================================
# predictor.py — Pipeline principal
# ============================================================

import logging
import time
import schedule
from datetime import datetime

from config import FOOTBALL_LEAGUES, CONFIDENCE_THRESHOLD
from data_fetcher import (
    init_db, fetch_upcoming_football_fixtures,
    fetch_upcoming_nba_games, fetch_football_odds,
    fetch_nba_odds, save_prediction
)
from feature_engineering import build_football_features, build_nba_features
from model import BettingModel, build_prediction_signal
from bankroll import BankrollTracker, recommended_stake
from telegram_bot import send_prediction_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/predictor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

football_model = BettingModel(sport="football")
nba_model      = BettingModel(sport="nba")
tracker        = BankrollTracker()


# ════════════════════════════════════════════════════════════
# PIPELINE FOOTBALL
# ════════════════════════════════════════════════════════════

def run_football_predictions():
    logger.info("═══ FOOTBALL PREDICTION CYCLE ═══")

    fixtures = fetch_upcoming_football_fixtures(days_ahead=3)
    if fixtures.empty:
        logger.info("No upcoming football fixtures.")
        return []

    league_odds_map = {
        "Ligue 1":          fetch_football_odds("soccer_france_ligue_one"),
        "Premier League":   fetch_football_odds("soccer_epl"),
        "La Liga":          fetch_football_odds("soccer_spain_la_liga"),
        "Serie A":          fetch_football_odds("soccer_italy_serie_a"),
        "Bundesliga":       fetch_football_odds("soccer_germany_bundesliga"),
        "Champions League": fetch_football_odds("soccer_uefa_champs_league"),
    }

    signals  = []
    bankroll = tracker.get_balance()

    for _, fix in fixtures.iterrows():
        try:
            features = build_football_features(
                home_id=fix["home_id"],
                away_id=fix["away_id"],
                league_code=fix["league_code"],
                home_name=fix["home_team"],
                away_name=fix["away_team"],
            )
            if features is None:
                continue

            odds_df  = league_odds_map.get(fix["league"])
            odds_row = _match_odds(odds_df, fix["home_team"], fix["away_team"])

            signal = build_prediction_signal(
                features=features,
                odds_row=odds_row,
                model=football_model,
                sport="football",
                match_info={
                    "league":    fix["league"],
                    "home_team": fix["home_team"],
                    "away_team": fix["away_team"],
                    "date":      fix["date"],
                }
            )

            stake_info = recommended_stake(signal, bankroll)
            signal["kelly_stake"]    = stake_info["stake_amount"]
            signal["stake_pct"]      = stake_info["stake_pct"]
            signal["expected_value"] = stake_info.get("expected_value", 0)

            signals.append(signal)
            _log_signal(signal, stake_info)
            save_prediction({**signal, "sport": "football"})

            if signal["is_value_bet"] or signal["confidence"] >= CONFIDENCE_THRESHOLD:
                send_prediction_alert(signal, stake_info)

            time.sleep(0.2)

        except Exception as e:
            logger.error(f"Error processing {fix['home_team']} vs {fix['away_team']}: {e}")
            continue

    logger.info(f"Football cycle done: {len(signals)} signals generated.")
    return signals


# ════════════════════════════════════════════════════════════
# PIPELINE NBA
# ════════════════════════════════════════════════════════════

def run_nba_predictions():
    logger.info("═══ NBA PREDICTION CYCLE ═══")

    games = fetch_upcoming_nba_games(days_ahead=2)
    if games.empty:
        logger.info("No upcoming NBA games.")
        return []

    nba_odds_df = fetch_nba_odds()
    signals     = []
    bankroll    = tracker.get_balance()

    for _, game in games.iterrows():
        # Skip matchs déjà en cours ou terminés
        # Status à venir = timestamp ISO long (ex: "2026-04-07T00:00:00Z")
        # Status en cours = court (ex: "1st Qtr", "Final", "Halftime")
        status = str(game.get("status", ""))
        if len(status) < 15 and status.strip() != "":
            continue

        try:
            home_id = NBA_TEAM_IDS.get(game["home_abbr"])
            away_id = NBA_TEAM_IDS.get(game["away_abbr"])
            if not home_id or not away_id:
                logger.warning(f"NBA team ID not found: {game['home_abbr']} / {game['away_abbr']}")
                continue

            features = build_nba_features(
                home_id=home_id,
                away_id=away_id,
                home_name=game["home_team"],
                away_name=game["away_team"],
            )
            if features is None:
                continue

            odds_row = _match_odds(nba_odds_df, game["home_team"], game["away_team"])

            signal = build_prediction_signal(
                features=features,
                odds_row=odds_row,
                model=nba_model,
                sport="nba",
                match_info={
                    "league":    "NBA",
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "date":      game["date"],
                }
            )

            stake_info = recommended_stake(signal, bankroll)
            signal["kelly_stake"]    = stake_info["stake_amount"]
            signal["stake_pct"]      = stake_info["stake_pct"]
            signal["expected_value"] = stake_info.get("expected_value", 0)

            signals.append(signal)
            _log_signal(signal, stake_info)
            save_prediction({**signal, "sport": "nba"})

            if signal["is_value_bet"] or signal["confidence"] >= CONFIDENCE_THRESHOLD:
                send_prediction_alert(signal, stake_info)

            time.sleep(0.2)

        except Exception as e:
            logger.error(f"Error NBA {game['home_team']} vs {game['away_team']}: {e}")
            continue

    logger.info(f"NBA cycle done: {len(signals)} signals generated.")
    return signals


# ════════════════════════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════════════════════════

def _match_odds(odds_df, home_team: str, away_team: str) -> dict:
    empty = {
        "odd_home": None, "odd_draw": None, "odd_away": None,
        "impl_home": 0.33, "impl_draw": 0.33, "impl_away": 0.33
    }
    if odds_df is None or odds_df.empty:
        return empty

    mask = (
        odds_df["home_team"].str.contains(home_team.split()[0], case=False, na=False) &
        odds_df["away_team"].str.contains(away_team.split()[0], case=False, na=False)
    )
    matches = odds_df[mask]
    if matches.empty:
        return empty

    row = matches.iloc[0]
    return {
        "odd_home":  row.get("odd_home"),
        "odd_draw":  row.get("odd_draw"),
        "odd_away":  row.get("odd_away"),
        "impl_home": row.get("impl_home", 0.33),
        "impl_draw": row.get("impl_draw", 0.33),
        "impl_away": row.get("impl_away", 0.33),
    }


def _log_signal(signal: dict, stake_info: dict):
    tag = "🔥 VALUE BET" if signal["is_value_bet"] else "📊 SIGNAL"
    logger.info(
        f"{tag} | {signal['home_team']} vs {signal['away_team']} "
        f"({signal['league']}) | "
        f"Prédiction: {signal['pred_name']} ({signal['confidence']:.1%}) | "
        f"Mise: {stake_info['stake_amount']:,.0f} FCFA | "
        f"EV: {stake_info.get('expected_value', 0):+.3f}"
    )


def run_all():
    logger.info(f"\n{'═'*60}")
    logger.info(f"  SPORTS BETTING AGENT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"  Bankroll: {tracker.get_balance():,.0f} FCFA")
    logger.info(f"{'═'*60}")

    football_signals = run_football_predictions()
    nba_signals      = run_nba_predictions()

    total = len(football_signals) + len(nba_signals)
    value = sum(1 for s in football_signals + nba_signals if s["is_value_bet"])
    logger.info(f"\nCycle terminé: {total} signaux | {value} value bets détectés")
    return football_signals + nba_signals


# ════════════════════════════════════════════════════════════
# SCHEDULER
# ════════════════════════════════════════════════════════════

def start_scheduler():
    logger.info("Scheduler démarré.")
    schedule.every().day.at("08:00").do(run_with_results)
    schedule.every().day.at("18:00").do(run_with_results)
    schedule.every().day.at("22:00").do(run_nba_predictions)
    while True:
        schedule.run_pending()
        time.sleep(60)


# ════════════════════════════════════════════════════════════
# TABLE NBA TEAM IDS (BallDontLie)
# ════════════════════════════════════════════════════════════

NBA_TEAM_IDS = {
    "ATL": 1,  "BOS": 2,  "BKN": 3,  "CHA": 4,  "CHI": 5,
    "CLE": 6,  "DAL": 7,  "DEN": 8,  "DET": 9,  "GSW": 10,
    "HOU": 11, "IND": 12, "LAC": 13, "LAL": 14, "MEM": 15,
    "MIA": 16, "MIL": 17, "MIN": 18, "NOP": 19, "NYK": 20,
    "OKC": 21, "ORL": 22, "PHI": 23, "PHX": 24, "POR": 25,
    "SAC": 26, "SAS": 27, "TOR": 28, "UTA": 29, "WAS": 30,
}


if __name__ == "__main__":
    import os, sys
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    init_db()
    if "--schedule" in sys.argv:
        start_scheduler()
    else:
        run_all()


# ════════════════════════════════════════════════════════════
# INTÉGRATION RESULT CHECKER
# ════════════════════════════════════════════════════════════

def run_with_results():
    """Cycle complet : prédictions + vérification résultats."""
    from result_checker import run_result_checker
    # 1. Vérifie les résultats des matchs passés
    run_result_checker()
    # 2. Génère les nouvelles prédictions
    return run_all()