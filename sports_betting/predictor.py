# ============================================================
# predictor.py — Pipeline principal
# ============================================================

import logging
import os
import shutil
import subprocess
import sys
import time
import schedule
from datetime import datetime, timedelta

from config import FOOTBALL_LEAGUES, CONFIDENCE_THRESHOLD, LOW_BANKROLL_THRESHOLD, ALERT_MIN_EDGE, FD_DAILY_LIMIT, MAX_DAILY_STAKE_PCT, DAILY_STOP_LOSS_PCT, ODDS_MOVEMENT_THRESHOLD, MAX_ACTIVE_BETS_PER_LEAGUE, MAX_TEAM_EXPOSURE_PCT
from db import get_conn, ph as _pred_ph
from data_fetcher import (
    init_db, fetch_upcoming_football_fixtures,
    fetch_upcoming_nba_games, fetch_football_odds, fetch_football_ou_odds,
    fetch_football_btts_odds, fetch_football_ah_odds,
    fetch_nba_odds, fetch_nba_injuries, save_prediction, is_already_predicted,
    get_fd_quota_used, count_active_bets_for_league, get_team_exposure,
)
from feature_engineering import build_football_features, build_nba_features
from model import BettingModel, build_prediction_signal, build_ou_signal, build_btts_signal, detect_ah_value_bet, detect_implied_value, detect_arbitrage, build_correct_score_signal, fetch_correct_score_odds, detect_rlm, fetch_opening_odds
from bankroll import BankrollTracker, recommended_stake
from telegram_bot import send_prediction_alert, send_bankroll_alert, send_weekly_summary, send_message, send_stop_loss_alert, send_odds_movement_alert, send_correct_score_alert, send_rlm_alert

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
ou_model       = BettingModel(sport="ou_football")
btts_model     = BettingModel(sport="btts_football")
nba_model      = BettingModel(sport="nba")
tracker        = BankrollTracker()


# ════════════════════════════════════════════════════════════
# PIPELINE FOOTBALL
# ════════════════════════════════════════════════════════════

def run_football_predictions():
    logger.info("═══ FOOTBALL PREDICTION CYCLE ═══")

    fixtures = fetch_upcoming_football_fixtures(days_ahead=3)
    if fixtures.empty:
        used = get_fd_quota_used()
        if used >= FD_DAILY_LIMIT:
            logger.warning(f"Quota football-data.org épuisé ({used}/{FD_DAILY_LIMIT}).")
            send_message(
                f"⚠️ <b>BetMind — Quota épuisé</b>\n\n"
                f"football-data.org : <b>{used}/{FD_DAILY_LIMIT}</b> requêtes utilisées.\n"
                "Aucune prédiction football possible aujourd'hui.\n"
                "Le quota se réinitialise à minuit UTC."
            )
        else:
            logger.info("No upcoming football fixtures.")
        return []

    _LEAGUE_KEYS = {
        "Ligue 1":          "soccer_france_ligue_one",
        "Premier League":   "soccer_epl",
        "La Liga":          "soccer_spain_la_liga",
        "Serie A":          "soccer_italy_serie_a",
        "Bundesliga":       "soccer_germany_bundesliga",
        "Champions League": "soccer_uefa_champs_league",
    }
    league_odds_map      = {lg: fetch_football_odds(key)      for lg, key in _LEAGUE_KEYS.items()}
    league_ou_odds_map   = {lg: fetch_football_ou_odds(key)   for lg, key in _LEAGUE_KEYS.items()}
    league_btts_odds_map = {lg: fetch_football_btts_odds(key) for lg, key in _LEAGUE_KEYS.items()}
    league_ah_odds_map   = {lg: fetch_football_ah_odds(key)   for lg, key in _LEAGUE_KEYS.items()}
    league_cs_odds_map   = {lg: fetch_correct_score_odds(key) for lg, key in _LEAGUE_KEYS.items()}
    league_open_odds_map = {lg: fetch_opening_odds(key)       for lg, key in _LEAGUE_KEYS.items()}

    signals       = []
    bankroll      = tracker.get_balance()
    daily_limit   = bankroll * MAX_DAILY_STAKE_PCT
    daily_staked  = tracker.get_today_staked()

    for _, fix in fixtures.iterrows():
        try:
            if is_already_predicted(fix["home_team"], fix["away_team"], fix["date"]):
                logger.info(f"Skip (déjà prédit): {fix['home_team']} vs {fix['away_team']}")
                continue

            active = count_active_bets_for_league(fix["league"])
            if active >= MAX_ACTIVE_BETS_PER_LEAGUE:
                logger.info(
                    f"Diversification: {fix['league']} déjà {active} paris actifs "
                    f"(max {MAX_ACTIVE_BETS_PER_LEAGUE}). Skip {fix['home_team']} vs {fix['away_team']}."
                )
                continue

            # ── Stop-loss dynamique par ligue — O ────────────
            suspended, susp_reason = tracker.is_league_suspended(fix["league"])
            if suspended:
                logger.warning(f"⛔ Ligue suspendue — {susp_reason}. Skip {fix['home_team']} vs {fix['away_team']}.")
                continue

            remaining = daily_limit - daily_staked
            if remaining <= 0:
                logger.info(f"Limite journalière atteinte ({daily_staked:,.0f}/{daily_limit:,.0f} FCFA). Arrêt football.")
                break

            features = build_football_features(
                home_id=fix["home_id"],
                away_id=fix["away_id"],
                league_code=fix["league_code"],
                home_name=fix["home_team"],
                away_name=fix["away_team"],
                league_name=fix["league"],
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
            if stake_info["stake_amount"] > remaining:
                logger.info(
                    f"Mise plafonnée: {stake_info['stake_amount']:,.0f} → {remaining:,.0f} FCFA (limite journalière)"
                )
                stake_info["stake_amount"] = round(remaining)
                stake_info["stake_pct"]    = round(remaining / bankroll, 4) if bankroll > 0 else 0

            signal["kelly_stake"]    = stake_info["stake_amount"]
            signal["stake_pct"]      = stake_info["stake_pct"]
            signal["expected_value"] = stake_info.get("expected_value", 0)

            # ── Exposition nette par équipe ───────────────────
            if signal["pred_result"] in ("H", "A"):
                w_team   = fix["home_team"] if signal["pred_result"] == "H" else fix["away_team"]
                team_exp = get_team_exposure(w_team)
                if team_exp + signal["kelly_stake"] > bankroll * MAX_TEAM_EXPOSURE_PCT:
                    logger.info(
                        f"Exposition {w_team}: {team_exp:,.0f}+{signal['kelly_stake']:,.0f} FCFA"
                        f" > {bankroll * MAX_TEAM_EXPOSURE_PCT:,.0f} FCFA"
                        f" (max {MAX_TEAM_EXPOSURE_PCT:.0%}). Skip."
                    )
                    continue

            daily_staked += stake_info["stake_amount"]
            signals.append(signal)
            _log_signal(signal, stake_info)
            save_prediction({**signal, "sport": "football"})

            if _should_alert(signal):
                send_prediction_alert(signal, stake_info)

            # ── AM — Reverse Line Movement ────────────────────────
            open_odds_all = league_open_odds_map.get(fix["league"], {})
            if open_odds_all and odds_row:
                open_odds_event = {}
                for event_key, oo in open_odds_all.items():
                    h_part = event_key.split("|")[0] if "|" in event_key else ""
                    if fix["home_team"].split()[0].lower() in h_part.lower():
                        open_odds_event = oo
                        break
                if open_odds_event:
                    curr_odds = {
                        "H": float(odds_row.get("odd_home") or 0) or None,
                        "D": float(odds_row.get("odd_draw") or 0) or None,
                        "A": float(odds_row.get("odd_away") or 0) or None,
                    }
                    rlm_signals = detect_rlm(open_odds_event, curr_odds)
                    for rlm in rlm_signals:
                        outcome_names = {"H": "Domicile", "D": "Nul", "A": "Extérieur"}
                        logger.info(
                            f"  [RLM] {fix['home_team']} vs {fix['away_team']} — "
                            f"{outcome_names[rlm['outcome']]} cote "
                            f"{rlm['odd_open']} → {rlm['odd_current']} "
                            f"({rlm['line_move_pct']:+.1%}) score={rlm['rlm_score']}"
                        )
                        # Enrich le signal principal avec le flag RLM si l'outcome correspond
                        if rlm["outcome"] == signal.get("pred_result"):
                            signal["rlm_detected"]  = True
                            signal["rlm_score"]     = rlm["rlm_score"]
                            signal["rlm_line_move"] = rlm["line_move_pct"]
                        # Alerte Telegram dédiée pour les RLM forts (score ≥ 0.6)
                        if rlm["rlm_score"] >= 0.60:
                            send_rlm_alert(
                                home_team=fix["home_team"],
                                away_team=fix["away_team"],
                                league=fix["league"],
                                outcome=outcome_names[rlm["outcome"]],
                                odd_open=rlm["odd_open"],
                                odd_current=rlm["odd_current"],
                                line_move_pct=rlm["line_move_pct"],
                                public_pct=rlm["public_pct"],
                            )

            # ── Arbitrage / implied value multi-bookmakers — M ─
            try:
                bk_odds = odds_row if isinstance(odds_row, dict) else {}
                if bk_odds:
                    # The Odds API retourne parfois plusieurs bookmakers dans odds_row
                    # On tente detect_implied_value sur les cotes disponibles
                    model_proba = {
                        "H": signal.get("prob_home", 0),
                        "D": signal.get("prob_draw", 0),
                        "A": signal.get("prob_away", 0),
                    }
                    # Construit un dict multi-bk à partir des champs odd_H/odd_D/odd_A
                    bk_dict = {"bookmaker_1": {
                        "H": float(bk_odds.get("odd_H") or bk_odds.get("home_odd") or 0),
                        "D": float(bk_odds.get("odd_D") or bk_odds.get("draw_odd") or 0),
                        "A": float(bk_odds.get("odd_A") or bk_odds.get("away_odd") or 0),
                    }}
                    implied_vbs = detect_implied_value(model_proba, bk_dict, min_edge=0.03)
                    if implied_vbs:
                        logger.info(
                            f"  [Multi-BK] {len(implied_vbs)} value(s) détectée(s) : "
                            + ", ".join(f"{v['outcome']} edge={v['edge']:.1%} @ {v['best_odd']}" for v in implied_vbs)
                        )
            except Exception:
                pass

            # ── Signal Asian Handicap -0.5 — K ───────────────
            if football_model.is_trained and daily_staked < daily_limit and signal:
                ah_odds_df  = league_ah_odds_map.get(fix["league"])
                ah_odds_row = _match_ou_odds(ah_odds_df, fix["home_team"], fix["away_team"]) \
                              if ah_odds_df is not None else {}
                ah_vb = detect_ah_value_bet(
                    prob_home=signal.get("prob_home", 0),
                    prob_away=signal.get("prob_away", 0),
                    prob_draw=signal.get("prob_draw", 0),
                    ah_odds_row=ah_odds_row,
                )
                if ah_vb:
                    remaining  = daily_limit - daily_staked
                    ah_sig     = {
                        **{k: signal[k] for k in ("sport", "league", "home_team", "away_team", "match_date")},
                        "pred_result":  ah_vb["result"],
                        "pred_name":    ah_vb["result_name"],
                        "prob_home":    signal.get("prob_home", 0),
                        "prob_draw":    0.0,
                        "prob_away":    signal.get("prob_away", 0),
                        "confidence":   ah_vb["p_model"],
                        "method":       "asian_handicap",
                        "value_bets":   [ah_vb],
                        "is_value_bet": True,
                        "edge":         ah_vb["edge"],
                        "odd_used":     ah_vb["odd"],
                        "market":       "AH-0.5",
                    }
                    ah_stake = recommended_stake(ah_sig, bankroll)
                    if ah_stake["stake_amount"] > remaining:
                        ah_stake["stake_amount"] = round(remaining)
                        ah_stake["stake_pct"] = round(remaining / bankroll, 4) if bankroll > 0 else 0
                    ah_sig["kelly_stake"]    = ah_stake["stake_amount"]
                    ah_sig["stake_pct"]      = ah_stake["stake_pct"]
                    ah_sig["expected_value"] = ah_stake.get("expected_value", 0)
                    daily_staked += ah_stake["stake_amount"]
                    signals.append(ah_sig)
                    _log_signal(ah_sig, ah_stake)
                    save_prediction(ah_sig)
                    if _should_alert(ah_sig):
                        send_prediction_alert(ah_sig, ah_stake)

            # ── Générer OU et BTTS (avant corrélation) — Z ───────
            ou_sig   = None
            btts_sig = None

            if ou_model.is_trained and daily_staked < daily_limit:
                ou_odds_df  = league_ou_odds_map.get(fix["league"])
                ou_odds_row = _match_ou_odds(ou_odds_df, fix["home_team"], fix["away_team"])
                ou_sig = build_ou_signal(features, ou_odds_row, ou_model, {
                    "league":    fix["league"],
                    "home_team": fix["home_team"],
                    "away_team": fix["away_team"],
                    "date":      fix["date"],
                })

            if btts_model.is_trained and daily_staked < daily_limit:
                btts_odds_df  = league_btts_odds_map.get(fix["league"])
                btts_odds_row = _match_ou_odds(btts_odds_df, fix["home_team"], fix["away_team"]) \
                                if btts_odds_df is not None else {}
                btts_sig = build_btts_signal(features, btts_odds_row, btts_model, {
                    "league":    fix["league"],
                    "home_team": fix["home_team"],
                    "away_team": fix["away_team"],
                    "date":      fix["date"],
                })

            # ── Z — Corrélation entre marchés ────────────────────
            if ou_sig and btts_sig:
                ou_is_over  = ou_sig["pred_result"] == "O"
                btts_is_yes = btts_sig["pred_result"] == "BTTS"
                btts_is_no  = btts_sig["pred_result"] == "No BTTS"
                if ou_is_over and btts_is_no:
                    # Signaux contradictoires : Over 2.5 mais BTTS non → ignorer les deux
                    logger.info(
                        f"Corrélation OU/BTTS contradictoire — "
                        f"{fix['home_team']} vs {fix['away_team']}: "
                        f"Over 2.5 + No BTTS. Les deux signaux ignorés."
                    )
                    ou_sig   = None
                    btts_sig = None
                elif ou_is_over and btts_is_yes:
                    # Signaux alignés : Over 2.5 + BTTS Yes → boost mise +20%
                    logger.info(
                        f"Corrélation OU/BTTS confirmée — "
                        f"{fix['home_team']} vs {fix['away_team']}: "
                        f"Over 2.5 + BTTS Yes. Mise +20%."
                    )
                    ou_sig["correlated_boost"]   = True
                    btts_sig["correlated_boost"] = True

            # ── Persister OU et BTTS ──────────────────────────────
            for mkt_sig in (ou_sig, btts_sig):
                if mkt_sig is None:
                    continue
                remaining_mkt = daily_limit - daily_staked
                mkt_stake = recommended_stake(mkt_sig, bankroll)
                if mkt_sig.get("correlated_boost"):
                    mkt_stake["stake_amount"] = round(mkt_stake["stake_amount"] * 1.20)
                    mkt_stake["stake_pct"]    = round(mkt_stake["stake_amount"] / bankroll, 4) if bankroll > 0 else 0
                if mkt_stake["stake_amount"] > remaining_mkt:
                    mkt_stake["stake_amount"] = round(remaining_mkt)
                    mkt_stake["stake_pct"]    = round(remaining_mkt / bankroll, 4) if bankroll > 0 else 0
                mkt_sig["kelly_stake"]    = mkt_stake["stake_amount"]
                mkt_sig["stake_pct"]      = mkt_stake["stake_pct"]
                mkt_sig["expected_value"] = mkt_stake.get("expected_value", 0)
                daily_staked += mkt_stake["stake_amount"]
                signals.append(mkt_sig)
                _log_signal(mkt_sig, mkt_stake)
                save_prediction(mkt_sig)
                if _should_alert(mkt_sig):
                    send_prediction_alert(mkt_sig, mkt_stake)

            # ── AL — Correct Score (Dixon-Coles bivarié) ─────────
            cs_odds_all = league_cs_odds_map.get(fix["league"], {})
            if cs_odds_all and daily_staked < daily_limit:
                # Cherche l'événement correspondant dans le dict {home|away: {...}}
                cs_odds_event = {}
                for event_key, odds in cs_odds_all.items():
                    h_part = event_key.split("|")[0] if "|" in event_key else ""
                    if fix["home_team"].split()[0].lower() in h_part.lower():
                        cs_odds_event = odds
                        break
                if cs_odds_event:
                    cs_sig = build_correct_score_signal(
                        features=features,
                        cs_odds=cs_odds_event,
                        match_info={
                            "league":    fix["league"],
                            "home_team": fix["home_team"],
                            "away_team": fix["away_team"],
                            "date":      fix["date"],
                        }
                    )
                    if cs_sig:
                        remaining_cs = daily_limit - daily_staked
                        cs_stake = recommended_stake(cs_sig, bankroll)
                        if cs_stake["stake_amount"] > remaining_cs:
                            cs_stake["stake_amount"] = round(remaining_cs)
                            cs_stake["stake_pct"] = round(remaining_cs / bankroll, 4) if bankroll > 0 else 0
                        cs_sig["kelly_stake"]    = cs_stake["stake_amount"]
                        cs_sig["stake_pct"]      = cs_stake["stake_pct"]
                        cs_sig["expected_value"] = cs_stake.get("expected_value", 0)
                        daily_staked += cs_stake["stake_amount"]
                        signals.append(cs_sig)
                        _log_signal(cs_sig, cs_stake)
                        save_prediction(cs_sig)
                        send_correct_score_alert(cs_sig, cs_stake)
                        logger.info(
                            f"  [CS] Value bet: {fix['home_team']} vs {fix['away_team']} "
                            f"score {cs_sig['value_bets'][0]['score']} "
                            f"edge={cs_sig['edge']:.1%} @ {cs_sig['odd_used']:.2f}"
                        )

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

    nba_odds_df   = fetch_nba_odds()
    nba_injuries  = fetch_nba_injuries()   # une requête ESPN pour tous les matchs
    signals       = []
    bankroll      = tracker.get_balance()
    daily_limit   = bankroll * MAX_DAILY_STAKE_PCT
    daily_staked  = tracker.get_today_staked()

    for _, game in games.iterrows():
        # Skip matchs déjà en cours ou terminés
        # Status à venir = timestamp ISO long (ex: "2026-04-07T00:00:00Z")
        # Status en cours = court (ex: "1st Qtr", "Final", "Halftime")
        status = str(game.get("status", ""))
        if len(status) < 15 and status.strip() != "":
            continue

        try:
            if is_already_predicted(game["home_team"], game["away_team"], game["date"]):
                logger.info(f"Skip (déjà prédit): {game['home_team']} vs {game['away_team']}")
                continue

            active_nba = count_active_bets_for_league("NBA")
            if active_nba >= MAX_ACTIVE_BETS_PER_LEAGUE:
                logger.info(
                    f"Diversification: NBA déjà {active_nba} paris actifs "
                    f"(max {MAX_ACTIVE_BETS_PER_LEAGUE}). Skip {game['home_team']} vs {game['away_team']}."
                )
                continue

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
                injuries_dict=nba_injuries,
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

            remaining = daily_limit - daily_staked
            if remaining <= 0:
                logger.info(f"Limite journalière atteinte ({daily_staked:,.0f}/{daily_limit:,.0f} FCFA). Arrêt NBA.")
                break

            stake_info = recommended_stake(signal, bankroll)
            if stake_info["stake_amount"] > remaining:
                logger.info(
                    f"Mise plafonnée: {stake_info['stake_amount']:,.0f} → {remaining:,.0f} FCFA (limite journalière)"
                )
                stake_info["stake_amount"] = round(remaining)
                stake_info["stake_pct"]    = round(remaining / bankroll, 4) if bankroll > 0 else 0

            signal["kelly_stake"]    = stake_info["stake_amount"]
            signal["stake_pct"]      = stake_info["stake_pct"]
            signal["expected_value"] = stake_info.get("expected_value", 0)

            # ── Exposition nette par équipe ───────────────────
            if signal["pred_result"] in ("H", "A"):
                w_team   = game["home_team"] if signal["pred_result"] == "H" else game["away_team"]
                team_exp = get_team_exposure(w_team)
                if team_exp + signal["kelly_stake"] > bankroll * MAX_TEAM_EXPOSURE_PCT:
                    logger.info(
                        f"Exposition {w_team}: {team_exp:,.0f}+{signal['kelly_stake']:,.0f} FCFA"
                        f" > {bankroll * MAX_TEAM_EXPOSURE_PCT:,.0f} FCFA. Skip."
                    )
                    continue

            daily_staked += stake_info["stake_amount"]

            # Propager les données de blessures dans le signal (affichage Telegram)
            for key in ("home_injuries_out", "home_injuries_dtd",
                        "away_injuries_out", "away_injuries_dtd", "injury_diff"):
                signal[key] = features.get(key, 0)

            signals.append(signal)
            _log_signal(signal, stake_info)
            save_prediction({**signal, "sport": "nba"})

            if _should_alert(signal):
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


def _match_ou_odds(ou_odds_df, home_team: str, away_team: str) -> dict:
    empty = {"odd_over": None, "odd_under": None, "impl_over": 0.5, "impl_under": 0.5}
    if ou_odds_df is None or ou_odds_df.empty:
        return empty
    mask = (
        ou_odds_df["home_team"].str.contains(home_team.split()[0], case=False, na=False) &
        ou_odds_df["away_team"].str.contains(away_team.split()[0], case=False, na=False)
    )
    matches = ou_odds_df[mask]
    if matches.empty:
        return empty
    row = matches.iloc[0]
    return {
        "odd_over":  row.get("odd_over"),
        "odd_under": row.get("odd_under"),
        "impl_over": row.get("impl_over", 0.5),
        "impl_under": row.get("impl_under", 0.5),
    }


def _should_alert(signal: dict) -> bool:
    """Envoie une alerte seulement si le signal est suffisamment fort.
    Priorité : value bet avec edge >= 7% ET mise Kelly > 0.
    Évite le bruit des signaux marginaux.
    """
    if not signal["is_value_bet"]:
        return False
    return signal.get("edge", 0) >= ALERT_MIN_EDGE and signal.get("kelly_stake", 0) > 0


def _log_signal(signal: dict, stake_info: dict):
    tag = "🔥 VALUE BET" if signal["is_value_bet"] else "📊 SIGNAL"
    logger.info(
        f"{tag} | {signal['home_team']} vs {signal['away_team']} "
        f"({signal['league']}) | "
        f"Prédiction: {signal['pred_name']} ({signal['confidence']:.1%}) | "
        f"Mise: {stake_info['stake_amount']:,.0f} FCFA | "
        f"EV: {stake_info.get('expected_value', 0):+.3f}"
    )


_LEAGUE_KEYS_OM = {
    "Ligue 1":          "soccer_france_ligue_one",
    "Premier League":   "soccer_epl",
    "La Liga":          "soccer_spain_la_liga",
    "Serie A":          "soccer_italy_serie_a",
    "Bundesliga":       "soccer_germany_bundesliga",
    "Champions League": "soccer_uefa_champs_league",
}


def check_odds_movement():
    """
    Pour les paris en attente générés dans les dernières 24h,
    vérifie si la cote a baissé de > ODDS_MOVEMENT_THRESHOLD depuis la prédiction.
    Si oui : annule la mise (kelly_stake = 0) + alerte Telegram.
    Signal de "sharp money" : le marché s'est retourné contre notre prédiction.
    """
    cutoff_24h = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as _om_conn:
        rows = _om_conn.execute(f"""
            SELECT id, sport, league, home_team, away_team, pred_result, odd_used, kelly_stake
            FROM predictions
            WHERE outcome IS NULL
              AND kelly_stake > 0
              AND created_at >= {_pred_ph}
        """, (cutoff_24h,)).fetchall()

    if not rows:
        return

    cols   = ("id", "sport", "league", "home_team", "away_team",
              "pred_result", "odd_used", "kelly_stake")
    recent = [dict(zip(cols, r)) for r in rows]
    logger.info(f"Odds movement check : {len(recent)} paris en attente à vérifier")

    # Cache pour éviter de re-fetcher la même ligue plusieurs fois
    cache_1x2: dict = {}
    cache_ou:  dict = {}

    for pred in recent:
        sport    = pred["sport"]
        league   = pred["league"]
        pred_r   = pred["pred_result"]
        saved    = float(pred["odd_used"] or 0)

        if saved <= 0:
            continue

        is_ou = pred_r in ("O", "U")

        try:
            if sport == "football":
                lg_key = _LEAGUE_KEYS_OM.get(league)
                if not lg_key:
                    continue
                if is_ou:
                    if lg_key not in cache_ou:
                        cache_ou[lg_key] = fetch_football_ou_odds(lg_key)
                    odds_row   = _match_ou_odds(cache_ou[lg_key], pred["home_team"], pred["away_team"])
                    current    = odds_row.get("odd_over") if pred_r == "O" else odds_row.get("odd_under")
                else:
                    if lg_key not in cache_1x2:
                        cache_1x2[lg_key] = fetch_football_odds(lg_key)
                    odds_row   = _match_odds(cache_1x2[lg_key], pred["home_team"], pred["away_team"])
                    current    = (
                        odds_row.get("odd_home") if pred_r == "H"
                        else odds_row.get("odd_draw") if pred_r == "D"
                        else odds_row.get("odd_away")
                    )
            elif sport == "nba":
                if "nba" not in cache_1x2:
                    cache_1x2["nba"] = fetch_nba_odds()
                odds_row = _match_odds(cache_1x2["nba"], pred["home_team"], pred["away_team"])
                current  = odds_row.get("odd_home") if pred_r == "H" else odds_row.get("odd_away")
            else:
                continue

            if not current:
                continue

            current_f = float(current)
            # movement_pct > 0 = odd dropped (market moved toward our pick = sharps confirm)
            # movement_pct < 0 = odd rose (market moved against our pick = bad sign)
            movement_pct = round((saved - current_f) / saved, 4)
            drop_pct     = -movement_pct if movement_pct < 0 else movement_pct

            try:
                with get_conn() as _upd_conn:
                    if movement_pct < -ODDS_MOVEMENT_THRESHOLD:
                        _upd_conn.execute(
                            f"UPDATE predictions SET odd_closing={_pred_ph}, kelly_stake=0, opening_movement_pct={_pred_ph} WHERE id={_pred_ph}",
                            (current_f, movement_pct, int(pred["id"]))
                        )
                    else:
                        _upd_conn.execute(
                            f"UPDATE predictions SET odd_closing={_pred_ph}, opening_movement_pct={_pred_ph} WHERE id={_pred_ph}",
                            (current_f, movement_pct, int(pred["id"]))
                        )
            except Exception as db_err:
                logger.error(f"check_odds_movement DB error: {db_err}")

            if movement_pct < -ODDS_MOVEMENT_THRESHOLD:
                # Marché contre nous : annuler
                logger.warning(
                    f"Cote bougée CONTRE — {pred['home_team']} vs {pred['away_team']} "
                    f"({pred_r}): {saved:.2f} → {current_f:.2f} ({movement_pct:+.1%}). Mise annulée."
                )
                send_odds_movement_alert(
                    home_team=pred["home_team"], away_team=pred["away_team"],
                    league=league, pred_result=pred_r,
                    odd_saved=saved, odd_current=current_f, drop_pct=abs(movement_pct),
                )
            elif movement_pct > 0.05:
                # Sharps confirment notre pari (cote a baissé > 5%)
                logger.info(
                    f"Sharp money ✓ — {pred['home_team']} vs {pred['away_team']} "
                    f"({pred_r}): {saved:.2f} → {current_f:.2f} ({movement_pct:+.1%})"
                )
        except Exception as e:
            logger.error(f"Odds movement check error ({pred['home_team']} vs {pred['away_team']}): {e}")


def _check_daily_stop_loss(bankroll: float) -> tuple[bool, float]:
    """
    Vérifie si le stop-loss journalier doit bloquer les nouvelles mises.
    Retourne (déclenché, pnl_récent).
    Basé sur le P&L des paris réglés sur les 2 derniers jours de matchs.
    """
    recent_pnl = tracker.get_recent_settled_pnl(days=2)
    threshold  = -(bankroll * DAILY_STOP_LOSS_PCT)
    return recent_pnl < threshold, recent_pnl


def run_all():
    logger.info(f"\n{'═'*60}")
    logger.info(f"  SPORTS BETTING AGENT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    bankroll = tracker.get_balance()
    logger.info(f"  Bankroll: {bankroll:,.0f} FCFA")
    logger.info(f"{'═'*60}")

    # Alerte si bankroll critique
    if bankroll < LOW_BANKROLL_THRESHOLD:
        logger.warning(f"Bankroll critique: {bankroll:,.0f} FCFA < {LOW_BANKROLL_THRESHOLD:,.0f} FCFA")
        send_bankroll_alert(bankroll, LOW_BANKROLL_THRESHOLD)

    # ── AA — Drawdown global : pause complète si > 20% depuis le pic ─
    dd_exceeded, dd_pct = tracker.is_global_drawdown_exceeded(max_drawdown_pct=20.0)
    if dd_exceeded:
        peak = tracker.get_peak_bankroll()
        logger.warning(
            f"Drawdown global déclenché : {dd_pct:.1f}% depuis pic "
            f"({peak:,.0f} FCFA → {bankroll:,.0f} FCFA). Aucune mise ce cycle."
        )
        send_message(
            f"🛑 <b>BetMind — Drawdown global</b>\n\n"
            f"Drawdown : <b>{dd_pct:.1f}%</b> depuis le pic ({peak:,.0f} FCFA)\n"
            f"Bankroll actuelle : <b>{bankroll:,.0f} FCFA</b>\n\n"
            f"⛔ Aucune nouvelle mise jusqu'à reprise du capital.\n"
            f"🤖 BetMind Agent"
        )
        return []

    # Stop-loss journalier : P&L récent < -5% bankroll → aucune nouvelle mise
    stop_triggered, recent_pnl = _check_daily_stop_loss(bankroll)
    if stop_triggered:
        logger.warning(
            f"Stop-loss journalier déclenché : P&L récent = {recent_pnl:+,.0f} FCFA "
            f"(seuil = -{DAILY_STOP_LOSS_PCT:.0%} × {bankroll:,.0f} FCFA)"
        )
        send_stop_loss_alert(recent_pnl, bankroll, DAILY_STOP_LOSS_PCT)
        return []

    # Vérification des mouvements de cotes (annule les mises sur cotes bougées)
    check_odds_movement()

    football_signals = run_football_predictions()
    nba_signals      = run_nba_predictions()

    total = len(football_signals) + len(nba_signals)
    value = sum(1 for s in football_signals + nba_signals if s["is_value_bet"])
    logger.info(f"\nCycle terminé: {total} signaux | {value} value bets détectés")
    return football_signals + nba_signals


def send_weekly_report():
    """Envoie le résumé hebdomadaire des performances via Telegram."""
    stats        = tracker.get_stats()
    weekly_stats = tracker.get_weekly_stats()
    conf_stats   = tracker.get_roi_by_confidence()
    clv_stats    = tracker.get_avg_clv()
    brier_stats  = tracker.get_brier_score()
    send_weekly_summary(stats, weekly_stats, conf_stats, clv_stats, brier_stats)
    logger.info("Résumé hebdomadaire envoyé.")


# ════════════════════════════════════════════════════════════
# RE-ENTRAÎNEMENT MENSUEL
# ════════════════════════════════════════════════════════════

def retrain_models():
    """Re-entraîne football + NBA si c'est le 1er lundi du mois."""
    today = datetime.now()
    # 1er lundi du mois = lundi (weekday 0) ET jour entre 1 et 7
    if today.weekday() != 0 or today.day > 7:
        return

    logger.info("═══ RETRAINING MENSUEL ═══")
    send_message("🔄 <b>BetMind — Retraining démarré</b>\n\nEntraînement mensuel football + NBA en cours...")

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_log = []  # [(script, ok, accuracy_str, error_str)]

    for script in ["train_from_csv.py", "train_over_under.py", "train_nba.py"]:
        path = os.path.join(script_dir, script)
        if not os.path.exists(path):
            logger.warning(f"Script non trouvé: {path}")
            results_log.append((script, False, None, "Script introuvable"))
            continue

        logger.info(f"Lancement: {script}")
        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True,
                timeout=1800,  # 30 min max
                cwd=script_dir,
            )
            if result.returncode == 0:
                logger.info(f"{script} terminé OK")
                if result.stdout:
                    logger.info(result.stdout[-1000:])
                acc = _parse_accuracy(result.stdout)
                results_log.append((script, True, acc, None))
            else:
                err = result.stderr[-500:] if result.stderr else "Erreur inconnue"
                logger.error(f"{script} ERREUR (code {result.returncode}):\n{result.stderr[-2000:]}")
                results_log.append((script, False, None, err))
        except subprocess.TimeoutExpired:
            logger.error(f"{script} timeout (>30 min)")
            results_log.append((script, False, None, "Timeout >30 min"))
        except Exception as e:
            logger.error(f"{script} exception: {e}")
            results_log.append((script, False, None, str(e)))

    # Recharger les modèles en mémoire
    global football_model, ou_model, btts_model, nba_model
    football_model = BettingModel(sport="football")
    ou_model       = BettingModel(sport="ou_football")
    btts_model     = BettingModel(sport="btts_football")
    nba_model      = BettingModel(sport="nba")
    logger.info("Modèles rechargés en mémoire après retraining.")

    # Notification Telegram
    _send_retrain_summary(results_log)


def _parse_accuracy(stdout: str) -> str | None:
    """Extrait le taux d'accuracy depuis la sortie d'un script d'entraînement."""
    import re
    # Cherche patterns : "Accuracy: 0.539" ou "53.9%" ou "accuracy=0.539"
    for pattern in (
        r"[Aa]ccuracy[:\s=]+([0-9]+\.[0-9]+%?)",
        r"([0-9]{2,3}\.[0-9]+%)",          # ex: "61.9%" — garder le % dans le groupe
    ):
        m = re.search(pattern, stdout)
        if m:
            val = m.group(1)
            # Convertir 0.539 → 53.9%
            if "%" not in val:
                try:
                    val = f"{float(val) * 100:.1f}%"
                except ValueError:
                    pass
            return val
    return None


def _send_retrain_summary(results_log: list):
    """Envoie le résumé du retraining mensuel via Telegram."""
    lines = ["🤖 <b>BetMind — Retraining mensuel terminé</b>", ""]
    all_ok = True

    sport_names = {
        "train_from_csv.py": "⚽ Football",
        "train_nba.py":      "🏀 NBA",
    }

    for script, ok, accuracy, error in results_log:
        name = sport_names.get(script, script)
        if ok:
            acc_str = f"  • Accuracy : <b>{accuracy}</b>" if accuracy else "  • Accuracy : n/a"
            lines += [f"✅ {name}", acc_str, ""]
        else:
            lines += [f"❌ {name}", f"  • Erreur : {error}", ""]
            all_ok = False

    if all_ok:
        lines.append("Modèles rechargés — prédictions actives.")
    else:
        lines.append("⚠️ Au moins un modèle n'a pas pu être mis à jour.")

    send_message("\n".join(lines))


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


# ════════════════════════════════════════════════════════════
# SCHEDULER
# ════════════════════════════════════════════════════════════

def send_daily_report():
    """Résumé quotidien envoyé après le dernier cycle de prédictions."""
    stats       = tracker.get_stats()
    today_stats = tracker.get_today_stats()
    from telegram_bot import send_daily_summary
    send_daily_summary(stats, today_stats)
    logger.info("Résumé quotidien envoyé.")


def backup_database():
    """
    Backup quotidien de la DB SQLite dans data/backups/.
    Rotation automatique : 7 jours conservés, les anciens sont purgés.
    Alerte Telegram si le backup échoue.
    """
    from config import DB_PATH, DATA_DIR
    backup_dir = os.path.join(DATA_DIR, "backups")
    os.makedirs(backup_dir, exist_ok=True)

    today       = datetime.now().strftime("%Y-%m-%d")
    backup_path = os.path.join(backup_dir, f"betting_{today}.db")
    try:
        shutil.copy2(DB_PATH, backup_path)
        logger.info(f"DB backup OK → {backup_path}")

        # Purge des backups de plus de 7 jours
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        for fname in os.listdir(backup_dir):
            if fname.startswith("betting_") and fname.endswith(".db"):
                date_part = fname[8:18]  # betting_YYYY-MM-DD.db
                if date_part < cutoff:
                    os.remove(os.path.join(backup_dir, fname))
                    logger.info(f"Backup purgé : {fname}")
    except Exception as e:
        logger.error(f"DB backup échoué : {e}")
        send_message(
            f"🚨 <b>BACKUP DB ÉCHOUÉ</b>\n\n"
            f"Erreur : <code>{e}</code>\n\n"
            "Vérifier l'espace disque et les permissions.\n\n"
            "🤖 BetMind Agent"
        )


def check_silence():
    """Alerte Telegram si aucune prédiction n'a été faite depuis 24h."""
    try:
        with get_conn() as _sl_conn:
            row = _sl_conn.execute("SELECT MAX(created_at) FROM predictions").fetchone()
        last_str = row[0] if row and row[0] else None
        if last_str is None:
            return
        from datetime import timezone
        last_dt = datetime.fromisoformat(last_str).replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
        if age_h > 24:
            from telegram_bot import send_message
            send_message(
                f"⚠️ <b>MONITORING BetMind</b>\n\n"
                f"Aucune prédiction depuis <b>{age_h:.0f}h</b>.\n"
                f"Vérifier le service : <code>sudo systemctl status betmind</code>"
            )
            logger.warning(f"Silence détecté : dernière prédiction il y a {age_h:.1f}h")
    except Exception as e:
        logger.error(f"check_silence error: {e}")


def run_health_check():
    """
    Healthcheck complet exécuté chaque matin à 07h30 (avant le cycle 08h00).
    Vérifie :
      1. Modèles chargeables (football + NBA + OU)
      2. DB accessible + cohérente (aucune prédiction avec kelly_stake IS NULL)
      3. Quota football-data.org restant (alerte si < 20 requêtes)
      4. Espace disque libre (alerte si < 500 Mo)
    Envoie un rapport Telegram condensé : ✅ tout OK ou liste des problèmes.
    """
    import sqlite3
    from config import DB_PATH, MODELS_DIR, FD_DAILY_LIMIT
    problems = []

    # 1. Modèles chargeables
    for sport, mdl in (("football", football_model), ("NBA", nba_model), ("OU", ou_model), ("BTTS", btts_model)):
        if not mdl.is_trained:
            problems.append(f"Modèle {sport} non chargé — relancer le retraining")

    # 2. DB accessible + cohérente
    try:
        with get_conn() as _hc_conn:
            bad = _hc_conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE kelly_stake IS NULL AND outcome IS NULL"
            ).fetchone()[0]
        if bad > 0:
            problems.append(f"DB incohérente : {bad} prédiction(s) avec kelly_stake NULL")
    except Exception as e:
        problems.append(f"DB inaccessible : {e}")

    # 3. Quota football-data.org
    try:
        used      = get_fd_quota_used()
        remaining = FD_DAILY_LIMIT - used
        if remaining < 20:
            problems.append(f"Quota FD faible : {remaining} requêtes restantes ({used}/{FD_DAILY_LIMIT})")
    except Exception as e:
        problems.append(f"Vérification quota FD échouée : {e}")

    # 4. Espace disque libre
    try:
        usage      = shutil.disk_usage(".")
        free_mb    = usage.free / (1024 * 1024)
        if free_mb < 500:
            problems.append(f"Disque critique : {free_mb:.0f} Mo libres (seuil 500 Mo)")
    except Exception as e:
        problems.append(f"Vérification disque échouée : {e}")

    # Rapport Telegram
    if problems:
        lines = ["⚠️ <b>HEALTHCHECK BetMind — PROBLÈMES DÉTECTÉS</b>", ""]
        for p in problems:
            lines.append(f"  ❌ {p}")
        lines += ["", "🤖 BetMind Agent"]
        logger.warning(f"Healthcheck : {len(problems)} problème(s) détecté(s)")
    else:
        lines = [
            "✅ <b>HEALTHCHECK BetMind — Tout OK</b>",
            "",
            "  • Modèles : football ✓ | NBA ✓ | OU ✓",
            f"  • DB : accessible et cohérente",
            f"  • Quota FD : {FD_DAILY_LIMIT - get_fd_quota_used()} req. restantes",
            f"  • Disque : {shutil.disk_usage('.').free / (1024*1024):.0f} Mo libres",
            "",
            "🤖 BetMind Agent",
        ]
        logger.info("Healthcheck : tout OK")

    send_message("\n".join(lines))


def start_scheduler():
    logger.info("Scheduler démarré.")
    schedule.every().day.at("07:30").do(run_health_check)      # healthcheck avant cycle
    schedule.every().day.at("08:00").do(run_with_results)
    schedule.every().day.at("18:00").do(run_with_results)
    schedule.every().day.at("22:00").do(run_nba_predictions)
    schedule.every().day.at("23:30").do(send_daily_report)    # résumé fin de journée
    schedule.every().day.at("02:00").do(backup_database)       # backup DB quotidien
    schedule.every().day.at("10:00").do(check_silence)        # monitoring quotidien
    schedule.every().monday.at("09:00").do(send_weekly_report)
    schedule.every().monday.at("03:00").do(retrain_models)  # 1er lundi du mois seulement
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