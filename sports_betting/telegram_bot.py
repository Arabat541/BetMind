# ============================================================
# telegram_bot.py — Alertes Telegram
# ============================================================

import logging
import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, INITIAL_BANKROLL

logger = logging.getLogger(__name__)

EMOJI_SPORT  = {"football": "⚽", "nba": "🏀"}
EMOJI_RESULT = {"H": "🏠", "D": "🤝", "A": "✈️"}
EMOJI_VALUE  = "🔥"
EMOJI_SIGNAL = "📊"


def send_message(text: str) -> bool:
    """Envoie un message Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured.")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        r = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       text,
            "parse_mode": "HTML"
        }, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False


def send_prediction_alert(signal: dict, stake_info: dict):
    """Formate et envoie une alerte de prédiction."""
    sport    = signal.get("sport", "football")
    is_value = signal.get("is_value_bet", False)

    header = f"{EMOJI_VALUE if is_value else EMOJI_SIGNAL} <b>{'VALUE BET' if is_value else 'SIGNAL'}</b>"
    sport_emoji = EMOJI_SPORT.get(sport, "🏆")

    lines = [
        header,
        "",
        f"{sport_emoji} <b>{signal['home_team']} vs {signal['away_team']}</b>",
        f"🏆 {signal.get('league', '')}",
        f"📅 {signal.get('match_date', '')[:10]}",
        "",
        f"<b>Prédiction :</b> {EMOJI_RESULT.get(signal['pred_result'], '')} {signal['pred_name']}",
        f"<b>Confiance :</b> {signal['confidence']:.1%}",
        "",
        _format_probas(signal, sport),
    ]

    if is_value and signal.get("value_bets"):
        vb = signal["value_bets"][0]
        lines += [
            "",
            "🎯 <b>Value Bet détecté :</b>",
            f"  • Mise sur : {vb['result_name']}",
            f"  • Cote      : {vb['odd']:.2f}",
            f"  • Edge      : +{vb['edge']:.1%}",
            f"  • EV        : {vb['expected_value']:+.3f}",
        ]

    if stake_info.get("stake_amount", 0) > 0:
        lines += [
            "",
            "💰 <b>Mise recommandée (Kelly) :</b>",
            f"  • {stake_info['stake_amount']:,.0f} FCFA ({stake_info['stake_pct']:.1%} bankroll)",
            f"  • Profit attendu : +{stake_info.get('profit_expected', 0):,.0f} FCFA",
        ]

    # Blessures NBA
    if sport == "nba":
        home_out = signal.get("home_injuries_out", 0)
        away_out = signal.get("away_injuries_out", 0)
        home_dtd = signal.get("home_injuries_dtd", 0)
        away_dtd = signal.get("away_injuries_dtd", 0)
        if home_out + away_out + home_dtd + away_dtd > 0:
            lines += ["", "🏥 <b>Blessures (ESPN) :</b>"]
            if home_out or home_dtd:
                lines.append(f"  🏠 {signal['home_team'][:20]}: {home_out} OUT, {home_dtd} DTD")
            if away_out or away_dtd:
                lines.append(f"  ✈️  {signal['away_team'][:20]}: {away_out} OUT, {away_dtd} DTD")

    # Sharp money — mouvement de cote depuis l'ouverture
    mvt = signal.get("opening_movement_pct")
    if mvt is not None:
        mvt_pct = mvt * 100
        if mvt_pct > 5:
            lines += ["", f"📡 <b>Sharp Money :</b> cote ↓ {mvt_pct:.1f}% — sharps confirment ✅"]
        elif mvt_pct < -5:
            lines += ["", f"📡 <b>Sharp Money :</b> cote ↑ {abs(mvt_pct):.1f}% — marché contre nous ⚠️"]

    # Météo — match sous la pluie ou vent fort
    if signal.get("rainy_match"):
        rain_mm   = signal.get("rain_mm", 0)
        wind_kmh  = signal.get("wind_kmh", 0)
        temp_c    = signal.get("temp_c")
        weather_parts = []
        if rain_mm:
            weather_parts.append(f"🌧️ {rain_mm:.1f} mm/h")
        if wind_kmh:
            weather_parts.append(f"💨 {wind_kmh:.0f} km/h")
        if temp_c is not None:
            weather_parts.append(f"🌡️ {temp_c:.0f}°C")
        lines += ["", f"⛈️ <b>Conditions météo :</b> {' | '.join(weather_parts)}"]

    # RLM — Reverse Line Movement détecté
    if signal.get("rlm_detected"):
        rlm_move = signal.get("rlm_line_move", 0)
        rlm_sc   = signal.get("rlm_score", 0)
        lines += ["", f"📡 <b>Reverse Line Movement :</b> cote ↑ {rlm_move:+.1%} contre le public (score {rlm_sc:.2f}) ⚡ Sharp money"]

    # xG Understat — AE
    h_xg  = signal.get("home_xg_avg", 0)
    a_xg  = signal.get("away_xg_avg", 0)
    h_xga = signal.get("home_xga_avg", 0)
    a_xga = signal.get("away_xga_avg", 0)
    if h_xg or a_xg:
        lines += [
            "",
            "⚽ <b>xG récents (Understat) :</b>",
            f"  🏠 {signal['home_team'][:18]}: xG {h_xg:.2f} | xGA {h_xga:.2f}",
            f"  ✈️  {signal['away_team'][:18]}: xG {a_xg:.2f} | xGA {a_xga:.2f}",
        ]

    # Valeur marchande effectifs — AE
    vr = signal.get("squad_value_ratio", 0)
    hv = signal.get("home_squad_value", 0)
    av = signal.get("away_squad_value", 0)
    if hv and av and vr and vr != 1.0:
        fav   = signal["home_team"] if vr >= 1 else signal["away_team"]
        ratio = vr if vr >= 1 else round(1 / vr, 2)
        lines += ["", f"💶 <b>Effectifs :</b> {fav[:18]} ×{ratio:.1f} plus cher ({hv:.0f}M€ vs {av:.0f}M€)"]

    lines += ["", "─" * 30, "🤖 BetMind Agent"]
    send_message("\n".join(lines))


def send_bankroll_alert(balance: float, threshold: float):
    """Alerte Telegram quand le bankroll descend sous le seuil critique."""
    pct_of_initial = balance / INITIAL_BANKROLL * 100
    lines = [
        "🚨 <b>ALERTE BANKROLL CRITIQUE</b>",
        "",
        f"💰 Bankroll actuel : <b>{balance:,.0f} FCFA</b>",
        f"⚠️  Seuil d'alerte  : {threshold:,.0f} FCFA",
        f"📉 Niveau           : {pct_of_initial:.0f}% du bankroll initial",
        "",
        "Actions recommandées :",
        "  • Réduire les mises (Kelly déjà conservateur)",
        "  • Suspendre le bot temporairement",
        "  • Recharger le bankroll si nécessaire",
        "",
        "🤖 BetMind Agent"
    ]
    send_message("\n".join(lines))


def send_weekly_summary(stats: dict, weekly_stats: dict = None,
                        conf_stats: list = None, clv_stats: dict = None,
                        brier_stats: dict = None):
    """Résumé hebdomadaire des performances envoyé chaque lundi."""
    balance  = stats.get("balance", 0)
    roi      = stats.get("roi", 0)
    wins     = stats.get("wins", 0)
    losses   = stats.get("losses", 0)
    total    = stats.get("total_bets", 0)
    win_rate = stats.get("win_rate", 0)
    pnl      = stats.get("total_pnl", 0)

    variation = balance - INITIAL_BANKROLL
    roi_emoji = "📈" if roi >= 0 else "📉"
    pnl_sign  = "+" if pnl >= 0 else ""

    lines = ["📊 <b>RÉSUMÉ HEBDOMADAIRE</b>", ""]

    # ── Stats de la semaine écoulée ──────────────────────────
    if weekly_stats:
        roi_w    = weekly_stats.get("roi", 0)
        pnl_w    = weekly_stats.get("pnl", 0)
        bets_w   = weekly_stats.get("bets", 0)
        wins_w   = weekly_stats.get("wins", 0)
        losses_w = weekly_stats.get("losses", 0)
        wr_w     = weekly_stats.get("win_rate", 0)

        roi_emoji_w = "📈" if roi_w >= 0 else "📉"
        pnl_sign_w  = "+" if pnl_w >= 0 else ""

        lines += [
            "📅 <b>7 derniers jours</b>",
            f"  🎯 {bets_w} paris  |  {wins_w}W / {losses_w}L  ({wr_w:.1f}%)",
            f"  {roi_emoji_w} ROI semaine : <b>{roi_w:+.2f}%</b>",
            f"  💵 P&amp;L : {pnl_sign_w}{pnl_w:,.0f} FCFA",
        ]

        best_league     = weekly_stats.get("best_league")
        best_league_pnl = weekly_stats.get("best_league_pnl", 0)
        if best_league:
            sign_lg = "+" if best_league_pnl >= 0 else ""
            lines.append(f"  🏆 Ligue + rentable : {best_league} ({sign_lg}{best_league_pnl:,.0f} FCFA)")

        best_bet = weekly_stats.get("best_bet")
        if best_bet:
            lines.append(
                f"  🔥 Meilleur pari : {best_bet['match']}"
                f" (+{best_bet['pnl']:,.0f} FCFA @ {best_bet['odd']:.2f})"
            )

        streak_type = weekly_stats.get("streak_type")
        streak_val  = weekly_stats.get("streak_val", 0)
        if streak_type and streak_val >= 2:
            streak_emoji = "🔥" if streak_type == "W" else "❄️"
            streak_label = "victoires" if streak_type == "W" else "défaites"
            lines.append(f"  {streak_emoji} Série : {streak_val} {streak_label} consécutives")

        lines.append("")

    # ── Closing Line Value ───────────────────────────────────
    if clv_stats:
        avg_clv   = clv_stats.get("avg_clv", 0)
        beat_rate = clv_stats.get("beat_rate", 0)
        n_clv     = clv_stats.get("n_bets", 0)
        clv_emoji = "📈" if avg_clv > 0 else "📉"
        clv_verdict = "Edge confirmé ✓" if avg_clv > 0 else "Marché nous devance ⚠️"
        lines += [
            "📐 <b>Closing Line Value</b>",
            f"  {clv_emoji} CLV moyen : <b>{avg_clv:+.2f}%</b>  ({n_clv} paris)",
            f"  🎯 Beat rate : {beat_rate:.1f}%  — {clv_verdict}",
            "",
        ]

    # ── ROI par tranche de confiance ─────────────────────────
    if conf_stats:
        lines.append("🎯 <b>ROI par confiance</b>")
        tier_map = {d["tier"]: d for d in conf_stats}
        for tier in ("<55%", "55-65%", ">65%"):
            d = tier_map.get(tier)
            if not d:
                continue
            roi   = d["roi"]
            sign  = "+" if roi >= 0 else ""
            emoji = "📈" if roi >= 0 else "📉"
            lines.append(
                f"  {emoji} {tier:>6}  {sign}{roi:.2f}%"
                f"  ({d['bets']} paris, {d['win_rate']:.0f}% WR)"
            )
        # Alerte si calibration inversée
        low  = tier_map.get("<55%",  {}).get("roi", 0)
        high = tier_map.get(">65%",  {}).get("roi", 0)
        if len(conf_stats) >= 2 and high < low:
            lines.append("  ⚠️ Calibration à vérifier : confiance >65% < <55%")
        lines.append("")

    # ── Brier Score (calibration post-déploiement) ──────────
    if brier_stats:
        bs    = brier_stats.get("score", 0)
        n_bs  = brier_stats.get("n", 0)
        if bs < 0.20:
            bs_verdict = "Calibration excellente ✓"
            bs_emoji   = "🟢"
        elif bs < 0.25:
            bs_verdict = "Calibration correcte ✓"
            bs_emoji   = "🟡"
        elif bs < 0.28:
            bs_verdict = "Calibration acceptable ⚠️"
            bs_emoji   = "🟠"
        else:
            bs_verdict = "Modèle sous-calibré — retraining conseillé ❌"
            bs_emoji   = "🔴"
        lines += [
            "📐 <b>Brier Score (calibration)</b>",
            f"  {bs_emoji} BS = <b>{bs:.4f}</b>  ({n_bs} paris réglés)",
            f"  → {bs_verdict}",
            "  <i>Références : &lt;0.20 excellent | 0.25 naïf (50/50) | &gt;0.28 alerte</i>",
            "",
        ]

    # ── Stats cumulées ───────────────────────────────────────
    lines += [
        "📊 <b>Cumulé</b>",
        f"  💰 Bankroll : <b>{balance:,.0f} FCFA</b>",
        f"  📊 Départ : {INITIAL_BANKROLL:,.0f} FCFA  |  {'+' if variation >= 0 else ''}{variation:,.0f} FCFA",
        "",
        f"  {roi_emoji} ROI global    : {roi:+.2f}%",
        f"  💵 P&amp;L total      : {pnl_sign}{pnl:,.0f} FCFA",
        "",
        f"  🎯 Paris joués  : {total}",
        f"  ✅ Gagnés       : {wins}  ({win_rate:.1f}%)",
        f"  ❌ Perdus       : {losses}",
        "",
        "🤖 BetMind Agent"
    ]
    send_message("\n".join(lines))


def send_daily_summary(stats: dict, today_stats: dict = None):
    """Résumé quotidien de la bankroll, avec stats du jour si disponibles."""
    lines = ["📈 <b>RÉSUMÉ QUOTIDIEN</b>", ""]

    # ── Stats du jour ────────────────────────────────────────
    if today_stats:
        bets     = today_stats.get("bets", 0)
        settled  = today_stats.get("settled", 0)
        wins     = today_stats.get("wins", 0)
        losses   = today_stats.get("losses", 0)
        pnl_d    = today_stats.get("pnl", 0)
        roi_d    = today_stats.get("roi", 0)
        wr_d     = today_stats.get("win_rate", 0)
        pnl_sign = "+" if pnl_d >= 0 else ""
        roi_sign = "+" if roi_d >= 0 else ""
        pnl_color_tag = ""  # HTML non supporté pour la couleur inline en Telegram

        lines += [
            "📅 <b>Aujourd'hui</b>",
            f"  🎯 Paris : {bets} générés  |  {settled} réglés",
        ]
        if settled > 0:
            lines += [
                f"  ✅ {wins}W / ❌ {losses}L  ({wr_d:.1f}%)",
                f"  💵 P&L : <b>{pnl_sign}{pnl_d:,.0f} FCFA</b>  ({roi_sign}{roi_d:.2f}%)",
            ]
        lines += [""]

    # ── Stats globales ───────────────────────────────────────
    roi   = stats.get("roi", 0)
    pnl   = stats.get("total_pnl", 0)
    lines += [
        "📊 <b>Cumulé</b>",
        f"  💰 Bankroll : <b>{stats.get('balance', 0):,.0f} FCFA</b>",
        f"  🎯 Paris réglés : {stats.get('total_bets', 0)}  "
        f"({stats.get('wins', 0)}W / {stats.get('losses', 0)}L — {stats.get('win_rate', 0):.1f}%)",
        f"  📈 ROI : {roi:+.2f}%  |  P&L : {pnl:+,.0f} FCFA",
        "",
        "🤖 BetMind Agent",
    ]
    send_message("\n".join(lines))


def send_odds_movement_alert(home_team: str, away_team: str, league: str,
                             pred_result: str, odd_saved: float,
                             odd_current: float, drop_pct: float):
    """Alerte Telegram quand une cote a trop bougé et qu'une mise est annulée."""
    NAMES = {"H": "Domicile", "D": "Nul", "A": "Extérieur",
             "O": "Over 2.5", "U": "Under 2.5"}
    sport_emoji = "⚽" if pred_result in ("H", "D", "A", "O", "U") else "🏀"
    lines = [
        f"📉 <b>COTE BOUGÉE — MISE ANNULÉE</b>",
        "",
        f"{sport_emoji} <b>{home_team} vs {away_team}</b>",
        f"🏆 {league}",
        "",
        f"Pari visé     : {NAMES.get(pred_result, pred_result)}",
        f"Cote initiale : {odd_saved:.2f}",
        f"Cote actuelle : <b>{odd_current:.2f}</b>  (−{drop_pct:.1%})",
        "",
        "⚠️ Signal de sharp money détecté.",
        "La mise a été annulée automatiquement.",
        "",
        "🤖 BetMind Agent"
    ]
    send_message("\n".join(lines))


def send_stop_loss_alert(pnl: float, bankroll: float, threshold_pct: float):
    """Alerte Telegram quand le stop-loss journalier est déclenché."""
    threshold_fcfa = bankroll * threshold_pct
    lines = [
        "🛑 <b>STOP-LOSS DÉCLENCHÉ</b>",
        "",
        f"📉 P&amp;L récent (48h) : <b>{pnl:+,.0f} FCFA</b>",
        f"⚠️  Seuil            : -{threshold_pct:.0%} bankroll  ({threshold_fcfa:,.0f} FCFA)",
        f"💰 Bankroll actuel  : {bankroll:,.0f} FCFA",
        "",
        "Aucune nouvelle mise aujourd'hui.",
        "Le bot reprendra automatiquement demain.",
        "",
        "🤖 BetMind Agent"
    ]
    send_message("\n".join(lines))


def send_model_drift_alert(win_rate: float, n_bets: int):
    """Alerte Telegram si le win rate récent chute sous 30% (dérive potentielle)."""
    lines = [
        "⚠️ <b>ALERTE : DÉRIVE DU MODÈLE</b>",
        "",
        f"📉 Win rate sur les {n_bets} derniers paris réglés : <b>{win_rate:.1f}%</b>",
        f"🎯 Seuil d'alerte : 30%",
        "",
        "Causes possibles :",
        "  • Données d'entraînement obsolètes (marché a changé)",
        "  • Changement structurel dans les ligues suivies",
        "  • Sur-ajustement du modèle (overfitting)",
        "",
        "Actions recommandées :",
        "  • Relancer l'entraînement : <code>train_from_csv.py</code>",
        "  • Suspendre temporairement les mises",
        "  • Vérifier les seuils d'edge dans config.py",
        "",
        "🤖 BetMind Agent"
    ]
    send_message("\n".join(lines))


def _format_probas(signal: dict, sport: str) -> str:
    """Formate le tableau des probabilités."""
    ph = signal.get("prob_home", 0)
    pa = signal.get("prob_away", 0)
    pd_ = signal.get("prob_draw", 0)

    bar_h = _prob_bar(ph)
    bar_a = _prob_bar(pa)

    if sport == "football":
        pd_bar = _prob_bar(pd_)
        return (
            f"<b>Probas :</b>\n"
            f"  🏠 Domicile : {ph:.1%} {bar_h}\n"
            f"  🤝 Nul      : {pd_:.1%} {pd_bar}\n"
            f"  ✈️  Extérieur: {pa:.1%} {bar_a}"
        )
    else:
        return (
            f"<b>Probas :</b>\n"
            f"  🏠 Domicile : {ph:.1%} {bar_h}\n"
            f"  ✈️  Extérieur: {pa:.1%} {bar_a}"
        )


def _prob_bar(p: float, length: int = 10) -> str:
    """Barre de progression ASCII."""
    filled = round(p * length)
    return "█" * filled + "░" * (length - filled)


def send_correct_score_alert(signal: dict, stake_info: dict):
    """Alerte Telegram pour un value bet Correct Score (marché AL)."""
    vb = signal["value_bets"][0]
    top = signal.get("top_scores", [])

    lines = [
        "🎯 <b>VALUE BET — CORRECT SCORE</b>",
        "",
        f"⚽ <b>{signal['home_team']} vs {signal['away_team']}</b>",
        f"🏆 {signal.get('league', '')}",
        f"📅 {signal.get('match_date', '')[:10]}",
        "",
        f"<b>Score prédit :</b> {vb['score']}",
        f"<b>Proba modèle :</b> {vb['p_model']:.1%}",
        f"<b>Proba implicite :</b> {vb['p_implied']:.1%}",
        f"<b>Edge :</b> +{vb['edge']:.1%}",
        f"<b>Cote :</b> {vb['odd']:.2f}",
        f"<b>EV :</b> {vb['ev']:+.3f}",
    ]

    if stake_info.get("stake_amount", 0) > 0:
        lines += [
            "",
            "💰 <b>Mise recommandée (Kelly) :</b>",
            f"  • {stake_info['stake_amount']:,.0f} FCFA ({stake_info['stake_pct']:.1%} bankroll)",
        ]

    if top:
        lines += ["", "📊 <b>Top 5 scores modèle :</b>"]
        for s in top[:5]:
            bar = _prob_bar(s["prob"], length=8)
            lines.append(f"  {s['score']:>4}  {s['prob']:.1%}  {bar}")

    lines += ["", "─" * 30, "🤖 BetMind Agent"]
    send_message("\n".join(lines))


def send_rlm_alert(home_team: str, away_team: str, league: str,
                   outcome: str, odd_open: float, odd_current: float,
                   line_move_pct: float, public_pct: float):
    """Alerte Reverse Line Movement : ligne monte contre le public."""
    lines = [
        "📡 <b>REVERSE LINE MOVEMENT</b>",
        "",
        f"⚽ <b>{home_team} vs {away_team}</b>",
        f"🏆 {league}",
        "",
        f"<b>Outcome :</b> {outcome}",
        f"<b>Cote ouverture :</b> {odd_open:.2f}",
        f"<b>Cote actuelle  :</b> <b>{odd_current:.2f}</b>  ({line_move_pct:+.1%})",
        f"<b>% mises public :</b> {public_pct:.0f}% contre ce side",
        "",
        "⚡ <b>Signal Sharp :</b> la ligne monte CONTRE le public.",
        "Les bookmakers suivent l'argent professionnel.",
        "",
        "🤖 BetMind Agent"
    ]
    send_message("\n".join(lines))


def send_bhg_alert(signal: dict, stake_info: dict):
    """Alerte Telegram pour un value bet Both Halves Goals (marché AQ)."""
    lines = [
        "⏱️ <b>VALUE BET — BOTH HALVES GOALS</b>",
        "",
        f"⚽ <b>{signal.get('home_team', '?')} vs {signal.get('away_team', '?')}</b>",
        f"🏆 {signal.get('league', '')}",
        f"📅 {signal.get('match_date', '')[:10]}",
        "",
        f"<b>Pari :</b> {signal.get('result_name', signal.get('side', '?'))}",
        f"<b>Proba modèle :</b> {signal.get('prob_model', 0):.1%}",
        f"<b>Proba implicite :</b> {signal.get('prob_impl', 0):.1%}",
        f"<b>Edge :</b> +{signal.get('edge', 0):.1%}",
        f"<b>Cote :</b> {signal.get('odd', 0):.2f}",
        f"<b>EV :</b> {signal.get('ev', 0):+.3f}",
    ]

    if stake_info.get("stake_amount", 0) > 0:
        lines += [
            "",
            "💰 <b>Mise recommandée (Kelly) :</b>",
            f"  • {stake_info['stake_amount']:,.0f} FCFA ({stake_info['stake_pct']:.1%} bankroll)",
        ]

    lines += [
        "",
        "💡 <i>BHG : les deux équipes marquent dans chaque mi-temps.</i>",
        "─" * 30,
        "🤖 BetMind Agent",
    ]
    send_message("\n".join(lines))


def send_dutch_alert(signal: dict):
    """Alerte Telegram pour une opportunité Dutching (AR)."""
    outcomes = signal.get("outcomes", [])
    roi      = signal.get("dutch_roi", 0)
    profit   = signal.get("dutch_profit", 0)
    total    = signal.get("total_stake", 0)

    lines = [
        "🎰 <b>DUTCHING — PROFIT GARANTI</b>",
        "",
        f"⚽ <b>{signal.get('home_team', '?')} vs {signal.get('away_team', '?')}</b>",
        f"🏆 {signal.get('league', '')}",
        f"📅 {signal.get('match_date', '')[:10]}",
        "",
        f"<b>Profit garanti :</b> {profit:,.0f} FCFA (+{roi:.1%} ROI)",
        f"<b>Mise totale :</b> {total:,.0f} FCFA",
        f"<b>Somme impl. :</b> {signal.get('impl_sum', 0):.3f} (< 1.0 = edge)",
        "",
        "<b>Répartition :</b>",
    ]

    for o in outcomes:
        lines.append(
            f"  • {o['label']:15s} @ {o['odd']:.2f} → {o['dutch_stake']:,.0f} FCFA"
        )

    lines += [
        "",
        "💡 <i>Dutching : mise répartie pour profit identique quel que soit le résultat.</i>",
        "─" * 30,
        "🤖 BetMind Agent",
    ]
    send_message("\n".join(lines))


def send_steam_alert(steam_move: dict):
    """Alerte Telegram pour un steam move détecté (AT)."""
    outcome_names = {"H": "Domicile", "D": "Nul", "A": "Extérieur"}
    direction_symbols = {"down": "⬇️ cote en baisse (signal sharp money)", "up": "⬆️ cote en hausse"}
    outcome = steam_move.get("outcome", "?")
    direction = steam_move.get("direction", "down")

    lines = [
        "🚨 <b>STEAM MOVE DÉTECTÉ</b>",
        "",
        f"⚽ <b>{steam_move.get('home_team', '?')} vs {steam_move.get('away_team', '?')}</b>",
        f"<b>Issue :</b> {outcome_names.get(outcome, outcome)}",
        f"<b>Direction :</b> {direction_symbols.get(direction, direction)}",
        f"<b>Variation :</b> {steam_move.get('change_pct', 0):.1%} en {steam_move.get('elapsed_sec', 0)}s",
        f"<b>Cote moy. avant :</b> {steam_move.get('avg_old_odd', 0):.2f}",
        f"<b>Cote moy. après :</b> {steam_move.get('avg_new_odd', 0):.2f}",
        f"<b>Books confirmant :</b> {steam_move.get('books_moved', 0)}",
        "",
        "💡 <i>Move simultané sur plusieurs books = argent professionnel.</i>",
        "─" * 30,
        "🤖 BetMind Agent",
    ]
    send_message("\n".join(lines))


def send_injury_sentiment_alert(home_team: str, away_team: str,
                                league: str, match_date: str,
                                home_data: dict, away_data: dict):
    """Alerte Telegram si blessures significatives détectées (AP)."""
    lines = [
        "🏥 <b>ALERTE BLESSURES — NLP</b>",
        "",
        f"⚽ <b>{home_team} vs {away_team}</b>",
        f"🏆 {league}",
        f"📅 {match_date[:10] if match_date else ''}",
        "",
    ]
    for label, team, data in [("Domicile", home_team, home_data),
                               ("Extérieur", away_team, away_data)]:
        if data.get("injury_count", 0) > 0 or data.get("sentiment", 0) < -0.3:
            lines.append(f"<b>{label} — {team}</b>")
            lines.append(f"  Sentiment : {data['sentiment']:+.2f} ({data['injury_count']} blessure(s))")
            for h in data.get("headlines", [])[:3]:
                lines.append(f"  • [{h['label']}] {h['title'][:80]}")
            lines.append("")

    lines += ["💡 <i>Source : RSS BBC Sport / Sky Sports / Guardian</i>",
              "─" * 30, "🤖 BetMind Agent"]
    send_message("\n".join(lines))


def send_account_health_alert(report: dict):
    """Alerte Telegram pour la santé des comptes bookmakers (AU)."""
    lines = [
        "⚠️ <b>COMPTE BOOKMAKER — ALERTE LIMITATION</b>",
        "",
        f"<b>Bookmaker :</b> {report.get('bookmaker', '?')}",
        f"<b>Score santé :</b> {report.get('health_score', 0):.0%}",
        "",
        "<b>Indicateurs :</b>",
    ]
    for indicator, status in report.get("indicators", {}).items():
        emoji = "🔴" if status.get("warning") else "🟢"
        lines.append(f"  {emoji} {indicator}: {status.get('value', 'N/A')}")

    lines += [
        "",
        f"<b>Recommandation :</b> {report.get('recommendation', 'Surveiller.')}",
        "─" * 30,
        "🤖 BetMind Agent",
    ]
    send_message("\n".join(lines))