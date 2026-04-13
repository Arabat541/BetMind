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
                lines.append(
                    f"  🏠 {signal['home_team'][:20]}: {home_out} OUT, {home_dtd} DTD"
                )
            if away_out or away_dtd:
                lines.append(
                    f"  ✈️  {signal['away_team'][:20]}: {away_out} OUT, {away_dtd} DTD"
                )

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


def send_weekly_summary(stats: dict):
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

    lines = [
        f"📊 <b>RÉSUMÉ HEBDOMADAIRE</b>",
        "",
        f"💰 Bankroll : <b>{balance:,.0f} FCFA</b>",
        f"   Départ : {INITIAL_BANKROLL:,.0f} FCFA  |  {'+' if variation >= 0 else ''}{variation:,.0f} FCFA",
        "",
        f"{roi_emoji} <b>ROI global</b>    : {roi:+.2f}%",
        f"💵 P&amp;L total      : {pnl_sign}{pnl:,.0f} FCFA",
        "",
        f"🎯 Paris joués  : {total}",
        f"✅ Gagnés       : {wins}  ({win_rate:.1f}%)",
        f"❌ Perdus       : {losses}",
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