# ============================================================
# telegram_bot.py — Alertes Telegram
# ============================================================

import logging
import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

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

    lines += ["", "─" * 30, "🤖 BetMind Agent"]
    send_message("\n".join(lines))


def send_daily_summary(stats: dict):
    """Résumé quotidien de la bankroll."""
    lines = [
        "📈 <b>RÉSUMÉ QUOTIDIEN</b>",
        "",
        f"💰 Bankroll : <b>{stats.get('balance', 0):,.0f} FCFA</b>",
        f"🎯 Paris joués : {stats.get('total_bets', 0)}",
        f"✅ Victoires : {stats.get('wins', 0)} ({stats.get('win_rate', 0):.1f}%)",
        f"❌ Défaites : {stats.get('losses', 0)}",
        f"📊 ROI : {stats.get('roi', 0):+.2f}%",
        f"💵 P&L total : {stats.get('total_pnl', 0):+,.0f} FCFA",
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