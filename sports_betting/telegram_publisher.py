# ============================================================
# telegram_publisher.py — Publication automatique @BetmindAcademy
#
# 3 jobs APScheduler :
#   job_analyse()     → 18h00 chaque jour   : prédictions du lendemain
#   job_resultats()   → 23h00 chaque jour   : résultats du jour
#   job_bilan_hebdo() → dimanche 20h00      : bilan de la semaine
#
# Variables d'env requises :
#   TELEGRAM_TOKEN           — bot token (même que le bot principal)
#   TELEGRAM_CHANNEL_PUBLIC  — ex. "@BetmindAcademy" ou chat_id numérique
#   DATABASE_URL             — postgresql://... (optionnel, fallback SQLite)
# ============================================================

import os
import sys
import logging
import requests
from datetime import datetime, timedelta, date

from dotenv import load_dotenv
load_dotenv()

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/publisher.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Config Telegram ──────────────────────────────────────────
BOT_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
CHANNEL_ID  = os.getenv("TELEGRAM_CHANNEL_PUBLIC", "@BetmindAcademy")

# ── DB ───────────────────────────────────────────────────────
# Import différé pour ne pas planter si les dépendances manquent au boot
def _get_db():
    from db import get_conn, ph, is_postgres, adapt_ddl
    return get_conn, ph, is_postgres, adapt_ddl


# ============================================================
# HELPERS TELEGRAM
# ============================================================

def _send(text: str, parse_mode: str = "HTML") -> bool:
    """
    Envoie un message sur le canal public (@BetmindAcademy).
    Réutilise le même bot token que telegram_bot.py, mais cible CHANNEL_ID.
    """
    if not BOT_TOKEN or not CHANNEL_ID:
        logger.error("TELEGRAM_TOKEN ou TELEGRAM_CHANNEL_PUBLIC manquant.")
        return False
    try:
        # Même endpoint que telegram_bot.send_message, chat_id différent
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={
            "chat_id":                  CHANNEL_ID,
            "text":                     text,
            "parse_mode":               parse_mode,
            "disable_web_page_preview": True,
        }, timeout=15)
        r.raise_for_status()
        logger.info("Publié sur %s (%d car.)", CHANNEL_ID, len(text))
        return True
    except Exception as e:
        logger.error("Telegram canal error: %s", e)
        return False


def _prob_bar(p: float, length: int = 10) -> str:
    filled = round(p * length)
    return "█" * filled + "░" * (length - filled)


def _confidence_label(c: float) -> str:
    if c >= 0.70:
        return "🟢 Très élevée"
    if c >= 0.60:
        return "🟡 Élevée"
    if c >= 0.52:
        return "🟠 Modérée"
    return "🔴 Faible"


RESULT_LABELS = {
    "H": "Victoire Domicile",
    "D": "Match Nul",
    "A": "Victoire Extérieur",
    "O": "Over 2.5",
    "U": "Under 2.5",
}

SPORT_EMOJI = {"football": "⚽", "nba": "🏀"}

# ── Leçons éducatives (rotation quotidienne) ─────────────────
_LESSONS = [
    (
        "📖 <b>Leçon du jour : La Valeur Attendue (EV)</b>\n\n"
        "Un pari est rentable à long terme si <b>EV = probabilité × cote − 1 &gt; 0</b>.\n"
        "Ex : si ton modèle donne 55% de chance à une cote de 2.00 :\n"
        "EV = 0.55 × 2.00 − 1 = <b>+0.10</b> → pari rentable ✅"
    ),
    (
        "📖 <b>Leçon du jour : Le Critère de Kelly</b>\n\n"
        "Kelly = (p × b − q) / b\n"
        "Où p = probabilité, b = profit net, q = 1 − p.\n"
        "BetMind applique le <b>Kelly fractionnel (25%)</b> pour limiter la variance.\n"
        "Ne jamais miser plus de 5% de sa bankroll sur un seul pari."
    ),
    (
        "📖 <b>Leçon du jour : Le Vig (marge du bookmaker)</b>\n\n"
        "Les cotes affichées incluent une marge (~5-10%).\n"
        "Pour trouver la vraie probabilité implicite, on utilise la méthode Shin.\n"
        "BetMind compare ces probas corrigées à son modèle pour trouver l'edge."
    ),
    (
        "📖 <b>Leçon du jour : Sharp Money vs Public Money</b>\n\n"
        "Le <b>sharp money</b> vient des parieurs professionnels.\n"
        "Quand la cote monte MALGRÉ un fort pourcentage de mises public →\n"
        "c'est un <b>Reverse Line Movement</b> : signe que les sharps jouent l'autre sens."
    ),
    (
        "📖 <b>Leçon du jour : Le Closing Line Value (CLV)</b>\n\n"
        "Un bon modèle bat la cote de fermeture (closing line).\n"
        "Si tu pariais à 2.10 et la cote ferme à 1.90, ton CLV = +10%.\n"
        "Le CLV est le meilleur indicateur de rentabilité à long terme."
    ),
    (
        "📖 <b>Leçon du jour : Gestion du Bankroll</b>\n\n"
        "La règle d'or : ne jamais exposer plus de 15% de sa bankroll par jour.\n"
        "BetMind applique 3 niveaux de protection :\n"
        "  • Stop-loss journalier (−5%)\n"
        "  • Drawdown global (−20%)\n"
        "  • Limite d'exposition par équipe (10%)"
    ),
    (
        "📖 <b>Leçon du jour : Le xG (Expected Goals)</b>\n\n"
        "Le xG mesure la qualité des occasions, pas seulement les buts marqués.\n"
        "Une équipe avec xG 1.8 qui gagne 1-0 est sous-performante.\n"
        "BetMind intègre le xG sur 10 matchs pour estimer la forme réelle."
    ),
]


def _daily_lesson() -> str:
    """Retourne la leçon du jour (rotation basée sur le jour de l'année)."""
    day_of_year = date.today().timetuple().tm_yday
    return _LESSONS[day_of_year % len(_LESSONS)]


# ============================================================
# MIGRATION DB : ajout colonne published_academy
# ============================================================

def _ensure_published_column():
    """Ajoute published_academy à predictions si elle n'existe pas."""
    get_conn, ph, is_postgres, adapt_ddl = _get_db()
    if_not_exists = "IF NOT EXISTS " if is_postgres() else ""
    try:
        with get_conn() as conn:
            conn.execute(
                f"ALTER TABLE predictions ADD COLUMN {if_not_exists}"
                "published_academy INTEGER DEFAULT 0"
            )
        logger.info("Colonne published_academy ajoutée.")
    except Exception:
        pass  # déjà présente


# ============================================================
# JOB 1 — ANALYSE DU LENDEMAIN (18h00)
# ============================================================

def job_analyse():
    """
    Publie les prédictions du lendemain sur le canal.
    Filtre : is_value_bet=1, confidence >= 0.52, published_academy=0.
    """
    logger.info("job_analyse: démarrage")
    get_conn, ph, is_postgres, adapt_ddl = _get_db()

    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT id, sport, league, home_team, away_team, match_date,
                   pred_result, confidence, edge, kelly_stake, odd_used,
                   prob_home, prob_draw, prob_away, market
            FROM predictions
            WHERE DATE(match_date) = {ph}
              AND published_academy = 0
            ORDER BY confidence DESC
            LIMIT 10
            """,
            (tomorrow,),
        ).fetchall()

    if not rows:
        logger.info("job_analyse: aucune prédiction pour %s", tomorrow)
        return

    col = ("id", "sport", "league", "home_team", "away_team", "match_date",
           "pred_result", "confidence", "edge", "kelly_stake", "odd_used",
           "prob_home", "prob_draw", "prob_away", "market")
    predictions = [dict(zip(col, r)) for r in rows]

    # ── Header du post ──────────────────────────────────────
    day_str = (date.today() + timedelta(days=1)).strftime("%A %d %B").capitalize()
    header_lines = [
        "🔮 <b>ANALYSE BETMIND — Prédictions du lendemain</b>",
        f"📅 {day_str}",
        f"🎯 {len(predictions)} match(s) analysé(s)",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]
    _send("\n".join(header_lines))

    ids_published = []
    for p in predictions:
        sport        = p["sport"] or "football"
        s_emoji      = SPORT_EMOJI.get(sport, "🏆")
        pred_label   = RESULT_LABELS.get(p["pred_result"], p["pred_result"] or "?")
        conf         = float(p["confidence"] or 0)
        edge         = float(p["edge"] or 0)
        kelly        = float(p["kelly_stake"] or 0)
        odd          = float(p["odd_used"] or 0)
        ph_         = float(p["prob_home"] or 0)
        pd_         = float(p["prob_draw"] or 0)
        pa_         = float(p["prob_away"] or 0)
        is_value    = edge >= 0.05

        lines = [
            f"{s_emoji} <b>{p['home_team']} vs {p['away_team']}</b>",
            f"🏆 {p['league']}  |  📅 {str(p['match_date'])[:10]}",
            "",
            f"📊 <b>Prédiction :</b> {pred_label}",
            f"🎯 <b>Confiance  :</b> {conf:.1%}  {_confidence_label(conf)}",
        ]

        if is_value and odd > 0:
            lines += [
                f"💎 <b>Value Bet   :</b> Edge +{edge:.1%}  @ cote {odd:.2f}",
            ]
        if kelly > 0:
            lines.append(f"💰 <b>Mise Kelly  :</b> {kelly:,.0f} FCFA")

        # Probas
        if sport == "football" and ph_ + pd_ + pa_ > 0:
            lines += [
                "",
                f"<b>Probabilités :</b>",
                f"  🏠 Dom. {ph_:.0%}  |  🤝 Nul {pd_:.0%}  |  ✈️ Ext. {pa_:.0%}",
            ]
        elif sport == "nba" and ph_ + pa_ > 0:
            lines += [
                "",
                f"<b>Probabilités :</b>  🏠 {ph_:.0%}  |  ✈️ {pa_:.0%}",
            ]

        lines += ["", "─" * 28]
        _send("\n".join(lines))
        ids_published.append(p["id"])

    # ── Leçon éducative ─────────────────────────────────────
    _send(_daily_lesson())

    # ── Footer ───────────────────────────────────────────────
    _send(
        "⚠️ <i>Ces prédictions sont générées par un modèle ML. "
        "Le pari sportif comporte des risques. Jouez de manière responsable.</i>\n\n"
        "🤖 BetMind Academy"
    )

    # ── Marquer comme publiés ────────────────────────────────
    if ids_published:
        get_conn2, ph2, _, _ = _get_db()
        placeholders = ", ".join([ph] * len(ids_published))
        with get_conn2() as conn:
            conn.execute(
                f"UPDATE predictions SET published_academy = 1 WHERE id IN ({placeholders})",
                ids_published,
            )
        logger.info("job_analyse: %d prédictions marquées publiées", len(ids_published))


# ============================================================
# JOB 2 — RÉSULTATS DU JOUR (23h00)
# ============================================================

def job_resultats():
    """
    Publie les résultats des prédictions d'aujourd'hui.
    Filtre : match_date = aujourd'hui, outcome IS NOT NULL.
    """
    logger.info("job_resultats: démarrage")
    get_conn, ph, is_postgres, adapt_ddl = _get_db()

    today = date.today().isoformat()

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT id, sport, league, home_team, away_team, match_date,
                   pred_result, confidence, outcome, pnl, odd_used, kelly_stake
            FROM predictions
            WHERE DATE(match_date) = {ph}
              AND outcome IS NOT NULL
            ORDER BY pnl DESC NULLS LAST
            """,
            (today,),
        ).fetchall()

    col = ("id", "sport", "league", "home_team", "away_team", "match_date",
           "pred_result", "confidence", "outcome", "pnl", "odd_used", "kelly_stake")
    settled = [dict(zip(col, r)) for r in rows]

    if not settled:
        logger.info("job_resultats: aucun résultat pour %s", today)
        return

    wins   = [p for p in settled if p["outcome"] == "W"]
    losses = [p for p in settled if p["outcome"] == "L"]
    total_pnl  = sum(float(p["pnl"] or 0) for p in settled)
    win_rate   = len(wins) / len(settled) * 100

    day_str = date.today().strftime("%A %d %B").capitalize()

    lines = [
        "📋 <b>RÉSULTATS DU JOUR — BetMind</b>",
        f"📅 {day_str}",
        "",
        f"✅ Gagnés : {len(wins)}  |  ❌ Perdus : {len(losses)}",
        f"🎯 Taux : {win_rate:.0f}%",
        f"💵 P&L  : <b>{'+' if total_pnl >= 0 else ''}{total_pnl:,.0f} FCFA</b>",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]
    _send("\n".join(lines))

    for p in settled:
        outcome   = p["outcome"]
        outcome_e = "✅" if outcome == "W" else ("❌" if outcome == "L" else "↩️")
        pred_label = RESULT_LABELS.get(p["pred_result"], p["pred_result"] or "?")
        pnl        = float(p["pnl"] or 0)
        odd        = float(p["odd_used"] or 0)
        conf       = float(p["confidence"] or 0)
        sport      = p["sport"] or "football"
        s_emoji    = SPORT_EMOJI.get(sport, "🏆")

        detail = [
            f"{outcome_e} {s_emoji} <b>{p['home_team']} vs {p['away_team']}</b>",
            f"🏆 {p['league']}",
            f"📊 Prédiction : {pred_label}  ({conf:.0%})",
        ]
        if odd > 0:
            detail.append(f"💸 Cote : {odd:.2f}")
        detail.append(
            f"💵 P&L : <b>{'+' if pnl >= 0 else ''}{pnl:,.0f} FCFA</b>"
        )
        detail.append("─" * 28)
        _send("\n".join(detail))

    # ── Commentaire pédagogique ──────────────────────────────
    if win_rate >= 60:
        comment = "🔥 Excellente journée ! Le modèle performe au-dessus de sa moyenne long terme."
    elif win_rate >= 45:
        comment = "📊 Journée correcte. La variance à court terme est normale — l'edge se révèle sur le long terme."
    else:
        comment = ("⚠️ Journée difficile. Rappel : même un modèle edge positif peut perdre sur "
                   "une courte série. L'important est de rester discipliné et de respecter les mises Kelly.")
    _send(comment + "\n\n🤖 BetMind Academy")


# ============================================================
# JOB 3 — BILAN HEBDOMADAIRE (dimanche 20h00)
# ============================================================

def job_bilan_hebdo():
    """
    Publie le bilan de la semaine : W/L, ROI, P&L, ligue la plus rentable.
    """
    logger.info("job_bilan_hebdo: démarrage")
    get_conn, ph, is_postgres, adapt_ddl = _get_db()

    week_ago = (date.today() - timedelta(days=7)).isoformat()

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT sport, league, pred_result, confidence, outcome, pnl, odd_used, kelly_stake
            FROM predictions
            WHERE DATE(match_date) >= {ph}
              AND outcome IS NOT NULL
            """,
            (week_ago,),
        ).fetchall()

    col = ("sport", "league", "pred_result", "confidence", "outcome", "pnl", "odd_used", "kelly_stake")
    bets = [dict(zip(col, r)) for r in rows]

    if not bets:
        _send("📊 <b>Bilan Hebdo BetMind</b>\n\nAucun pari réglé cette semaine.")
        return

    wins   = [b for b in bets if b["outcome"] == "W"]
    losses = [b for b in bets if b["outcome"] == "L"]
    total_pnl  = sum(float(b["pnl"] or 0) for b in bets)
    total_mise = sum(float(b["kelly_stake"] or 0) for b in bets)
    roi        = (total_pnl / total_mise * 100) if total_mise > 0 else 0
    win_rate   = len(wins) / len(bets) * 100

    # Ligue la plus rentable
    from collections import defaultdict
    league_pnl: dict = defaultdict(float)
    for b in bets:
        league_pnl[b["league"] or "?"] += float(b["pnl"] or 0)
    best_league     = max(league_pnl, key=lambda k: league_pnl[k])
    best_league_pnl = league_pnl[best_league]

    # Meilleur pari
    best_bet = max(bets, key=lambda b: float(b["pnl"] or 0))

    # Série en cours
    streak, streak_type = 0, None
    for b in reversed(bets):
        if streak_type is None:
            streak_type = b["outcome"]
        if b["outcome"] == streak_type:
            streak += 1
        else:
            break

    pnl_sign = "+" if total_pnl >= 0 else ""
    roi_sign = "+" if roi >= 0 else ""
    roi_emoji = "📈" if roi >= 0 else "📉"

    week_start = (date.today() - timedelta(days=7)).strftime("%d/%m")
    week_end   = date.today().strftime("%d/%m")

    lines = [
        "📊 <b>BILAN HEBDOMADAIRE — BetMind Academy</b>",
        f"📅 Semaine du {week_start} au {week_end}",
        "",
        f"🎯 Paris joués   : <b>{len(bets)}</b>",
        f"✅ Gagnés        : {len(wins)}",
        f"❌ Perdus        : {len(losses)}",
        f"📉 Taux de réussite : <b>{win_rate:.1f}%</b>",
        "",
        f"{roi_emoji} ROI semaine : <b>{roi_sign}{roi:.2f}%</b>",
        f"💵 P&L total     : <b>{pnl_sign}{total_pnl:,.0f} FCFA</b>",
    ]

    if total_mise > 0:
        lines.append(f"💰 Mise totale   : {total_mise:,.0f} FCFA")

    lines += [
        "",
        f"🏆 Ligue + rentable : {best_league}  ({'+' if best_league_pnl >= 0 else ''}{best_league_pnl:,.0f} FCFA)",
    ]

    if best_bet and float(best_bet.get("pnl") or 0) > 0:
        odd = float(best_bet.get("odd_used") or 0)
        pnl_b = float(best_bet["pnl"])
        lines.append(
            f"🔥 Meilleur pari : {best_bet.get('home_team', '')[:12]}…"
            f"  +{pnl_b:,.0f} FCFA @ {odd:.2f}"
        )

    if streak >= 3:
        streak_emoji = "🔥" if streak_type == "W" else "❄️"
        streak_label = "victoires" if streak_type == "W" else "défaites"
        lines.append(f"{streak_emoji} Série actuelle : {streak} {streak_label} consécutives")

    # Interprétation pédagogique
    lines += ["", "━━━━━━━━━━━━━━━━━━━━━━"]
    if roi >= 5:
        lines.append("🚀 <b>Analyse :</b> Semaine exceptionnelle. L'edge modèle est bien réel à court terme.")
    elif roi >= 0:
        lines.append("✅ <b>Analyse :</b> Semaine positive. Continuez à respecter la discipline Kelly.")
    elif roi >= -5:
        lines.append("⚠️ <b>Analyse :</b> Légèrement négatif. La variance court terme peut effacer 1-2 semaines positives.")
    else:
        lines.append(
            "🔴 <b>Analyse :</b> Semaine difficile. "
            "Le stop-loss est là pour protéger le capital. Pas de surenchère."
        )

    lines += ["", "🤖 BetMind Academy"]
    _send("\n".join(lines))


# ============================================================
# TEST — message de vérification avant activation du scheduler
# ============================================================

def send_test_message() -> bool:
    """Envoie un message de test pour valider la configuration."""
    text = (
        "🤖 <b>BetMind Academy — Test de connexion</b>\n\n"
        f"✅ Bot connecté au canal\n"
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        "Les publications automatiques sont configurées :\n"
        "  • 18h00 → Prédictions du lendemain\n"
        "  • 23h00 → Résultats du jour\n"
        "  • Dimanche 20h00 → Bilan hebdomadaire\n\n"
        "🤖 BetMind Academy"
    )
    ok = _send(text)
    if ok:
        logger.info("Message de test envoyé avec succès sur %s", CHANNEL_ID)
    else:
        logger.error("Échec du message de test — vérifier TELEGRAM_TOKEN et TELEGRAM_CHANNEL_PUBLIC")
    return ok


# ============================================================
# SCHEDULER (APScheduler)
# ============================================================

def start_scheduler():
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="Africa/Abidjan")

    # job_analyse : chaque jour à 18h00
    scheduler.add_job(
        job_analyse,
        CronTrigger(hour=18, minute=0),
        id="analyse",
        name="Prédictions du lendemain",
        misfire_grace_time=300,
    )

    # job_resultats : chaque jour à 23h00
    scheduler.add_job(
        job_resultats,
        CronTrigger(hour=23, minute=0),
        id="resultats",
        name="Résultats du jour",
        misfire_grace_time=300,
    )

    # job_bilan_hebdo : chaque dimanche à 20h00
    scheduler.add_job(
        job_bilan_hebdo,
        CronTrigger(day_of_week="sun", hour=20, minute=0),
        id="bilan_hebdo",
        name="Bilan hebdomadaire",
        misfire_grace_time=600,
    )

    logger.info(
        "Publisher démarré — canal %s | jobs : analyse@18h, résultats@23h, bilan@dim20h",
        CHANNEL_ID,
    )
    scheduler.start()


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import argparse

    os.makedirs("logs", exist_ok=True)

    parser = argparse.ArgumentParser(description="BetMind Telegram Publisher")
    parser.add_argument("--test",      action="store_true", help="Envoie un message de test et quitte")
    parser.add_argument("--analyse",   action="store_true", help="Force job_analyse maintenant")
    parser.add_argument("--resultats", action="store_true", help="Force job_resultats maintenant")
    parser.add_argument("--bilan",     action="store_true", help="Force job_bilan_hebdo maintenant")
    args = parser.parse_args()

    _ensure_published_column()

    if args.test:
        ok = send_test_message()
        sys.exit(0 if ok else 1)
    elif args.analyse:
        job_analyse()
    elif args.resultats:
        job_resultats()
    elif args.bilan:
        job_bilan_hebdo()
    else:
        # Mode normal : test puis scheduler
        if not send_test_message():
            logger.error("Message de test échoué — arrêt.")
            sys.exit(1)
        start_scheduler()
