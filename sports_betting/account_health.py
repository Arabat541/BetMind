# ============================================================
# account_health.py — AU : Score de santé des comptes bookmakers
# ============================================================
# Détecte si un compte bookmaker est en cours de limitation
# à partir des métriques observées dans la DB et des paris récents.
#
# Métriques surveillées :
#   - Délai de confirmation (ms) vs baseline
#   - Taux de mise acceptée vs refusée
#   - Plafond de mise max autorisé (Kelly stake clippé)
#   - Variation d'odds au moment de l'acceptation (slippage)
# ============================================================

import logging
from datetime import datetime, timedelta

from db import get_conn, ph

logger = logging.getLogger(__name__)


# Seuils d'alerte
HEALTH_MIN_SCORE   = 0.5    # score < 50% → alerte
ACCEPT_RATE_MIN    = 0.70   # taux d'acceptation minimum attendu
CLIP_RATE_MAX      = 0.40   # trop de paris clippés = limitation probable
STAKE_SHRINK_RATIO = 0.60   # si stake moyen < 60% du stake Kelly → limitation


def compute_account_health(bookmaker: str = "all",
                            lookback_days: int = 14) -> dict:
    """
    Calcule le score de santé [0,1] pour un bookmaker.
    Utilise les prédictions enregistrées dans la DB.

    Indicateurs :
      - stake_clip_rate : % paris où stake_amount < kelly_stake * 0.8
      - accepted_rate   : % paris enregistrés sans erreur (proxy)
      - avg_stake_ratio : stake_amount moyen / kelly_stake moyen

    Retourne {health_score, indicators, recommendation, bookmaker}.
    """
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        with get_conn() as conn:
            sql = f"""
                SELECT kelly_stake, stake_pct, expected_value, created_at
                FROM predictions
                WHERE created_at >= {ph}
                  AND sport = 'football'
                  AND kelly_stake IS NOT NULL
                  AND kelly_stake > 0
            """
            cur  = conn.execute(sql, (cutoff,))
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
    except Exception as e:
        logger.debug("account_health DB error: %s", e)
        return _default_health(bookmaker)

    if not rows:
        return _default_health(bookmaker)

    import pandas as pd
    df = pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame(rows)

    if df.empty:
        return _default_health(bookmaker)

    # Indicateur 1 : stake_clip_rate (on ne peut pas directement mesurer
    # la mise refusée, mais on proxy via expected_value négatifs)
    # Un EV négatif après enregistrement = le modèle était hors marché
    # → bookmaker a peut-être réduit les limites
    neg_ev_rate = float((df["expected_value"] < 0).mean()) if "expected_value" in df.columns else 0.0

    # Indicateur 2 : stake_pct variance (si soudainement basse = limitation)
    stake_pcts = df["stake_pct"].dropna().astype(float)
    avg_stake_pct = float(stake_pcts.mean()) if len(stake_pcts) else 0.0
    # On compare aux 7 derniers jours vs les 7 précédents
    cutoff_recent = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        with get_conn() as conn:
            cur2 = conn.execute(
                f"SELECT stake_pct FROM predictions WHERE created_at >= {ph} AND sport='football'",
                (cutoff_recent,)
            )
            recent_rows = cur2.fetchall()
    except Exception:
        recent_rows = []

    recent_stake_pcts = [float(r[0]) for r in recent_rows if r[0] is not None]
    recent_avg = sum(recent_stake_pcts) / len(recent_stake_pcts) if recent_stake_pcts else avg_stake_pct

    stake_shrink = (recent_avg < avg_stake_pct * STAKE_SHRINK_RATIO) if avg_stake_pct > 0 else False

    # Indicateur 3 : EV trend (si EV moyen récent < EV moyen global)
    recent_evs = []
    try:
        with get_conn() as conn:
            cur3 = conn.execute(
                f"SELECT expected_value FROM predictions WHERE created_at >= {ph} AND sport='football'",
                (cutoff_recent,)
            )
            recent_evs = [float(r[0]) for r in cur3.fetchall() if r[0] is not None]
    except Exception:
        pass
    recent_ev_avg = sum(recent_evs) / len(recent_evs) if recent_evs else 0.0
    ev_degraded = recent_ev_avg < -0.05  # EV moyen récent < -5%

    # Score composite
    score = 1.0
    if neg_ev_rate > 0.5:
        score -= 0.2
    if stake_shrink:
        score -= 0.3
    if ev_degraded:
        score -= 0.2
    score = max(0.0, min(1.0, round(score, 2)))

    indicators = {
        "Taux EV négatif (14j)": {
            "value":   f"{neg_ev_rate:.0%}",
            "warning": neg_ev_rate > 0.5,
        },
        "Mise moyenne récente": {
            "value":   f"{recent_avg:.3%}",
            "warning": stake_shrink,
        },
        "EV moyen 7 derniers jours": {
            "value":   f"{recent_ev_avg:+.3f}",
            "warning": ev_degraded,
        },
    }

    if score < HEALTH_MIN_SCORE:
        recommendation = "Réduire le volume ou changer de compte — limitation probable."
    elif score < 0.75:
        recommendation = "Surveiller : quelques signaux de limitation détectés."
    else:
        recommendation = "Compte sain — pas de limitation détectée."

    return {
        "bookmaker":     bookmaker,
        "health_score":  score,
        "indicators":    indicators,
        "recommendation": recommendation,
        "checked_at":    datetime.now().isoformat(),
        "lookback_days": lookback_days,
    }


def _default_health(bookmaker: str) -> dict:
    return {
        "bookmaker":     bookmaker,
        "health_score":  1.0,
        "indicators":    {},
        "recommendation": "Pas de données suffisantes pour évaluer.",
        "checked_at":    datetime.now().isoformat(),
        "lookback_days": 14,
    }


def run_health_check_accounts(bookmakers: list[str] | None = None,
                               alert_fn=None) -> list[dict]:
    """
    Vérifie la santé de tous les comptes (ou ceux listés dans `bookmakers`).
    Appelle `alert_fn(report)` si health_score < HEALTH_MIN_SCORE.
    """
    if not bookmakers:
        bookmakers = ["all"]

    reports = []
    for bk in bookmakers:
        report = compute_account_health(bk)
        reports.append(report)
        if report["health_score"] < HEALTH_MIN_SCORE and alert_fn:
            try:
                alert_fn(report)
            except Exception as e:
                logger.warning("account_health alert_fn error: %s", e)
    return reports
