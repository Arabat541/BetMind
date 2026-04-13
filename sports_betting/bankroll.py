# ============================================================
# bankroll.py — Kelly Criterion + Gestion de bankroll
# ============================================================

import logging
import sqlite3
import pandas as pd
from datetime import datetime
from config import (
    KELLY_FRACTION, KELLY_FRACTION_DRAW, MAX_BET_PCT, INITIAL_BANKROLL, DB_PATH
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# KELLY CRITERION
# ════════════════════════════════════════════════════════════

def kelly_stake(prob: float, odd: float, bankroll: float,
                fraction: float = KELLY_FRACTION,
                max_pct: float = MAX_BET_PCT) -> dict:
    """
    Calcule la mise optimale selon le critère de Kelly fractionnaire.

    Args:
        prob      : probabilité estimée par le modèle (0-1)
        odd       : cote décimale du bookmaker (ex: 1.85)
        bankroll  : capital disponible en FCFA
        fraction  : fraction Kelly (0.25 = Kelly/4, conservateur)
        max_pct   : plafond de mise en % du bankroll

    Returns:
        dict avec stake_pct, stake_amount, expected_value, kelly_raw
    """
    if odd <= 1.0 or prob <= 0 or prob >= 1:
        return {"stake_pct": 0, "stake_amount": 0, "expected_value": 0, "kelly_raw": 0}

    b = odd - 1  # profit net pour 1 unité misée
    q = 1 - prob

    # Formule Kelly : f* = (bp - q) / b
    kelly_raw = (b * prob - q) / b

    if kelly_raw <= 0:
        # Kelly négatif → pas de value, ne pas miser
        return {"stake_pct": 0, "stake_amount": 0, "expected_value": 0, "kelly_raw": round(kelly_raw, 4)}

    # Kelly fractionnaire + plafonnement
    stake_pct = min(kelly_raw * fraction, max_pct)
    stake_amount = round(bankroll * stake_pct)

    # Expected Value
    ev = prob * b - q
    profit_expected = stake_amount * ev

    return {
        "kelly_raw":       round(kelly_raw, 4),
        "stake_pct":       round(stake_pct, 4),
        "stake_amount":    stake_amount,     # FCFA
        "expected_value":  round(ev, 4),
        "profit_expected": round(profit_expected, 0),
    }


def _is_draw_bet(result_name: str) -> bool:
    """Détecte si le pari porte sur le nul (outcome le plus imprévisible)."""
    return result_name.strip().lower() in ("draw", "d", "nul", "draw win")


def recommended_stake(signal: dict, bankroll: float) -> dict:
    """
    Calcule la mise recommandée pour un signal de pari.
    Utilise le meilleur value bet s'il existe, sinon le pred principal.
    Les paris sur le nul reçoivent un Kelly réduit (KELLY_FRACTION_DRAW).
    """
    value_bets = signal.get("value_bets", [])

    if value_bets:
        vb = value_bets[0]  # meilleur edge
        fraction = KELLY_FRACTION_DRAW if _is_draw_bet(vb.get("result_name", "")) else KELLY_FRACTION
        result = kelly_stake(
            prob=vb["p_model"],
            odd=vb["odd"],
            bankroll=bankroll,
            fraction=fraction,
        )
        result["bet_on"]    = vb["result_name"]
        result["odd"]       = vb["odd"]
        result["edge"]      = vb["edge"]
        result["is_value"]  = True
    else:
        # Pas de value bet → mise minimale de conviction
        conf = signal.get("confidence", 0)
        odd  = signal.get("odd_used") or 1.0
        pred_name = signal.get("pred_name", "")
        fraction = KELLY_FRACTION_DRAW if _is_draw_bet(pred_name) else KELLY_FRACTION
        result = kelly_stake(prob=conf, odd=odd, bankroll=bankroll, fraction=fraction)
        result["bet_on"]   = pred_name
        result["odd"]      = odd
        result["edge"]     = 0.0
        result["is_value"] = False

    result["bankroll"] = bankroll
    return result


# ════════════════════════════════════════════════════════════
# BANKROLL TRACKER
# ════════════════════════════════════════════════════════════

class BankrollTracker:
    """
    Suit l'évolution de la bankroll au fil des paris.
    Persiste en SQLite.
    """

    def __init__(self):
        self._ensure_initial_balance()

    def _ensure_initial_balance(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM bankroll")
        if c.fetchone()[0] == 0:
            c.execute("""
                INSERT INTO bankroll (balance, total_bets, wins, losses, roi)
                VALUES (?, 0, 0, 0, 0.0)
            """, (INITIAL_BANKROLL,))
            conn.commit()
            logger.info(f"Bankroll initialisée à {INITIAL_BANKROLL:,.0f} FCFA")
        conn.close()

    def get_balance(self) -> float:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT balance FROM bankroll ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        return row[0] if row else INITIAL_BANKROLL

    def get_today_staked(self) -> float:
        """Retourne la somme des mises déjà enregistrées aujourd'hui (en FCFA)."""
        today = datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT COALESCE(SUM(kelly_stake), 0) FROM predictions WHERE DATE(created_at) = ?",
            (today,)
        ).fetchone()
        conn.close()
        return float(row[0]) if row else 0.0

    def get_stats(self) -> dict:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM bankroll ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        cols = [d[0] for d in c.description]
        conn.close()
        if not row:
            return {}
        stats = dict(zip(cols, row))

        # Calcul ROI global depuis les prédictions
        df = self._get_settled_bets()
        if not df.empty:
            total_staked = df["kelly_stake"].sum()
            total_pnl    = df["pnl"].sum()
            stats["roi"] = round(total_pnl / total_staked * 100, 2) if total_staked > 0 else 0
            stats["total_bets"]  = len(df)
            stats["wins"]        = int((df["pnl"] > 0).sum())
            stats["losses"]      = int((df["pnl"] < 0).sum())
            stats["win_rate"]    = round(stats["wins"] / len(df) * 100, 1)
            stats["total_pnl"]   = round(total_pnl, 0)
        return stats

    def get_today_stats(self) -> dict:
        """Stats des paris d'aujourd'hui uniquement."""
        today = datetime.now().strftime("%Y-%m-%d")
        conn  = sqlite3.connect(DB_PATH)
        row   = conn.execute("""
            SELECT
                COUNT(*)                                                  AS bets,
                SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END)      AS settled,
                SUM(CASE WHEN pnl > 0             THEN 1 ELSE 0 END)      AS wins,
                SUM(CASE WHEN pnl < 0             THEN 1 ELSE 0 END)      AS losses,
                COALESCE(SUM(CASE WHEN outcome IS NOT NULL THEN pnl        ELSE 0 END), 0) AS pnl,
                COALESCE(SUM(CASE WHEN outcome IS NOT NULL THEN kelly_stake ELSE 0 END), 0) AS staked
            FROM predictions
            WHERE DATE(created_at) = ?
        """, (today,)).fetchone()
        conn.close()
        bets, settled, wins, losses, pnl, staked = row or (0, 0, 0, 0, 0, 0)
        settled  = int(settled  or 0)
        staked   = float(staked or 0)
        pnl      = float(pnl    or 0)
        return {
            "bets":     int(bets     or 0),
            "settled":  settled,
            "wins":     int(wins     or 0),
            "losses":   int(losses   or 0),
            "pnl":      round(pnl,   0),
            "staked":   round(staked, 0),
            "roi":      round(pnl / staked * 100, 2) if staked > 0 else 0.0,
            "win_rate": round(int(wins or 0) / settled * 100, 1) if settled > 0 else 0.0,
        }

    def settle_bet(self, prediction_id: int, outcome: str):
        """
        Enregistre le résultat réel d'un pari.
        outcome : "H", "D" ou "A"
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT pred_result, kelly_stake, odd_used FROM predictions WHERE id=?",
                  (prediction_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return

        pred_result, stake, odd = row
        won = (pred_result == outcome)
        pnl = round(stake * (odd - 1), 0) if won else -stake

        c.execute("UPDATE predictions SET outcome=?, pnl=? WHERE id=?",
                  (outcome, pnl, prediction_id))

        # Mise à jour balance
        c.execute("SELECT balance FROM bankroll ORDER BY id DESC LIMIT 1")
        balance = c.fetchone()[0]
        new_balance = balance + pnl

        c.execute("""
            INSERT INTO bankroll (balance, total_bets, wins, losses, roi)
            SELECT ?, total_bets+1, wins+?, losses+?, 0
            FROM bankroll ORDER BY id DESC LIMIT 1
        """, (new_balance, 1 if won else 0, 0 if won else 1))

        conn.commit()
        conn.close()
        logger.info(f"Bet settled #{prediction_id}: {'WIN' if won else 'LOSS'} | PnL: {pnl:+,.0f} FCFA")

    def _get_settled_bets(self) -> pd.DataFrame:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM predictions WHERE outcome IS NOT NULL", conn
        )
        conn.close()
        return df

    def performance_summary(self) -> pd.DataFrame:
        """Résumé de performance par sport/ligue."""
        df = self._get_settled_bets()
        if df.empty:
            return pd.DataFrame()
        summary = df.groupby(["sport", "league"]).agg(
            bets=("id", "count"),
            wins=("pnl", lambda x: (x > 0).sum()),
            total_pnl=("pnl", "sum"),
            avg_stake=("kelly_stake", "mean"),
        ).reset_index()
        summary["win_rate"] = (summary["wins"] / summary["bets"] * 100).round(1)
        summary["roi"]      = (summary["total_pnl"] / (summary["avg_stake"] * summary["bets"]) * 100).round(2)
        return summary