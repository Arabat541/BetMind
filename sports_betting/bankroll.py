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
from db import get_conn, raw_conn, ph as _ph

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
        with get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM bankroll")
            if c.fetchone()[0] == 0:
                c.execute(f"""
                    INSERT INTO bankroll (balance, total_bets, wins, losses, roi)
                    VALUES ({_ph}, 0, 0, 0, 0.0)
                """, (INITIAL_BANKROLL,))
                logger.info(f"Bankroll initialisée à {INITIAL_BANKROLL:,.0f} FCFA")

    def get_balance(self) -> float:
        with get_conn() as conn:
            row = conn.execute("SELECT balance FROM bankroll ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else INITIAL_BANKROLL

    def get_today_staked(self) -> float:
        """Retourne la somme des mises déjà enregistrées aujourd'hui (en FCFA)."""
        today = datetime.now().strftime("%Y-%m-%d")
        with get_conn() as conn:
            row = conn.execute(
                f"SELECT COALESCE(SUM(kelly_stake), 0) FROM predictions WHERE DATE(created_at) = {_ph}",
                (today,)
            ).fetchone()
        return float(row[0]) if row else 0.0

    def get_stats(self) -> dict:
        conn = raw_conn()
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
        with get_conn() as conn:
            row = conn.execute(f"""
                SELECT
                    COUNT(*)                                                  AS bets,
                    SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END)      AS settled,
                    SUM(CASE WHEN pnl > 0             THEN 1 ELSE 0 END)      AS wins,
                    SUM(CASE WHEN pnl < 0             THEN 1 ELSE 0 END)      AS losses,
                    COALESCE(SUM(CASE WHEN outcome IS NOT NULL THEN pnl        ELSE 0 END), 0) AS pnl,
                    COALESCE(SUM(CASE WHEN outcome IS NOT NULL THEN kelly_stake ELSE 0 END), 0) AS staked
                FROM predictions
                WHERE DATE(created_at) = {_ph}
            """, (today,)).fetchone()
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
        Opération atomique : rollback complet si une étape échoue.
        """
        try:
            with get_conn() as conn:
                c = conn.cursor()
                c.execute(f"SELECT pred_result, kelly_stake, odd_used FROM predictions WHERE id={_ph}",
                          (prediction_id,))
                row = c.fetchone()
                if not row:
                    return

                pred_result, stake, odd = row[0], row[1], row[2]
                won = (pred_result == outcome)
                pnl = round(stake * (odd - 1), 0) if won else -stake

                c.execute(f"UPDATE predictions SET outcome={_ph}, pnl={_ph} WHERE id={_ph}",
                          (outcome, pnl, prediction_id))

                c.execute("SELECT balance FROM bankroll ORDER BY id DESC LIMIT 1")
                balance = c.fetchone()[0]
                new_balance = balance + pnl

                c.execute(f"""
                    INSERT INTO bankroll (balance, total_bets, wins, losses, roi)
                    SELECT {_ph}, total_bets+1, wins+{_ph}, losses+{_ph}, 0
                    FROM bankroll ORDER BY id DESC LIMIT 1
                """, (new_balance, 1 if won else 0, 0 if won else 1))

            logger.info(f"Bet settled #{prediction_id}: {'WIN' if won else 'LOSS'} | PnL: {pnl:+,.0f} FCFA")

        except Exception as e:
            logger.error(f"settle_bet rollback #{prediction_id}: {e}")
            raise

    def _get_settled_bets(self) -> pd.DataFrame:
        conn = raw_conn()
        df = pd.read_sql_query("SELECT * FROM predictions WHERE outcome IS NOT NULL", conn)
        conn.close()
        return df

    def get_roi_by_confidence(self) -> list:
        """
        ROI, win rate et nombre de paris par tranche de confiance du modèle.
        Tranches : <55%, 55-65%, >65%.
        Utilisé pour vérifier la calibration du modèle.
        """
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT
                    CASE
                        WHEN confidence < 0.55 THEN '<55%'
                        WHEN confidence < 0.65 THEN '55-65%'
                        ELSE '>65%'
                    END AS tier,
                    COUNT(*)                                    AS bets,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)   AS wins,
                    COALESCE(SUM(pnl), 0)                       AS pnl,
                    COALESCE(SUM(kelly_stake), 0)               AS staked
                FROM predictions
                WHERE outcome IS NOT NULL AND kelly_stake > 0
                GROUP BY tier
                ORDER BY MIN(confidence)
            """).fetchall()

        result = []
        for tier, bets, wins, pnl, staked in rows:
            roi = round(float(pnl) / float(staked) * 100, 2) if staked else 0.0
            wr  = round(int(wins or 0) / bets * 100, 1)      if bets   else 0.0
            result.append({
                "tier":     tier,
                "bets":     bets,
                "wins":     int(wins or 0),
                "win_rate": wr,
                "pnl":      round(float(pnl), 0),
                "roi":      roi,
            })
        return result

    def get_brier_score(self) -> dict:
        """
        Brier Score sur les paris réglés : BS = mean((confidence - won)²).
        Mesure la calibration du modèle en production (pas seulement à l'entraînement).
        Références : <0.20 excellent | 0.20-0.25 correct | >0.25 pire que naïf (50/50).
        """
        with get_conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) AS n,
                    AVG(
                        (confidence - CASE WHEN pred_result = outcome THEN 1.0 ELSE 0.0 END) *
                        (confidence - CASE WHEN pred_result = outcome THEN 1.0 ELSE 0.0 END)
                    ) AS brier_score
                FROM predictions
                WHERE outcome IS NOT NULL
                  AND kelly_stake > 0
                  AND confidence IS NOT NULL
            """).fetchone()

        n, score = row
        if not n or score is None:
            return {}
        return {"score": round(float(score), 4), "n": int(n)}

    def get_avg_clv(self) -> dict:
        """
        Closing Line Value moyen sur les paris réglés où odd_closing est disponible.
        CLV = (odd_used / odd_closing - 1) × 100
        CLV > 0 = on a obtenu une meilleure cote que la fermeture → edge réel confirmé.
        CLV < 0 = le marché avait raison contre nous → à surveiller.
        """
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT odd_used, odd_closing
                FROM predictions
                WHERE outcome IS NOT NULL
                  AND kelly_stake > 0
                  AND odd_used    > 0
                  AND odd_closing IS NOT NULL
                  AND odd_closing > 0
            """).fetchall()

        if not rows:
            return {}

        clvs = [(float(r[0]) / float(r[1]) - 1) * 100 for r in rows]
        return {
            "avg_clv":   round(sum(clvs) / len(clvs), 2),
            "beat_rate": round(sum(1 for c in clvs if c > 0) / len(clvs) * 100, 1),
            "n_bets":    len(clvs),
        }

    def get_recent_settled_pnl(self, days: int = 2) -> float:
        """
        P&L total des paris réglés sur les N derniers jours de matchs.
        Utilisé par le stop-loss journalier.
        """
        with get_conn() as conn:
            row = conn.execute(f"""
                SELECT COALESCE(SUM(pnl), 0) FROM predictions
                WHERE outcome IS NOT NULL
                  AND DATE(match_date) >= date('now', {_ph})
            """, (f"-{days} days",)).fetchone()
        return float(row[0]) if row else 0.0

    def get_weekly_stats(self) -> dict:
        """Stats des paris réglés sur les 7 derniers jours."""
        conn = raw_conn()
        df   = pd.read_sql_query("""
            SELECT * FROM predictions
            WHERE outcome IS NOT NULL
              AND DATE(match_date) >= date('now', '-7 days')
            ORDER BY match_date ASC
        """, conn)
        conn.close()

        if df.empty:
            return {}

        total_staked = df["kelly_stake"].sum()
        total_pnl    = df["pnl"].sum()
        wins         = int((df["pnl"] > 0).sum())
        losses       = int((df["pnl"] < 0).sum())
        n            = len(df)

        roi_week = round(total_pnl / total_staked * 100, 2) if total_staked > 0 else 0.0

        # Ligue la plus rentable
        best_league     = None
        best_league_pnl = 0.0
        if "league" in df.columns:
            by_league = df.groupby("league")["pnl"].sum()
            if not by_league.empty:
                best_league     = by_league.idxmax()
                best_league_pnl = round(float(by_league.max()), 0)

        # Meilleure prédiction (plus grosse mise gagnée)
        best_bet = None
        winners  = df[df["pnl"] > 0]
        if not winners.empty:
            best_row = winners.loc[winners["pnl"].idxmax()]
            best_bet = {
                "match": f"{best_row.get('home_team', '?')} vs {best_row.get('away_team', '?')}",
                "pnl":   round(float(best_row["pnl"]), 0),
                "odd":   round(float(best_row.get("odd_used") or 0), 2),
            }

        # Série en cours (W/L)
        streak_type = None
        streak_val  = 0
        for _, row in df.sort_values("match_date", ascending=False).iterrows():
            won = float(row["pnl"]) > 0
            if streak_type is None:
                streak_type = "W" if won else "L"
                streak_val  = 1
            elif (streak_type == "W" and won) or (streak_type == "L" and not won):
                streak_val += 1
            else:
                break

        return {
            "bets":            n,
            "wins":            wins,
            "losses":          losses,
            "win_rate":        round(wins / n * 100, 1) if n > 0 else 0.0,
            "pnl":             round(float(total_pnl), 0),
            "roi":             roi_week,
            "best_league":     best_league,
            "best_league_pnl": best_league_pnl,
            "best_bet":        best_bet,
            "streak_type":     streak_type,
            "streak_val":      streak_val,
        }

    def get_league_recent_stats(self, league: str, n_recent: int = 10) -> dict:
        """
        Stats des N derniers paris réglés pour une ligue donnée.
        Retourne : bets, wins, losses, pnl, roi, consecutive_losses
        """
        with get_conn() as conn:
            rows = conn.execute(f"""
                SELECT pnl, kelly_stake
                FROM predictions
                WHERE outcome IS NOT NULL
                  AND kelly_stake > 0
                  AND league = {_ph}
                ORDER BY match_date DESC
                LIMIT {_ph}
            """, (league, n_recent)).fetchall()

        if not rows:
            return {"bets": 0, "wins": 0, "losses": 0, "pnl": 0.0, "roi": 0.0, "consecutive_losses": 0}

        pnls   = [float(r[0]) for r in rows]
        stakes = [float(r[1]) for r in rows]
        wins   = sum(1 for p in pnls if p > 0)
        losses = len(pnls) - wins
        pnl    = sum(pnls)
        staked = sum(stakes)
        roi    = round(pnl / staked * 100, 2) if staked > 0 else 0.0

        # Série de défaites consécutives (depuis le match le plus récent)
        consec_losses = 0
        for p in pnls:
            if p < 0:
                consec_losses += 1
            else:
                break

        return {
            "bets": len(pnls), "wins": wins, "losses": losses,
            "pnl": round(pnl, 0), "roi": roi,
            "consecutive_losses": consec_losses,
        }

    def is_league_suspended(self, league: str,
                            min_bets: int = 5,
                            max_consec_losses: int = 5,
                            min_roi_threshold: float = -20.0) -> tuple[bool, str]:
        """
        Décide si une ligue doit être suspendue temporairement.

        Critères (sur les 10 derniers paris réglés) :
          - ≥ 5 défaites consécutives → suspension
          - ROI < -20% sur 10 paris → suspension

        Returns:
            (suspended: bool, reason: str)
        """
        stats = self.get_league_recent_stats(league, n_recent=10)

        if stats["bets"] < min_bets:
            return False, ""   # Pas assez de données pour décider

        if stats["consecutive_losses"] >= max_consec_losses:
            reason = (f"Stop-loss ligue : {stats['consecutive_losses']} défaites "
                      f"consécutives en {league}")
            logger.warning(reason)
            return True, reason

        if stats["roi"] < min_roi_threshold:
            reason = (f"Stop-loss ligue : ROI {stats['roi']}% < {min_roi_threshold}% "
                      f"sur {stats['bets']} derniers paris en {league}")
            logger.warning(reason)
            return True, reason

        return False, ""

    def get_peak_bankroll(self) -> float:
        """Retourne le solde maximum jamais atteint (pic historique)."""
        with get_conn() as conn:
            row = conn.execute("SELECT MAX(balance) FROM bankroll").fetchone()
        return float(row[0]) if row and row[0] is not None else INITIAL_BANKROLL

    def get_drawdown_pct(self) -> float:
        """
        Drawdown actuel en % par rapport au pic historique.
        Retourne 0 si on est au-dessus ou égal au pic.
        """
        peak    = self.get_peak_bankroll()
        current = self.get_balance()
        if peak <= 0:
            return 0.0
        dd = (peak - current) / peak * 100
        return round(max(dd, 0.0), 2)

    def is_global_drawdown_exceeded(self, max_drawdown_pct: float = 20.0) -> tuple[bool, float]:
        """
        Vérifie si le drawdown global dépasse le seuil autorisé.
        Retourne (exceeded: bool, current_drawdown_pct: float).
        """
        dd = self.get_drawdown_pct()
        return dd >= max_drawdown_pct, dd

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