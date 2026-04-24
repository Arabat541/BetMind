# ============================================================
# pdf_report.py — Rapport PDF hebdomadaire automatique (BA)
# Génère un PDF multi-pages avec matplotlib et l'envoie via Telegram.
# Aucune dépendance supplémentaire (matplotlib + reportlab optionnel).
# ============================================================

import logging
import os
import tempfile
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _get_weekly_data(days: int = 7) -> dict:
    """Récupère les données des N derniers jours depuis la DB."""
    from db import get_conn, ph
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        with get_conn() as conn:
            preds = conn.execute(f"""
                SELECT league, sport, pred_result, outcome, kelly_stake, pnl,
                       odd_used, confidence, created_at, market
                FROM predictions
                WHERE created_at >= {ph}
                ORDER BY created_at ASC
            """, (since,)).fetchall()
    except Exception as e:
        logger.warning(f"PDF report DB error: {e}")
        preds = []

    rows = [dict(r) if hasattr(r, "keys") else {
        "league": r[0], "sport": r[1], "pred_result": r[2], "outcome": r[3],
        "kelly_stake": r[4], "pnl": r[5], "odd_used": r[6], "confidence": r[7],
        "created_at": r[8], "market": r[9]
    } for r in preds]

    return {"rows": rows, "since": since}


def _pnl_cumsum(rows: list) -> tuple:
    """Retourne (dates, cumsum_pnl) pour les paris réglés avec mise > 0."""
    settled = [r for r in rows if r.get("outcome") and (r.get("kelly_stake") or 0) > 0]
    dates, cumsum = [], []
    total = 0.0
    for r in settled:
        try:
            total += float(r.get("pnl") or 0)
            dates.append(r["created_at"][:10])
            cumsum.append(round(total, 0))
        except Exception:
            pass
    return dates, cumsum


def _league_stats(rows: list) -> list:
    """Stats par ligue : paris, wins, pnl."""
    stats: dict = {}
    for r in rows:
        if not r.get("outcome") or not (r.get("kelly_stake") or 0) > 0:
            continue
        lg = r.get("league", "?")
        if lg not in stats:
            stats[lg] = {"n": 0, "wins": 0, "pnl": 0.0}
        stats[lg]["n"]    += 1
        stats[lg]["wins"] += 1 if (r.get("pnl") or 0) > 0 else 0
        stats[lg]["pnl"]  += float(r.get("pnl") or 0)
    result = [{"league": k, **v} for k, v in stats.items()]
    result.sort(key=lambda x: x["pnl"], reverse=True)
    return result


def generate_weekly_report(output_path: str | None = None) -> str | None:
    """
    Génère un PDF hebdomadaire avec :
    - Page 1 : courbe P&L cumulé + stats globales
    - Page 2 : tableau P&L par ligue (top 6 + flop 6)
    Retourne le chemin du fichier PDF ou None si erreur.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        logger.error("matplotlib requis pour le rapport PDF")
        return None

    data  = _get_weekly_data(days=7)
    rows  = data["rows"]
    since = data["since"]

    if not rows:
        logger.info("Aucun paris cette semaine — rapport PDF annulé.")
        return None

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        output_path = tmp.name
        tmp.close()

    settled = [r for r in rows if r.get("outcome") and (r.get("kelly_stake") or 0) > 0]
    n_total   = len(rows)
    n_settled = len(settled)
    n_wins    = sum(1 for r in settled if (r.get("pnl") or 0) > 0)
    total_pnl = sum(float(r.get("pnl") or 0) for r in settled)
    win_rate  = (n_wins / n_settled * 100) if n_settled > 0 else 0.0
    roi       = (total_pnl / sum(float(r.get("kelly_stake") or 0)
                                  for r in settled) * 100) if n_settled > 0 else 0.0

    dates, cumsum = _pnl_cumsum(rows)
    league_stats  = _league_stats(rows)

    with PdfPages(output_path) as pdf:
        # ── Page 1 : Courbe P&L + résumé ────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle(
            f"BetMind — Rapport hebdomadaire\n{since} → {datetime.now().strftime('%Y-%m-%d')}",
            fontsize=14, fontweight="bold", y=0.98
        )

        ax1 = axes[0]
        if dates and cumsum:
            color = "#27ae60" if cumsum[-1] >= 0 else "#e74c3c"
            ax1.plot(dates, cumsum, marker="o", markersize=3,
                     linewidth=2, color=color, label="P&L cumulé (FCFA)")
            ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax1.fill_between(dates, cumsum, 0,
                             where=[v >= 0 for v in cumsum],
                             alpha=0.15, color="#27ae60")
            ax1.fill_between(dates, cumsum, 0,
                             where=[v < 0 for v in cumsum],
                             alpha=0.15, color="#e74c3c")
            ax1.set_title("Évolution P&L cumulé (paris réglés avec mise)", fontsize=11)
            ax1.set_xlabel("Date")
            ax1.set_ylabel("FCFA")
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
        else:
            ax1.text(0.5, 0.5, "Aucun pari réglé cette semaine",
                     ha="center", va="center", transform=ax1.transAxes, fontsize=12)
            ax1.set_title("P&L cumulé")

        ax2 = axes[1]
        ax2.axis("off")
        summary = [
            ["Métrique", "Valeur"],
            ["Prédictions générées", str(n_total)],
            ["Paris réglés",          str(n_settled)],
            ["Victoires",             f"{n_wins} ({win_rate:.1f}%)"],
            ["P&L net",               f"{total_pnl:+,.0f} FCFA"],
            ["ROI",                   f"{roi:+.1f}%"],
        ]
        tbl = ax2.table(cellText=summary[1:], colLabels=summary[0],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.4, 2.0)
        for (row_i, col_i), cell in tbl.get_celld().items():
            if row_i == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif col_i == 1 and row_i > 0:
                val = summary[row_i][1]
                if "+" in val:
                    cell.set_facecolor("#d5f5e3")
                elif val.startswith("-"):
                    cell.set_facecolor("#fadbd8")
        ax2.set_title("Résumé de la semaine", fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2 : P&L par ligue ──────────────────────────
        if league_stats:
            fig2, ax3 = plt.subplots(figsize=(10, 6))
            leagues = [s["league"][:20] for s in league_stats]
            pnls    = [s["pnl"] for s in league_stats]
            colors  = ["#27ae60" if p >= 0 else "#e74c3c" for p in pnls]
            bars = ax3.barh(leagues[::-1], pnls[::-1], color=colors[::-1])
            ax3.axvline(0, color="gray", linewidth=0.8)
            ax3.set_title("P&L par ligue (semaine)", fontsize=11)
            ax3.set_xlabel("FCFA")
            for bar, pnl in zip(bars, pnls[::-1]):
                ax3.text(bar.get_width() + (max(abs(p) for p in pnls) * 0.02 if pnls else 1),
                         bar.get_y() + bar.get_height() / 2,
                         f"{pnl:+,.0f}", va="center", fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

    logger.info(f"Rapport PDF généré : {output_path} ({n_settled} paris)")
    return output_path
