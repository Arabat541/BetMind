# ============================================================
# dixon_coles.py — Correction Dixon-Coles sur Poisson bivarié
#
# Dixon & Coles (1997) : corrige la sous-estimation des scores
# faibles (0-0, 1-0, 0-1, 1-1) par la distribution de Poisson.
#
# Le paramètre ρ (rho) mesure la dépendance entre les buts des
# deux équipes — typiquement dans [-0.15, 0] pour le football.
# ρ = 0 → Poisson pur. ρ < 0 → correction activée.
# ============================================================

import json
import logging
import os
from math import exp, log, factorial

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

RHO_CACHE_PATH = os.path.join("data", "dc_rho.json")
MAX_GOALS = 8


# ════════════════════════════════════════════════════════════
# CORRECTION τ (tau)
# ════════════════════════════════════════════════════════════

def _tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """
    Facteur de correction Dixon-Coles pour le score (x, y).
    Vaut 1 pour tous les scores sauf (0,0), (1,0), (0,1), (1,1).
    """
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    if x == 1 and y == 0:
        return 1.0 + lam_a * rho
    if x == 0 and y == 1:
        return 1.0 + lam_h * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam) * (lam ** k) / factorial(k)


def dc_score_matrix(lam_h: float, lam_a: float, rho: float,
                    max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Matrice [max_goals+1 x max_goals+1] des probabilités P(home=i, away=j).
    Ligne i = buts domicile, colonne j = buts extérieur.
    """
    mat = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = _poisson_pmf(i, lam_h) * _poisson_pmf(j, lam_a)
            mat[i, j] = p * _tau(i, j, lam_h, lam_a, rho)
    # Normalise (la somme peut légèrement dériver avec la correction)
    total = mat.sum()
    return mat / total if total > 0 else mat


def dc_1x2(lam_h: float, lam_a: float, rho: float) -> dict:
    """Probabilités 1X2 corrigées Dixon-Coles."""
    mat = dc_score_matrix(lam_h, lam_a, rho)
    prob_home = float(np.tril(mat, -1).sum())   # i > j
    prob_draw = float(np.trace(mat))             # i == j
    prob_away = float(np.triu(mat, 1).sum())     # i < j
    total = prob_home + prob_draw + prob_away
    if total <= 0:
        return {"prob_home": 1/3, "prob_draw": 1/3, "prob_away": 1/3}
    return {
        "prob_home": round(prob_home / total, 4),
        "prob_draw": round(prob_draw / total, 4),
        "prob_away": round(prob_away / total, 4),
    }


def dc_over_under(lam_h: float, lam_a: float, rho: float,
                  threshold: float = 2.5) -> dict:
    """Probabilités Over/Under corrigées DC."""
    mat = dc_score_matrix(lam_h, lam_a, rho)
    prob_over = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            if i + j > threshold:
                prob_over += mat[i, j]
    prob_under = 1.0 - prob_over
    return {
        "prob_over":  round(prob_over,  4),
        "prob_under": round(prob_under, 4),
    }


def dc_correct_scores(lam_h: float, lam_a: float, rho: float,
                      top_n: int = 12) -> list[dict]:
    """
    Retourne les top_n scores les plus probables avec leur probabilité.
    Format : [{"score": "1-0", "prob": 0.142, "lam_h": ..., "lam_a": ...}, ...]
    """
    mat = dc_score_matrix(lam_h, lam_a, rho)
    scores = []
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            scores.append({
                "score": f"{i}-{j}",
                "home":  i,
                "away":  j,
                "prob":  round(float(mat[i, j]), 5),
            })
    scores.sort(key=lambda s: s["prob"], reverse=True)
    return scores[:top_n]


# ════════════════════════════════════════════════════════════
# EXACT GOALS — AY
# ════════════════════════════════════════════════════════════

def dc_exact_goals(lam_h: float, lam_a: float, rho: float) -> dict:
    """
    P(total goals = 0 / 1 / 2 / 3 / 4+) corrigé Dixon-Coles.
    Retourne dict {"0": p, "1": p, "2": p, "3": p, "4+": p}.
    """
    mat = dc_score_matrix(lam_h, lam_a, rho)
    result: dict = {}
    for n in range(4):
        p = sum(mat[i, j]
                for i in range(MAX_GOALS + 1)
                for j in range(MAX_GOALS + 1)
                if i + j == n)
        result[str(n)] = round(float(p), 5)
    p4plus = sum(mat[i, j]
                 for i in range(MAX_GOALS + 1)
                 for j in range(MAX_GOALS + 1)
                 if i + j >= 4)
    result["4+"] = round(float(p4plus), 5)
    return result


# ════════════════════════════════════════════════════════════
# HALF TIME / FULL TIME — AZ
# ════════════════════════════════════════════════════════════

def dc_htft(lam_h: float, lam_a: float, rho: float,
            ht_ratio: float = 0.50) -> dict:
    """
    Probabilités HT/FT pour les 9 combinaisons (HH, HD, HA, DH, DD, DA, AH, AD, AA).
    Modélise chaque mi-temps par une distribution Poisson indépendante.
    ht_ratio = part des buts attendus en première mi-temps (défaut 50%).
    """
    lh1, la1 = lam_h * ht_ratio,        lam_a * ht_ratio
    lh2, la2 = lam_h * (1 - ht_ratio),  lam_a * (1 - ht_ratio)

    def _res(h: int, a: int) -> str:
        return "H" if h > a else ("D" if h == a else "A")

    probs: dict = {f"{r1}{r2}": 0.0 for r1 in "HDA" for r2 in "HDA"}
    cap = min(MAX_GOALS, 6)  # 7^4 = 2401 combos — rapide

    for h1 in range(cap + 1):
        ph1 = _poisson_pmf(h1, lh1)
        for a1 in range(cap + 1):
            p_ht = ph1 * _poisson_pmf(a1, la1)
            ht_r = _res(h1, a1)
            for h2 in range(cap + 1):
                ph2 = _poisson_pmf(h2, lh2)
                for a2 in range(cap + 1):
                    p_h2 = ph2 * _poisson_pmf(a2, la2)
                    probs[f"{ht_r}{_res(h1 + h2, a1 + a2)}"] += p_ht * p_h2

    total = sum(probs.values())
    return {k: round(v / total, 5) for k, v in probs.items()} if total > 0 else probs


# ════════════════════════════════════════════════════════════
# CALIBRATION DE ρ
# ════════════════════════════════════════════════════════════

def _dc_log_likelihood(rho: float, matches: list[tuple]) -> float:
    """
    Log-vraisemblance négative des matchs historiques pour un rho donné.
    matches : liste de (home_goals, away_goals, lam_h, lam_a)
    """
    ll = 0.0
    for hg, ag, lh, la in matches:
        p_h = _poisson_pmf(int(hg), lh)
        p_a = _poisson_pmf(int(ag), la)
        tau = _tau(int(hg), int(ag), lh, la, rho)
        val = p_h * p_a * tau
        if val <= 0:
            ll += -20.0   # pénalité log(ε) pour éviter log(0)
        else:
            ll += log(val)
    return -ll   # minimisation → négatif


def calibrate_rho(matches: list[tuple]) -> float:
    """
    Calibre ρ sur des matchs historiques par minimisation MLE.
    matches : [(home_goals, away_goals, lam_h, lam_a), ...]
    Retourne rho ∈ [-0.3, 0].
    """
    if len(matches) < 50:
        logger.warning(f"Calibration rho : seulement {len(matches)} matchs, rho=−0.08 par défaut")
        return -0.08

    result = minimize_scalar(
        lambda r: _dc_log_likelihood(r, matches),
        bounds=(-0.3, 0.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    rho = round(float(result.x), 5)
    logger.info(f"Dixon-Coles ρ calibré : {rho:.5f} sur {len(matches)} matchs")
    return rho


# ════════════════════════════════════════════════════════════
# PERSISTANCE ρ
# ════════════════════════════════════════════════════════════

def save_rho(rho: float, league: str = "global"):
    os.makedirs("data", exist_ok=True)
    data = {}
    if os.path.exists(RHO_CACHE_PATH):
        try:
            with open(RHO_CACHE_PATH) as f:
                data = json.load(f)
        except Exception:
            pass
    data[league] = rho
    with open(RHO_CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_rho(league: str = "global", default: float = -0.08) -> float:
    if not os.path.exists(RHO_CACHE_PATH):
        return default
    try:
        with open(RHO_CACHE_PATH) as f:
            data = json.load(f)
        return float(data.get(league, data.get("global", default)))
    except Exception:
        return default


# ════════════════════════════════════════════════════════════
# CALIBRATION DEPUIS CSV FOOTBALL-DATA
# ════════════════════════════════════════════════════════════

def calibrate_from_csv(all_df, avg_goals: float = 2.5) -> float:
    """
    Calibre ρ depuis le DataFrame historique (train_from_csv.py format).
    Construit les lambdas Poisson approximatifs pour chaque match.
    """
    import pandas as pd
    half_avg = avg_goals / 2.0
    matches = []

    needed = ["FTHG", "FTAG", "HomeTeam", "AwayTeam", "Date"]
    if not all(c in all_df.columns for c in needed):
        return -0.08

    all_df = all_df.dropna(subset=needed).copy()
    all_df = all_df.sort_values("Date").reset_index(drop=True)

    # Calcul des lambdas glissants (10 derniers matchs)
    for i, row in all_df.iterrows():
        home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        past = all_df[(
            ((all_df["HomeTeam"] == home) | (all_df["AwayTeam"] == home)) &
            (all_df["Date"] < date)
        )].tail(10)
        past_a = all_df[(
            ((all_df["HomeTeam"] == away) | (all_df["AwayTeam"] == away)) &
            (all_df["Date"] < date)
        )].tail(10)
        if len(past) < 5 or len(past_a) < 5:
            continue

        def gf_avg(df, team):
            gf = [int(r["FTHG"]) if r["HomeTeam"] == team else int(r["FTAG"])
                  for _, r in df.iterrows()]
            return sum(gf) / len(gf) if gf else half_avg

        def ga_avg(df, team):
            ga = [int(r["FTAG"]) if r["HomeTeam"] == team else int(r["FTHG"])
                  for _, r in df.iterrows()]
            return sum(ga) / len(ga) if ga else half_avg

        h_att = gf_avg(past,   home) / half_avg
        h_def = ga_avg(past,   home) / half_avg
        a_att = gf_avg(past_a, away) / half_avg
        a_def = ga_avg(past_a, away) / half_avg

        lam_h = max(h_att * a_def * half_avg, 0.1)
        lam_a = max(a_att * h_def * half_avg, 0.1)
        matches.append((int(row["FTHG"]), int(row["FTAG"]), lam_h, lam_a))

        if len(matches) >= 3000:   # cap pour la vitesse
            break

    return calibrate_rho(matches)


if __name__ == "__main__":
    # Test rapide
    lh, la, rho = 1.5, 1.1, -0.08
    probs = dc_1x2(lh, la, rho)
    print(f"1X2 DC  : {probs}")
    ou = dc_over_under(lh, la, rho)
    print(f"O/U DC  : {ou}")
    scores = dc_correct_scores(lh, la, rho, top_n=6)
    print("Top scores :")
    for s in scores:
        print(f"  {s['score']} → {s['prob']:.1%}")
