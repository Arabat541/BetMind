# ============================================================
# sentiment_fetcher.py — AP : Sentiment NLP (rumeurs blessures)
# ============================================================
# Scrape les flux RSS de Sky Sports, BBC Sport, The Guardian
# pour détecter les rumeurs de blessures avant publication officielle.
#
# Détection de mots-clés hiérarchisée :
#   OUT      → score -1.0  (joueur absent certain)
#   DOUBTFUL → score -0.5  (incertain)
#   RETURN   → score +0.5  (retour de blessure)
#
# Cache mémoire TTL=1h (signal très temporel).
# Aucune clé API requise — RSS publics.
# ============================================================

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

# ── Sources RSS ──────────────────────────────────────────────
_RSS_FEEDS = [
    ("BBC Sport", "https://feeds.bbci.co.uk/sport/football/rss.xml"),
    ("Sky Sports", "https://www.skysports.com/rss/12040"),         # Football injuries/team news
    ("Guardian",   "https://www.theguardian.com/football/rss"),
    ("RMC Sport",  "https://rmcsport.bfmtv.com/rss/football/"),
]

# ── Patterns NLP ─────────────────────────────────────────────
_PATTERNS_OUT = [
    r"\bruled out\b", r"\bwill miss\b", r"\bwon'?t feature\b",
    r"\bsidelined\b", r"\bout for\b", r"\binjured\b", r"\boutside\b",
    r"\bne jouera pas\b", r"\babsent\b", r"\bbless[eé]\b", r"\bforfeits?\b",
    r"\bout of action\b", r"\bmiss(es|ing)? out\b",
]

_PATTERNS_DOUBTFUL = [
    r"\bdoubtful\b", r"\bfitness concern\b", r"\bfitness doubt\b",
    r"\b50[- ]?50\b", r"\bunlikely to feature\b", r"\btouch and go\b",
    r"\bincertain\b", r"\bdoute\b", r"\bnot (fully )?fit\b",
    r"\blast[- ]minute fitness\b", r"\bin doubt\b",
]

_PATTERNS_RETURN = [
    r"\breturns?\b", r"\bback (in )?training\b", r"\bcleared\b",
    r"\bavailable\b", r"\bfit (and available|again)\b",
    r"\bde retour\b", r"\brecov[eé]r\w+\b",
]

_RE_OUT      = [re.compile(p, re.IGNORECASE) for p in _PATTERNS_OUT]
_RE_DOUBTFUL = [re.compile(p, re.IGNORECASE) for p in _PATTERNS_DOUBTFUL]
_RE_RETURN   = [re.compile(p, re.IGNORECASE) for p in _PATTERNS_RETURN]

# ── Cache mémoire ─────────────────────────────────────────────
_cache: dict = {}          # {cache_key: (result, expiry_ts)}
_CACHE_TTL   = 3600        # 1 heure


def _is_cached(key: str):
    if key in _cache:
        result, exp = _cache[key]
        if time.time() < exp:
            return result
    return None


def _set_cache(key: str, result):
    _cache[key] = (result, time.time() + _CACHE_TTL)


# ── Fetch RSS ─────────────────────────────────────────────────

def _fetch_rss(url: str, timeout: int = 8) -> list[dict]:
    """
    Fetch un flux RSS et retourne la liste des entrées
    avec {title, summary, published}.
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; BetMind/1.0)"
        })
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        # Support RSS 2.0 et Atom
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item") or root.findall(".//atom:entry", ns)

        entries = []
        for item in items:
            title   = (item.findtext("title") or item.findtext("atom:title", namespaces=ns) or "").strip()
            summary = (item.findtext("description") or item.findtext("atom:summary", namespaces=ns) or "").strip()
            pub     = (item.findtext("pubDate") or item.findtext("atom:published", namespaces=ns) or "")
            entries.append({"title": title, "summary": summary, "published": pub})
        return entries
    except Exception as e:
        logger.debug("RSS fetch error %s: %s", url, e)
        return []


# ── Keyword classification ────────────────────────────────────

def _classify_text(text: str) -> tuple[str, float]:
    """
    Retourne (label, score) pour un texte.
    OUT=-1.0, DOUBTFUL=-0.5, RETURN=+0.5, NEUTRAL=0.0
    """
    if any(r.search(text) for r in _RE_OUT):
        return "OUT", -1.0
    if any(r.search(text) for r in _RE_DOUBTFUL):
        return "DOUBTFUL", -0.5
    if any(r.search(text) for r in _RE_RETURN):
        return "RETURN", +0.5
    return "NEUTRAL", 0.0


def _team_matches(team_name: str, text: str) -> bool:
    """Vérifie si le nom d'équipe (ou un mot significatif) apparaît dans le texte."""
    words = [w for w in team_name.split() if len(w) >= 4]
    if not words:
        return team_name.lower() in text.lower()
    return any(w.lower() in text.lower() for w in words)


# ── Public API ────────────────────────────────────────────────

def fetch_injury_sentiment(team_name: str,
                            lookback_hours: int = 48) -> dict:
    """
    Cherche les nouvelles de blessures pour `team_name` dans les flux RSS.

    Retourne :
    {
      "team":          str,
      "sentiment":     float  — moyenne des scores (-1 à +1)
      "injury_count":  int    — nombre d'articles OUT/DOUBTFUL
      "return_count":  int    — nombre d'articles RETURN
      "headlines":     list[dict]  — [{title, label, score, source}]
      "fetched_at":    str
    }
    """
    cache_key = f"injury_{team_name}_{lookback_hours}"
    cached = _is_cached(cache_key)
    if cached is not None:
        return cached

    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    all_headlines = []

    for source_name, rss_url in _RSS_FEEDS:
        entries = _fetch_rss(rss_url)
        for entry in entries:
            full_text = f"{entry['title']} {entry['summary']}"
            if not _team_matches(team_name, full_text):
                continue
            label, score = _classify_text(full_text)
            if label == "NEUTRAL":
                continue
            all_headlines.append({
                "title":  entry["title"][:120],
                "label":  label,
                "score":  score,
                "source": source_name,
                "published": entry.get("published", ""),
            })

    if not all_headlines:
        result = {
            "team":         team_name,
            "sentiment":    0.0,
            "injury_count": 0,
            "return_count": 0,
            "headlines":    [],
            "fetched_at":   datetime.now().isoformat(),
        }
    else:
        scores      = [h["score"] for h in all_headlines]
        sentiment   = round(sum(scores) / len(scores), 4)
        inj_count   = sum(1 for h in all_headlines if h["label"] in ("OUT", "DOUBTFUL"))
        ret_count   = sum(1 for h in all_headlines if h["label"] == "RETURN")
        result = {
            "team":         team_name,
            "sentiment":    sentiment,
            "injury_count": inj_count,
            "return_count": ret_count,
            "headlines":    all_headlines[:10],
            "fetched_at":   datetime.now().isoformat(),
        }
        if inj_count > 0:
            logger.info(
                "Sentiment [%s] : sentiment=%.2f (%d blessures, %d retours)",
                team_name, sentiment, inj_count, ret_count,
            )

    _set_cache(cache_key, result)
    return result


def build_sentiment_features(home_name: str, away_name: str) -> dict:
    """
    Features NLP pour le vecteur ML.

    {
      "home_injury_sentiment": float  — [-1, +1] (négatif = blessures)
      "away_injury_sentiment": float
      "home_injury_count":     int
      "away_injury_count":     int
    }
    """
    try:
        h = fetch_injury_sentiment(home_name)
        a = fetch_injury_sentiment(away_name)
        return {
            "home_injury_sentiment": h["sentiment"],
            "away_injury_sentiment": a["sentiment"],
            "home_injury_count":     float(h["injury_count"]),
            "away_injury_count":     float(a["injury_count"]),
        }
    except Exception as e:
        logger.debug("sentiment features error: %s", e)
        return {
            "home_injury_sentiment": 0.0,
            "away_injury_sentiment": 0.0,
            "home_injury_count":     0.0,
            "away_injury_count":     0.0,
        }


def get_team_injury_news(team_name: str) -> list[dict]:
    """Retourne les titres de blessures pour affichage dashboard."""
    data = fetch_injury_sentiment(team_name)
    return data.get("headlines", [])
