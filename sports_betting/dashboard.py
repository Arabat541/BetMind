# ============================================================
# dashboard.py — Flask Dashboard v2.0
# ============================================================

import json
import logging
import math
import os
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, request
from flask.json.provider import DefaultJSONProvider
from data_fetcher import get_all_predictions
from bankroll import BankrollTracker
from config import FLASK_PORT, FLASK_DEBUG
from db import get_conn, raw_conn, ph as _ph, is_postgres


def _sanitize_nan(obj):
    """Remplace récursivement NaN/Inf par None pour un JSON valide."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nan(v) for v in obj]
    return obj


class _NaNSafeProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        return super().dumps(_sanitize_nan(obj), **kwargs)


app = Flask(__name__)
app.json_provider_class = _NaNSafeProvider
app.json = _NaNSafeProvider(app)
tracker = BankrollTracker()
logger  = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ════════════════════════════════════════════════════════════

TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BetMind — Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {
    --bg: #080c10; --surface: #0e1520; --surface2: #141d2e;
    --border: #1e2d45; --accent: #00d4ff; --accent2: #ff6b2b;
    --gold: #f5c842; --green: #00e676; --red: #ff4757;
    --purple: #a855f7; --teal: #14b8a6;
    --text: #e8eef5; --muted: #5a7a99;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg); color: var(--text);
    background-image:
      linear-gradient(rgba(0,212,255,.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,.03) 1px, transparent 1px);
    background-size: 40px 40px;
  }
  header {
    padding: 16px 32px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    background: rgba(8,12,16,.9); backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
  }
  .logo { font-family: 'Bebas Neue'; font-size: 28px; color: var(--accent); letter-spacing: 2px; }
  .tag { font-family: 'JetBrains Mono'; font-size: 11px; color: var(--muted);
         background: var(--surface2); padding: 3px 8px; border-radius: 4px; }
  .live { width: 8px; height: 8px; background: var(--green);
          border-radius: 50%; margin-left: auto; box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.4 } }

  main { max-width: 1400px; margin: 0 auto; padding: 32px 24px; }
  section-title {
    font-family: 'Bebas Neue'; font-size: 16px; letter-spacing: 1px;
    color: var(--muted); text-transform: uppercase; margin-bottom: 12px;
    display: block;
  }

  /* KPI Cards */
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }
  .kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
    transition: border-color .2s;
  }
  .kpi-card:hover { border-color: var(--accent); }
  .kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 8px; }
  .kpi-value { font-family: 'Bebas Neue'; font-size: 36px; line-height: 1; }
  .kpi-value.accent  { color: var(--accent); }
  .kpi-value.gold    { color: var(--gold); }
  .kpi-value.green   { color: var(--green); }
  .kpi-value.red     { color: var(--red); }
  .kpi-value.purple  { color: var(--purple); }
  .kpi-sub { font-size: 12px; color: var(--muted); margin-top: 4px; font-family: 'JetBrains Mono'; }

  /* Models status */
  .models-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }
  .model-pill {
    display: flex; align-items: center; gap: 8px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; font-size: 12px;
    font-family: 'JetBrains Mono';
  }
  .model-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .model-dot.loaded   { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .model-dot.missing  { background: var(--muted); }
  .model-dot.ensemble { background: var(--purple); box-shadow: 0 0 6px var(--purple); }

  /* Charts */
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 32px; }
  .charts-row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 32px; }
  .chart-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
  }
  .chart-title { font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
                 color: var(--muted); margin-bottom: 16px; }

  /* Sharp money */
  .sharp-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
  .sharp-card {
    background: var(--surface2); border-radius: 8px; padding: 12px 14px; text-align: center;
  }
  .sharp-val { font-family: 'Bebas Neue'; font-size: 28px; }
  .sharp-lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-top: 2px; }

  /* Table */
  .table-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
  }
  .table-header {
    padding: 16px 20px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
  }
  .table-title { font-family: 'Bebas Neue'; font-size: 18px; letter-spacing: 1px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    padding: 10px 16px; text-align: left;
    font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
    color: var(--muted); background: var(--surface2);
    font-weight: 500;
  }
  td { padding: 12px 16px; border-top: 1px solid var(--border); }
  tr:hover td { background: rgba(0,212,255,.03); }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-family: 'JetBrains Mono'; font-weight: 500;
  }
  .badge-value    { background: rgba(245,200,66,.15);  color: var(--gold); }
  .badge-signal   { background: rgba(0,212,255,.1);    color: var(--accent); }
  .badge-win      { background: rgba(0,230,118,.12);   color: var(--green); }
  .badge-loss     { background: rgba(255,71,87,.12);   color: var(--red); }
  .badge-h        { background: rgba(0,212,255,.1);    color: var(--accent); }
  .badge-d        { background: rgba(90,122,153,.2);   color: var(--muted); }
  .badge-a        { background: rgba(255,107,43,.12);  color: var(--accent2); }
  .badge-over     { background: rgba(0,230,118,.12);   color: var(--green); }
  .badge-under    { background: rgba(255,71,87,.12);   color: var(--red); }
  .badge-btts     { background: rgba(20,184,166,.12);  color: var(--teal); }
  .badge-no-btts  { background: rgba(90,122,153,.2);   color: var(--muted); }
  .badge-ah       { background: rgba(168,85,247,.12);  color: var(--purple); }
  .badge-market-1x2   { background: rgba(0,212,255,.08);   color: var(--accent);  padding:2px 6px; border-radius:3px; font-size:10px; font-family:'JetBrains Mono'; }
  .badge-market-ou    { background: rgba(0,230,118,.08);   color: var(--green);   padding:2px 6px; border-radius:3px; font-size:10px; font-family:'JetBrains Mono'; }
  .badge-market-btts  { background: rgba(20,184,166,.08);  color: var(--teal);    padding:2px 6px; border-radius:3px; font-size:10px; font-family:'JetBrains Mono'; }
  .badge-market-ah    { background: rgba(168,85,247,.08);  color: var(--purple);  padding:2px 6px; border-radius:3px; font-size:10px; font-family:'JetBrains Mono'; }
  .prob-bar { display: flex; gap: 3px; align-items: center; }
  .prob-bar span { height: 6px; border-radius: 3px; display: inline-block; min-width: 2px; }
  .refresh-btn {
    font-family: 'JetBrains Mono'; font-size: 11px;
    background: var(--surface2); color: var(--accent);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 12px; cursor: pointer; transition: all .2s;
  }
  .refresh-btn:hover { background: var(--accent); color: var(--bg); }
  select {
    background: var(--surface2); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 4px 8px; font-size: 13px;
  }
  @media (max-width: 900px) {
    .charts-row, .charts-row-3 { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">BetMind</div>
  <span class="tag">AI Prediction Engine v2.0</span>
  <span class="tag">⚽ 1X2 | ⚖️ O/U 2.5 | 🎯 BTTS | 🏀 NBA | 🔰 AH -0.5</span>
  <div style="margin-left:auto;display:flex;align-items:center;gap:12px;">
    <span style="font-size:12px;color:var(--muted);font-family:'JetBrains Mono';" id="updated">—</span>
    <div class="live"></div>
  </div>
</header>

<main>

  <!-- KPI -->
  <div class="kpi-grid" id="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">💰 Bankroll</div>
      <div class="kpi-value accent" id="kpi-balance">—</div>
      <div class="kpi-sub">FCFA</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">📊 Paris totaux</div>
      <div class="kpi-value gold" id="kpi-total">—</div>
      <div class="kpi-sub" id="kpi-sub-total">—</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">✅ Win Rate</div>
      <div class="kpi-value green" id="kpi-winrate">—</div>
      <div class="kpi-sub" id="kpi-sub-wr">—</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">📈 ROI</div>
      <div class="kpi-value" id="kpi-roi" style="color:var(--green)">—</div>
      <div class="kpi-sub" id="kpi-pnl">—</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">🔥 Value Bets</div>
      <div class="kpi-value gold" id="kpi-vb">—</div>
      <div class="kpi-sub">détectés</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">🎯 Brier Score</div>
      <div class="kpi-value purple" id="kpi-brier">—</div>
      <div class="kpi-sub" id="kpi-sub-brier">calibration modèle</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">📐 CLV Moyen</div>
      <div class="kpi-value" id="kpi-clv" style="color:var(--muted)">—</div>
      <div class="kpi-sub" id="kpi-sub-clv">closing line value</div>
    </div>
  </div>

  <!-- Models status -->
  <div style="margin-bottom:24px;">
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:10px;">
      Modèles chargés
    </div>
    <div class="models-row" id="models-row">
      <div class="model-pill">
        <div class="model-dot missing" id="dot-ensemble"></div>
        <span id="lbl-ensemble">Football Ensemble</span>
      </div>
      <div class="model-pill">
        <div class="model-dot missing" id="dot-football"></div>
        <span id="lbl-football">Football XGB</span>
      </div>
      <div class="model-pill">
        <div class="model-dot missing" id="dot-ou"></div>
        <span id="lbl-ou">Over/Under</span>
      </div>
      <div class="model-pill">
        <div class="model-dot missing" id="dot-btts"></div>
        <span id="lbl-btts">BTTS</span>
      </div>
      <div class="model-pill">
        <div class="model-dot missing" id="dot-nba"></div>
        <span id="lbl-nba">NBA</span>
      </div>
    </div>
  </div>

  <!-- Exposition journalière -->
  <div class="chart-card" style="margin-bottom:16px;">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
      <span class="chart-title" style="margin:0">⚡ Exposition journalière</span>
      <span id="exposure-pct" style="font-family:'JetBrains Mono';font-size:13px;color:var(--muted)">—</span>
    </div>
    <div style="background:var(--surface2);border-radius:6px;height:10px;overflow:hidden;margin-bottom:8px;">
      <div id="exposure-bar" style="height:100%;width:0%;border-radius:6px;background:var(--green);transition:width .6s,background .6s;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono';font-size:11px;color:var(--muted);">
      <span id="exposure-staked">0 FCFA misés ce jour</span>
      <span id="exposure-limit">Limite : — FCFA</span>
    </div>
  </div>

  <!-- Bankroll chart -->
  <div class="chart-card" style="margin-bottom:16px;">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
      <span class="chart-title" style="margin:0">Évolution bankroll</span>
      <span id="bankroll-delta" style="font-family:'JetBrains Mono';font-size:12px;color:var(--muted)">—</span>
    </div>
    <canvas id="chartBankroll" height="100"></canvas>
  </div>

  <!-- Charts row: Répartition + Confiance + Marchés -->
  <div class="charts-row-3" style="margin-bottom:16px;">
    <div class="chart-card">
      <div class="chart-title">Répartition 1X2</div>
      <canvas id="chartResult" height="180"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">Confiance distribution</div>
      <canvas id="chartConf" height="180"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">Marchés couverts</div>
      <canvas id="chartMarket" height="180"></canvas>
    </div>
  </div>

  <!-- ROI par tranche de confiance -->
  <div class="chart-card" style="margin-bottom:16px;" id="conf-roi-card">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
      <span class="chart-title" style="margin:0">ROI par tranche de confiance</span>
      <span id="conf-calibration" style="font-family:'JetBrains Mono';font-size:11px;color:var(--muted)"></span>
    </div>
    <canvas id="chartConfRoi" height="80"></canvas>
  </div>

  <!-- ROI par ligue — Bar chart -->
  <div class="chart-card" style="margin-bottom:16px;">
    <div class="chart-title">ROI par ligue / sport</div>
    <canvas id="chartLeague" height="120"></canvas>
  </div>

  <!-- Sharp Money -->
  <div class="chart-card" style="margin-bottom:16px;" id="sharp-section">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
      <span class="chart-title" style="margin:0">📡 Sharp Money — mouvements de cotes</span>
      <span id="sharp-avg" style="font-family:'JetBrains Mono';font-size:11px;color:var(--muted)"></span>
    </div>
    <div class="sharp-grid">
      <div class="sharp-card">
        <div class="sharp-val" style="color:var(--green)" id="sharp-confirmed">—</div>
        <div class="sharp-lbl">Sharps confirment ✓</div>
      </div>
      <div class="sharp-card">
        <div class="sharp-val" style="color:var(--muted)" id="sharp-neutral">—</div>
        <div class="sharp-lbl">Mouvement neutre</div>
      </div>
      <div class="sharp-card">
        <div class="sharp-val" style="color:var(--red)" id="sharp-cancelled">—</div>
        <div class="sharp-lbl">Paris annulés</div>
      </div>
    </div>
  </div>

  <!-- AG — Model History -->
  <div class="table-card" id="model-history-section" style="margin-bottom:16px;">
    <div class="table-header">
      <span class="table-title">📈 Historique retrains</span>
      <span style="color:var(--muted);font-size:12px;font-family:'JetBrains Mono'" id="mh-status"></span>
    </div>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Modèle</th><th>Date</th><th>Accuracy</th><th>ΔAcc</th><th>Log Loss</th><th>Samples</th>
          </tr>
        </thead>
        <tbody id="model-history-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- ROI par ligue — Table -->
  <div class="table-card" id="roi-section" style="margin-bottom:16px;">
    <div class="table-header">
      <span class="table-title">Performance par ligue</span>
    </div>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Ligue</th><th>Sport</th><th>Paris</th><th>W</th>
            <th>Win Rate</th><th>P&L (FCFA)</th><th>ROI</th>
          </tr>
        </thead>
        <tbody id="roi-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- ROI par marché — Table -->
  <div class="table-card" id="market-roi-section" style="margin-bottom:16px;">
    <div class="table-header">
      <span class="table-title">Performance par marché</span>
    </div>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Marché</th><th>Paris</th><th>W</th><th>Win Rate</th><th>P&L (FCFA)</th><th>ROI</th>
          </tr>
        </thead>
        <tbody id="market-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Backtesting interactif -->
  <div class="chart-card" style="margin-bottom:16px;">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:8px;">
      <span class="chart-title" style="margin:0">🔬 Backtesting — Simuler une période</span>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <label style="font-size:12px;color:var(--muted);font-family:'JetBrains Mono'">Du</label>
        <input type="date" id="bt-from"
          style="background:var(--surface2);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:12px;font-family:'JetBrains Mono'">
        <label style="font-size:12px;color:var(--muted);font-family:'JetBrains Mono'">au</label>
        <input type="date" id="bt-to"
          style="background:var(--surface2);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:12px;font-family:'JetBrains Mono'">
        <select id="bt-sport" style="background:var(--surface2);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:12px;">
          <option value="">Tous sports</option>
          <option value="football">⚽ Football</option>
          <option value="nba">🏀 NBA</option>
        </select>
        <select id="bt-market" style="background:var(--surface2);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:12px;">
          <option value="">Tous marchés</option>
          <option value="1X2">1X2</option>
          <option value="OU_2.5">O/U 2.5</option>
          <option value="BTTS">BTTS</option>
          <option value="AH_-0.5">AH -0.5</option>
        </select>
        <button class="refresh-btn" onclick="runBacktest()">▶ Simuler</button>
      </div>
    </div>
    <div id="bt-results" style="display:none;">
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px;" id="bt-kpis"></div>
      <canvas id="chartBacktest" height="80"></canvas>
    </div>
    <div id="bt-empty" style="color:var(--muted);font-size:13px;font-family:'JetBrains Mono';text-align:center;padding:20px 0;">
      Sélectionne une période et clique sur Simuler
    </div>
  </div>

  <!-- AF — xG + Squad values widget -->
  <div class="chart-card" style="margin-bottom:16px;" id="xg-squad-section">
    <div class="table-header" style="margin-bottom:12px;">
      <span class="table-title">⚽ xG Understat &amp; 💶 Valeurs effectifs</span>
      <span style="color:var(--muted);font-size:12px;font-family:'JetBrains Mono'">Rolling 8 matchs</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
      <div>
        <div style="color:var(--muted);font-size:11px;font-family:'JetBrains Mono';margin-bottom:6px;">TOP xG OFFENSIF</div>
        <table style="width:100%;font-size:12px;">
          <thead><tr>
            <th style="text-align:left;color:var(--muted);font-weight:400;padding:2px 6px">#</th>
            <th style="text-align:left;color:var(--muted);font-weight:400;padding:2px 6px">Équipe</th>
            <th style="text-align:right;color:var(--muted);font-weight:400;padding:2px 6px">xG</th>
            <th style="text-align:right;color:var(--muted);font-weight:400;padding:2px 6px">xGA</th>
          </tr></thead>
          <tbody id="xg-tbody"></tbody>
        </table>
      </div>
      <div>
        <div style="color:var(--muted);font-size:11px;font-family:'JetBrains Mono';margin-bottom:6px;">TOP VALEUR MARCHANDE</div>
        <table style="width:100%;font-size:12px;">
          <thead><tr>
            <th style="text-align:left;color:var(--muted);font-weight:400;padding:2px 6px">#</th>
            <th style="text-align:left;color:var(--muted);font-weight:400;padding:2px 6px">Équipe</th>
            <th style="text-align:right;color:var(--muted);font-weight:400;padding:2px 6px">M€</th>
          </tr></thead>
          <tbody id="sq-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Predictions table -->
  <div class="table-card">
    <div class="table-header">
      <span class="table-title">Prédictions récentes</span>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <select id="filter-sport" onchange="applyFilter()">
          <option value="">Tous les sports</option>
          <option value="football">⚽ Football</option>
          <option value="nba">🏀 NBA</option>
        </select>
        <select id="filter-market" onchange="applyFilter()">
          <option value="">Tous les marchés</option>
          <option value="1X2">1X2</option>
          <option value="OU_2.5">O/U 2.5</option>
          <option value="BTTS">BTTS</option>
          <option value="AH_-0.5">Asian Handicap</option>
        </select>
        <select id="filter-type" onchange="applyFilter()">
          <option value="">Tous les types</option>
          <option value="value">🔥 Value Bets</option>
        </select>
        <button class="refresh-btn" onclick="loadData()">⟳ Actualiser</button>
      </div>
    </div>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Date</th><th>Sport</th><th>Match</th><th>Ligue</th>
            <th>Marché</th><th>Prédiction</th><th>Probas</th><th>Confiance</th>
            <th>Type</th><th>Mise (FCFA)</th><th>Résultat</th><th>P&L</th>
          </tr>
        </thead>
        <tbody id="pred-tbody"></tbody>
      </table>
    </div>
  </div>

</main>

<script>
let chartResult = null, chartConf = null, chartBankroll = null, chartLeague = null;
let chartConfRoi = null, chartMarket = null, chartBacktest = null;
let allPreds = [];

// ── Initialise les dates du backtester ──────────────────────
(function initBacktestDates() {
  const today = new Date();
  const from  = new Date(today);
  from.setDate(from.getDate() - 30);
  document.addEventListener('DOMContentLoaded', () => {
    const fmt = d => d.toISOString().split('T')[0];
    document.getElementById('bt-from').value = fmt(from);
    document.getElementById('bt-to').value   = fmt(today);
  });
})();

async function loadData() {
  const [statsRes, predsRes, roiRes, bkRes, expRes, confRes, modelsRes, marketRes, brierRes, sharpRes, clvRes, xgSqRes, mhRes] = await Promise.all([
    fetch('/api/stats'),
    fetch('/api/predictions'),
    fetch('/api/roi_by_league'),
    fetch('/api/bankroll_history'),
    fetch('/api/daily_exposure'),
    fetch('/api/roi_by_confidence'),
    fetch('/api/models_status'),
    fetch('/api/market_stats'),
    fetch('/api/brier_score'),
    fetch('/api/sharp_money'),
    fetch('/api/clv'),
    fetch('/api/xg_squad'),
    fetch('/api/model_history'),
  ]);

  const stats     = await statsRes.json();
  const preds     = await predsRes.json();
  const roiData   = await roiRes.json();
  const bkData    = await bkRes.json();
  const expData   = await expRes.json();
  const confData  = await confRes.json();
  const models    = await modelsRes.json();
  const marketData= await marketRes.json();
  const brier     = await brierRes.json();
  const sharp     = await sharpRes.json();
  const clv       = await clvRes.json();
  const xgSq      = await xgSqRes.json();
  const mhData    = await mhRes.json();
  allPreds = preds;

  document.getElementById('updated').textContent = new Date().toLocaleTimeString('fr-FR');

  // ── KPIs ───────────────────────────────────────────────────
  document.getElementById('kpi-balance').textContent = (stats.balance || 0).toLocaleString('fr-FR');
  document.getElementById('kpi-total').textContent   = stats.total_bets || 0;
  document.getElementById('kpi-sub-total').textContent = `${stats.wins || 0}W / ${stats.losses || 0}L`;
  document.getElementById('kpi-winrate').textContent  = `${(stats.win_rate || 0).toFixed(1)}%`;
  document.getElementById('kpi-sub-wr').textContent   = 'sur paris réglés';
  const roi = stats.roi || 0;
  const roiEl = document.getElementById('kpi-roi');
  roiEl.textContent = `${roi > 0 ? '+' : ''}${roi.toFixed(2)}%`;
  roiEl.style.color = roi >= 0 ? 'var(--green)' : 'var(--red)';
  const pnl = stats.total_pnl || 0;
  document.getElementById('kpi-pnl').textContent = `${pnl > 0 ? '+' : ''}${pnl.toLocaleString('fr-FR')} FCFA`;
  document.getElementById('kpi-vb').textContent = preds.filter(p => p.is_value_bet).length;

  // Brier Score
  const brierEl = document.getElementById('kpi-brier');
  if (brier.brier_score != null) {
    brierEl.textContent = brier.brier_score.toFixed(3);
    document.getElementById('kpi-sub-brier').textContent = `${brier.n} paris réglés`;
    // Brier Score < 0.25 = bon, interprétation inversée (moins = meilleur)
    brierEl.style.color = brier.brier_score < 0.20 ? 'var(--green)' : brier.brier_score < 0.25 ? 'var(--gold)' : 'var(--red)';
  } else {
    brierEl.textContent = '—';
  }

  // CLV
  const clvEl = document.getElementById('kpi-clv');
  if (clv.avg_clv != null) {
    clvEl.textContent = `${clv.avg_clv >= 0 ? '+' : ''}${clv.avg_clv.toFixed(2)}%`;
    clvEl.style.color = clv.avg_clv > 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('kpi-sub-clv').textContent =
      `${clv.beat_rate}% > closing | ${clv.n_bets} paris`;
  }

  // ── Models status ───────────────────────────────────────────
  const dotMap = {
    'dot-ensemble': models.football_ensemble,
    'dot-football': models.football_xgb && !models.football_ensemble,
    'dot-ou':       models.ou_football,
    'dot-btts':     models.btts_football,
    'dot-nba':      models.nba,
  };
  if (models.football_ensemble) {
    document.getElementById('dot-ensemble').className = 'model-dot ensemble';
    document.getElementById('lbl-ensemble').textContent = 'Football Ensemble ✦';
    document.getElementById('dot-football').className = 'model-dot loaded';
  } else if (models.football_xgb) {
    document.getElementById('dot-ensemble').className = 'model-dot missing';
    document.getElementById('dot-football').className = 'model-dot loaded';
  }
  if (models.ou_football)   document.getElementById('dot-ou').className   = 'model-dot loaded';
  if (models.btts_football) document.getElementById('dot-btts').className = 'model-dot loaded';
  if (models.nba)           document.getElementById('dot-nba').className   = 'model-dot loaded';

  // ── Sharp money ─────────────────────────────────────────────
  document.getElementById('sharp-confirmed').textContent = sharp.confirmed ?? 0;
  document.getElementById('sharp-neutral').textContent   = sharp.neutral   ?? 0;
  document.getElementById('sharp-cancelled').textContent = sharp.cancelled  ?? 0;
  if (sharp.confirmed + sharp.cancelled + sharp.neutral > 0) {
    const avgTxt = `Mouvement moyen : ${sharp.avg_movement >= 0 ? '+' : ''}${sharp.avg_movement}%`;
    document.getElementById('sharp-avg').textContent = avgTxt;
    document.getElementById('sharp-avg').style.color = sharp.avg_movement >= 0 ? 'var(--green)' : 'var(--muted)';
  }

  buildExposureBar(expData);
  buildBankrollChart(bkData);
  buildConfRoiChart(confData);
  buildCharts(preds);
  buildMarketChart(marketData);
  buildTable(preds);
  buildRoiTable(roiData);
  buildLeagueChart(roiData);
  buildMarketTable(marketData);
  buildXgSquadWidget(xgSq);
  buildModelHistory(mhData);
}

function buildXgSquadWidget(data) {
  const xgBody = document.getElementById('xg-tbody');
  const sqBody = document.getElementById('sq-tbody');
  if (!xgBody || !sqBody) return;

  const medals = ['🥇','🥈','🥉'];
  const xgColor = v => v >= 2.0 ? 'var(--green)' : v >= 1.5 ? 'var(--gold)' : 'var(--muted)';
  const xgaColor = v => v <= 1.0 ? 'var(--green)' : v <= 1.5 ? 'var(--gold)' : 'var(--red)';

  xgBody.innerHTML = (data.xg || []).map((r, i) => `
    <tr>
      <td style="padding:3px 6px;color:var(--muted);font-size:11px">${medals[i] || (i+1)}</td>
      <td style="padding:3px 6px;font-family:'JetBrains Mono';font-size:12px">${r.team.substring(0,22)}</td>
      <td style="padding:3px 6px;text-align:right;font-family:'JetBrains Mono';color:${xgColor(r.xg_avg)}">${r.xg_avg.toFixed(2)}</td>
      <td style="padding:3px 6px;text-align:right;font-family:'JetBrains Mono';color:${xgaColor(r.xga_avg)}">${r.xga_avg.toFixed(2)}</td>
    </tr>`).join('');

  const maxVal = Math.max(...(data.squad || []).map(r => r.value_m), 1);
  sqBody.innerHTML = (data.squad || []).map((r, i) => {
    const barPct = Math.round(r.value_m / maxVal * 100);
    const barColor = i === 0 ? 'var(--gold)' : i < 3 ? 'var(--accent)' : 'var(--accent2)';
    return `<tr>
      <td style="padding:3px 6px;color:var(--muted);font-size:11px">${medals[i] || (i+1)}</td>
      <td style="padding:3px 6px;font-family:'JetBrains Mono';font-size:12px">
        ${r.team.substring(0,20)}
        <div style="height:3px;width:${barPct}%;background:${barColor};opacity:.6;border-radius:2px;margin-top:2px"></div>
      </td>
      <td style="padding:3px 6px;text-align:right;font-family:'JetBrains Mono';color:${barColor}">${r.value_m >= 1000 ? (r.value_m/1000).toFixed(2)+'bn' : r.value_m.toFixed(0)+'M'}</td>
    </tr>`;
  }).join('');
}

function buildModelHistory(data) {
  const tbody = document.getElementById('model-history-tbody');
  const status = document.getElementById('mh-status');
  if (!tbody) return;

  const MODEL_LABELS = {
    football_1x2: '⚽ 1X2', over_under: '📊 O/U 2.5', btts: '🎯 BTTS'
  };

  // Flatten toutes les entrées, triées par date desc
  const rows = [];
  for (const [model, runs] of Object.entries(data)) {
    runs.forEach((r, i) => {
      const prev = i > 0 ? runs[i-1].accuracy : null;
      rows.push({ model, ...r, prev_acc: prev });
    });
  }
  rows.sort((a, b) => b.ts.localeCompare(a.ts));

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted);text-align:center;padding:12px">Aucun retrain enregistré</td></tr>';
    status.textContent = 'Pas encore de données';
    return;
  }

  // Badge dégradation
  let degraded = 0;
  tbody.innerHTML = rows.slice(0, 30).map(r => {
    const delta = r.prev_acc != null ? r.accuracy - r.prev_acc : null;
    const deltaStr = delta != null
      ? `<span style="color:${delta >= 0 ? 'var(--green)' : 'var(--red)'}">${delta >= 0 ? '+' : ''}${(delta*100).toFixed(1)}%</span>`
      : '<span style="color:var(--muted)">—</span>';
    if (delta != null && delta < -0.02) degraded++;
    const accColor = r.accuracy >= 0.54 ? 'var(--green)' : r.accuracy >= 0.52 ? 'var(--gold)' : 'var(--red)';
    const date = r.ts ? r.ts.replace('T', ' ').slice(0, 16) : '—';
    return `<tr>
      <td style="font-family:'JetBrains Mono';font-size:12px">${MODEL_LABELS[r.model] || r.model}</td>
      <td style="color:var(--muted);font-size:11px;font-family:'JetBrains Mono'">${date}</td>
      <td style="color:${accColor};font-family:'JetBrains Mono'">${(r.accuracy*100).toFixed(1)}%</td>
      <td>${deltaStr}</td>
      <td style="font-family:'JetBrains Mono';color:var(--muted)">${r.log_loss != null ? r.log_loss.toFixed(4) : '—'}</td>
      <td style="font-family:'JetBrains Mono';color:var(--muted)">${r.n_samples != null ? r.n_samples.toLocaleString('fr-FR') : '—'}</td>
    </tr>`;
  }).join('');

  status.textContent = degraded > 0
    ? `⚠ ${degraded} dégradation(s) détectée(s)`
    : `${rows.length} run(s) — stable`;
  status.style.color = degraded > 0 ? 'var(--red)' : 'var(--green)';
}

function buildExposureBar(data) {
  const pct   = data.pct_used || 0;
  const color = pct >= 80 ? 'var(--red)' : pct >= 50 ? 'var(--gold)' : 'var(--green)';
  document.getElementById('exposure-bar').style.width      = pct + '%';
  document.getElementById('exposure-bar').style.background = color;
  document.getElementById('exposure-pct').textContent      = pct.toFixed(1) + '%';
  document.getElementById('exposure-pct').style.color      = color;
  document.getElementById('exposure-staked').textContent   = (data.staked_today || 0).toLocaleString('fr-FR') + ' FCFA misés ce jour';
  document.getElementById('exposure-limit').textContent    = 'Limite : ' + (data.daily_limit || 0).toLocaleString('fr-FR') + ' FCFA';
}

function buildBankrollChart(data) {
  if (chartBankroll) chartBankroll.destroy();
  if (!data.length) return;
  const byDate = {};
  data.forEach(d => { byDate[d.date] = d.balance; });
  const labels = Object.keys(byDate).sort();
  const values = labels.map(d => byDate[d]);
  const initial = values[0] || 100000;
  const last    = values[values.length - 1];
  const delta   = last - initial;
  const deltaEl = document.getElementById('bankroll-delta');
  deltaEl.textContent = `${delta >= 0 ? '+' : ''}${delta.toLocaleString('fr-FR')} FCFA depuis le départ`;
  deltaEl.style.color = delta >= 0 ? 'var(--green)' : 'var(--red)';
  const lineColor = last >= initial ? '#00e676' : '#ff4757';
  chartBankroll = new Chart(document.getElementById('chartBankroll'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: values, borderColor: lineColor, backgroundColor: `${lineColor}14`,
        borderWidth: 2, pointRadius: values.length < 30 ? 4 : 1,
        pointBackgroundColor: lineColor, fill: true, tension: 0.3,
      }]
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ctx.parsed.y.toLocaleString('fr-FR') + ' FCFA' } }
      },
      scales: {
        x: { ticks: { color: '#5a7a99', maxTicksLimit: 10 }, grid: { color: 'rgba(255,255,255,.04)' } },
        y: { ticks: { color: '#5a7a99', callback: v => (v/1000).toFixed(0) + 'k' }, grid: { color: 'rgba(255,255,255,.04)' } }
      }
    }
  });
}

function buildCharts(preds) {
  const results = {H: 0, D: 0, A: 0};
  const confBuckets = {'>80%': 0, '70-80%': 0, '60-70%': 0, '<60%': 0};
  preds.forEach(p => {
    if (['H','D','A'].includes(p.pred_result)) results[p.pred_result] = (results[p.pred_result] || 0) + 1;
    const c = p.confidence;
    if (c >= 0.8) confBuckets['>80%']++;
    else if (c >= 0.7) confBuckets['70-80%']++;
    else if (c >= 0.6) confBuckets['60-70%']++;
    else confBuckets['<60%']++;
  });

  if (chartResult) chartResult.destroy();
  chartResult = new Chart(document.getElementById('chartResult'), {
    type: 'doughnut',
    data: {
      labels: ['Dom.', 'Nul', 'Ext.'],
      datasets: [{ data: [results.H, results.D, results.A], backgroundColor: ['#00d4ff', '#5a7a99', '#ff6b2b'], borderWidth: 0 }]
    },
    options: { plugins: { legend: { labels: { color: '#e8eef5', font: { size: 12 } } } } }
  });

  if (chartConf) chartConf.destroy();
  chartConf = new Chart(document.getElementById('chartConf'), {
    type: 'bar',
    data: {
      labels: Object.keys(confBuckets),
      datasets: [{ data: Object.values(confBuckets), backgroundColor: ['#00e676', '#00d4ff', '#f5c842', '#ff4757'], borderWidth: 0, borderRadius: 4 }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#5a7a99' }, grid: { color: 'rgba(255,255,255,.05)' } },
        y: { ticks: { color: '#5a7a99' }, grid: { color: 'rgba(255,255,255,.05)' } }
      }
    }
  });
}

function buildMarketChart(data) {
  if (chartMarket) chartMarket.destroy();
  if (!data.length) return;
  const MARKET_COLORS = {
    '1X2':    '#00d4ff',
    'OU_2.5': '#00e676',
    'BTTS':   '#14b8a6',
    'AH_-0.5': '#a855f7',
  };
  const labels  = data.map(d => d.market);
  const values  = data.map(d => d.bets);
  const colors  = labels.map(l => MARKET_COLORS[l] || '#5a7a99');
  chartMarket = new Chart(document.getElementById('chartMarket'), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: colors, borderWidth: 0 }]
    },
    options: {
      plugins: {
        legend: { labels: { color: '#e8eef5', font: { size: 12 } } },
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = data[ctx.dataIndex];
              return [`${ctx.label} : ${d.bets} paris`, `Win rate : ${d.win_rate}% | ROI : ${d.roi >= 0 ? '+' : ''}${d.roi}%`];
            }
          }
        }
      }
    }
  });
}

function buildConfRoiChart(data) {
  if (chartConfRoi) chartConfRoi.destroy();
  const card = document.getElementById('conf-roi-card');
  if (!data.length) { card.style.display = 'none'; return; }
  card.style.display = '';
  const labels      = data.map(d => d.tier);
  const values      = data.map(d => d.roi);
  const bgColors    = values.map(v => v >= 0 ? 'rgba(0,230,118,.18)'  : 'rgba(255,71,87,.18)');
  const bdColors    = values.map(v => v >= 0 ? '#00e676' : '#ff4757');
  const t = {};
  data.forEach(d => t[d.tier] = d);
  const low = t['<55%']?.roi ?? 0, high = t['>65%']?.roi ?? 0;
  const calibEl = document.getElementById('conf-calibration');
  if (data.length >= 2 && high < low) {
    calibEl.textContent = '⚠️ Calibration à vérifier'; calibEl.style.color = 'var(--red)';
  } else if (data.length >= 2) {
    calibEl.textContent = '✓ Calibration correcte'; calibEl.style.color = 'var(--green)';
  } else {
    calibEl.textContent = '';
  }
  chartConfRoi = new Chart(document.getElementById('chartConfRoi'), {
    type: 'bar',
    data: { labels, datasets: [{ data: values, backgroundColor: bgColors, borderColor: bdColors, borderWidth: 2, borderRadius: 6 }] },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = data[ctx.dataIndex];
              const s = ctx.parsed.y >= 0 ? '+' : '';
              return [`ROI : ${s}${ctx.parsed.y.toFixed(2)}%`, `${d.bets} paris | ${d.wins}W (${d.win_rate}%) | P&L ${d.pnl >= 0 ? '+' : ''}${d.pnl.toLocaleString('fr-FR')} FCFA`];
            }
          }
        }
      },
      scales: {
        x: { ticks: { color: '#e8eef5', font: { size: 13 } }, grid: { display: false } },
        y: { ticks: { color: '#5a7a99', callback: v => (v >= 0 ? '+' : '') + v.toFixed(1) + '%' }, grid: { color: 'rgba(255,255,255,.05)' } }
      }
    }
  });
}

function buildLeagueChart(data) {
  if (chartLeague) chartLeague.destroy();
  const canvas = document.getElementById('chartLeague');
  if (!data.length) { canvas.parentElement.style.display = 'none'; return; }
  canvas.parentElement.style.display = '';
  const sorted    = [...data].sort((a, b) => b.roi - a.roi);
  const labels    = sorted.map(r => (r.sport === 'football' ? '⚽ ' : '🏀 ') + r.league);
  const values    = sorted.map(r => r.roi);
  const bgColors  = values.map(v => v >= 0 ? 'rgba(0,230,118,.18)'  : 'rgba(255,71,87,.18)');
  const bdColors  = values.map(v => v >= 0 ? '#00e676' : '#ff4757');
  chartLeague = new Chart(canvas, {
    type: 'bar',
    data: { labels, datasets: [{ data: values, backgroundColor: bgColors, borderColor: bdColors, borderWidth: 2, borderRadius: 4 }] },
    options: {
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const r = sorted[ctx.dataIndex];
              const s = ctx.parsed.x >= 0 ? '+' : '';
              return [`ROI : ${s}${ctx.parsed.x.toFixed(2)}%`, `${r.bets} paris | ${r.wins}W (${r.win_rate}%) | P&L ${r.pnl >= 0 ? '+' : ''}${r.pnl.toLocaleString('fr-FR')} FCFA`];
            }
          }
        }
      },
      scales: {
        x: { ticks: { color: '#5a7a99', callback: v => (v >= 0 ? '+' : '') + v.toFixed(1) + '%' }, grid: { color: 'rgba(255,255,255,.05)' } },
        y: { ticks: { color: '#e8eef5', font: { size: 12 } }, grid: { display: false } }
      }
    }
  });
}

function applyFilter() {
  const sport  = document.getElementById('filter-sport').value;
  const market = document.getElementById('filter-market').value;
  const type   = document.getElementById('filter-type').value;
  let filtered = allPreds;
  if (sport)  filtered = filtered.filter(p => p.sport === sport);
  if (market) filtered = filtered.filter(p => (p.market || '1X2') === market);
  if (type === 'value') filtered = filtered.filter(p => p.is_value_bet);
  buildTable(filtered);
}

function buildRoiTable(data) {
  const tbody = document.getElementById('roi-tbody');
  if (!data.length) { tbody.innerHTML = '<tr><td colspan="7" style="color:var(--muted);text-align:center">Aucun pari réglé</td></tr>'; return; }
  tbody.innerHTML = data.map(r => {
    const roiColor = r.roi >= 0 ? 'var(--green)' : 'var(--red)';
    const pnlColor = r.pnl >= 0 ? 'var(--green)' : 'var(--red)';
    return `<tr>
      <td><b>${r.league}</b></td>
      <td>${r.sport === 'football' ? '⚽' : '🏀'}</td>
      <td style="font-family:'JetBrains Mono'">${r.bets}</td>
      <td style="font-family:'JetBrains Mono'">${r.wins}</td>
      <td style="font-family:'JetBrains Mono';color:var(--gold)">${r.win_rate}%</td>
      <td style="font-family:'JetBrains Mono';color:${pnlColor}">${r.pnl >= 0 ? '+' : ''}${r.pnl.toLocaleString('fr-FR')}</td>
      <td style="font-family:'JetBrains Mono';color:${roiColor};font-weight:700">${r.roi >= 0 ? '+' : ''}${r.roi}%</td>
    </tr>`;
  }).join('');
}

function buildMarketTable(data) {
  const tbody = document.getElementById('market-tbody');
  if (!data.length) { tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted);text-align:center">Aucun paris réglé</td></tr>'; return; }
  const MARKET_LABELS = { '1X2': '⚽ 1X2', 'OU_2.5': '⚖️ O/U 2.5', 'BTTS': '🎯 BTTS', 'AH_-0.5': '🔰 AH -0.5' };
  tbody.innerHTML = data.map(r => {
    const roiColor = r.roi >= 0 ? 'var(--green)' : 'var(--red)';
    const pnlColor = r.pnl >= 0 ? 'var(--green)' : 'var(--red)';
    return `<tr>
      <td><b>${MARKET_LABELS[r.market] || r.market}</b></td>
      <td style="font-family:'JetBrains Mono'">${r.bets}</td>
      <td style="font-family:'JetBrains Mono'">${r.wins}</td>
      <td style="font-family:'JetBrains Mono';color:var(--gold)">${r.win_rate}%</td>
      <td style="font-family:'JetBrains Mono';color:${pnlColor}">${r.pnl >= 0 ? '+' : ''}${r.pnl.toLocaleString('fr-FR')}</td>
      <td style="font-family:'JetBrains Mono';color:${roiColor};font-weight:700">${r.roi >= 0 ? '+' : ''}${r.roi}%</td>
    </tr>`;
  }).join('');
}

function marketBadge(market) {
  if (!market || market === '1X2')  return '<span class="badge-market-1x2">1X2</span>';
  if (market === 'OU_2.5')         return '<span class="badge-market-ou">O/U 2.5</span>';
  if (market === 'BTTS')           return '<span class="badge-market-btts">BTTS</span>';
  if (market === 'AH_-0.5')        return '<span class="badge-market-ah">AH -0.5</span>';
  return `<span class="badge-market-1x2">${market}</span>`;
}

function buildTable(preds) {
  const NAMES = {
    H: 'Domicile', D: 'Nul', A: 'Extérieur',
    OVER: 'Over 2.5', UNDER: 'Under 2.5',
    BTTS: 'BTTS ✓', NO_BTTS: 'BTTS ✗',
    AH_H: 'AH Dom.', AH_A: 'AH Ext.',
  };
  const BADGE_CLS = {
    H: 'badge-h', D: 'badge-d', A: 'badge-a',
    OVER: 'badge-over', UNDER: 'badge-under',
    BTTS: 'badge-btts', NO_BTTS: 'badge-no-btts',
    AH_H: 'badge-ah', AH_A: 'badge-ah',
  };

  const tbody = document.getElementById('pred-tbody');
  tbody.innerHTML = preds.slice(0, 60).map(p => {
    const conf      = (p.confidence * 100).toFixed(1);
    const confColor = p.confidence >= 0.7 ? 'var(--green)' : p.confidence >= 0.6 ? 'var(--gold)' : 'var(--muted)';
    const ph  = Math.round(p.prob_home * 100);
    const pd_ = Math.round((p.prob_draw || 0) * 100);
    const pa  = Math.round(p.prob_away * 100);
    const pnlStr = p.pnl != null
      ? `<span style="color:${p.pnl >= 0 ? 'var(--green)' : 'var(--red)'}; font-family:'JetBrains Mono'; font-size:12px;">${p.pnl >= 0 ? '+' : ''}${p.pnl.toLocaleString('fr-FR')}</span>`
      : '<span style="color:var(--muted)">—</span>';

    // Outcome badge: check if correct prediction for any market
    let outcomeStr;
    if (p.outcome) {
      const correct = p.pred_result === p.outcome;
      const outcomeName = NAMES[p.outcome] || p.outcome;
      outcomeStr = `<span class="badge ${correct ? 'badge-win' : 'badge-loss'}">${outcomeName}</span>`;
    } else {
      outcomeStr = '<span style="color:var(--muted);font-size:12px;">En attente</span>';
    }

    const predName  = NAMES[p.pred_result] || p.pred_result;
    const predBadge = BADGE_CLS[p.pred_result] || 'badge-signal';

    // Sharp money indicator
    let sharpIndicator = '';
    if (p.opening_movement_pct != null) {
      const mvt = p.opening_movement_pct * 100;
      if (mvt > 5) sharpIndicator = ' <span style="color:var(--green);font-size:10px" title="Sharps confirment">⬆</span>';
      else if (mvt < -5) sharpIndicator = ' <span style="color:var(--red);font-size:10px" title="Paris annulé">⬇</span>';
    }

    return `<tr>
      <td style="font-family:'JetBrains Mono';font-size:11px;color:var(--muted)">${p.created_at?.split(' ')[0] || '—'}</td>
      <td>${p.sport === 'football' ? '⚽' : '🏀'}</td>
      <td><b>${p.home_team}</b><br><span style="color:var(--muted);font-size:11px">vs ${p.away_team}</span></td>
      <td style="color:var(--muted);font-size:12px">${p.league || '—'}</td>
      <td>${marketBadge(p.market)}</td>
      <td><span class="badge ${predBadge}">${predName}</span>${sharpIndicator}</td>
      <td>
        <div class="prob-bar">
          <span style="width:${ph*1.2}px;background:var(--accent);opacity:.8"></span>
          ${p.prob_draw ? `<span style="width:${pd_*1.2}px;background:var(--muted);opacity:.6"></span>` : ''}
          <span style="width:${pa*1.2}px;background:var(--accent2);opacity:.8"></span>
        </div>
        <span style="font-family:'JetBrains Mono';font-size:10px;color:var(--muted)">${ph}/${pd_ || 0}/${pa}</span>
      </td>
      <td style="color:${confColor};font-family:'JetBrains Mono';font-size:13px">${conf}%</td>
      <td>${p.is_value_bet ? '<span class="badge badge-value">🔥 VALUE</span>' : '<span class="badge badge-signal">SIGNAL</span>'}</td>
      <td style="font-family:'JetBrains Mono';font-size:12px">${p.kelly_stake ? p.kelly_stake.toLocaleString('fr-FR') : '—'}</td>
      <td>${outcomeStr}</td>
      <td>${pnlStr}</td>
    </tr>`;
  }).join('');
}

async function runBacktest() {
  const from   = document.getElementById('bt-from').value;
  const to     = document.getElementById('bt-to').value;
  const sport  = document.getElementById('bt-sport').value;
  const market = document.getElementById('bt-market').value;
  if (!from || !to) return;

  const params = new URLSearchParams({ from, to });
  if (sport)  params.append('sport',  sport);
  if (market) params.append('market', market);

  const res  = await fetch('/api/backtest?' + params);
  const data = await res.json();

  document.getElementById('bt-empty').style.display   = 'none';
  document.getElementById('bt-results').style.display = '';

  // KPIs
  const kpiColor = v => v >= 0 ? 'var(--green)' : 'var(--red)';
  document.getElementById('bt-kpis').innerHTML = `
    <div class="sharp-card">
      <div class="sharp-val" style="color:var(--gold)">${data.bets}</div>
      <div class="sharp-lbl">Paris réglés</div>
    </div>
    <div class="sharp-card">
      <div class="sharp-val" style="color:var(--green)">${data.wins}</div>
      <div class="sharp-lbl">Victoires</div>
    </div>
    <div class="sharp-card">
      <div class="sharp-val" style="color:var(--gold)">${data.win_rate}%</div>
      <div class="sharp-lbl">Win Rate</div>
    </div>
    <div class="sharp-card">
      <div class="sharp-val" style="color:${kpiColor(data.roi)}">${data.roi >= 0 ? '+' : ''}${data.roi}%</div>
      <div class="sharp-lbl">ROI période</div>
    </div>
    <div class="sharp-card">
      <div class="sharp-val" style="color:${kpiColor(data.pnl)};font-size:20px">${data.pnl >= 0 ? '+' : ''}${(data.pnl||0).toLocaleString('fr-FR')}</div>
      <div class="sharp-lbl">P&L (FCFA)</div>
    </div>
    <div class="sharp-card">
      <div class="sharp-val" style="color:var(--purple)">${data.brier != null ? data.brier.toFixed(3) : '—'}</div>
      <div class="sharp-lbl">Brier Score</div>
    </div>
  `;

  // Courbe bankroll simulée
  if (chartBacktest) chartBacktest.destroy();
  const labels = data.curve.map(d => d.date);
  const values = data.curve.map(d => d.cumulative_pnl);
  const last   = values[values.length - 1] ?? 0;
  const lineColor = last >= 0 ? '#00e676' : '#ff4757';

  chartBacktest = new Chart(document.getElementById('chartBacktest'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'P&L cumulé (FCFA)',
        data: values,
        borderColor: lineColor,
        backgroundColor: `${lineColor}14`,
        borderWidth: 2,
        pointRadius: values.length < 40 ? 3 : 0,
        fill: true,
        tension: 0.3,
      }]
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => (ctx.parsed.y >= 0 ? '+' : '') + ctx.parsed.y.toLocaleString('fr-FR') + ' FCFA'
          }
        }
      },
      scales: {
        x: { ticks: { color: '#5a7a99', maxTicksLimit: 12 }, grid: { color: 'rgba(255,255,255,.04)' } },
        y: {
          ticks: { color: '#5a7a99', callback: v => (v >= 0 ? '+' : '') + (v/1000).toFixed(0) + 'k' },
          grid: { color: 'rgba(255,255,255,.04)' }
        }
      }
    }
  });
}

loadData();
setInterval(loadData, 60000);
</script>
</body>
</html>
"""

# ════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(TEMPLATE)


@app.route("/api/predictions")
def api_predictions():
    import math
    df = get_all_predictions()
    if df.empty:
        return jsonify([])
    df = df.where(df.notna(), other=None)
    records = df.to_dict(orient="records")
    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    records = [{k: clean(v) for k, v in row.items()} for row in records]
    return jsonify(records)


@app.route("/api/stats")
def api_stats():
    import math
    stats = tracker.get_stats()
    clean = {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
             for k, v in stats.items()}
    return jsonify(clean)


@app.route("/api/roi_by_league")
def api_roi_by_league():
    import math
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT league, sport,
                   COUNT(*) as bets,
                   SUM(CASE WHEN pred_result = outcome THEN 1 ELSE 0 END) as wins,
                   SUM(COALESCE(pnl, 0))  as pnl,
                   SUM(COALESCE(kelly_stake, 0)) as staked
            FROM predictions
            WHERE outcome IS NOT NULL AND kelly_stake > 0
            GROUP BY league, sport
            ORDER BY pnl DESC
        """).fetchall()
    result = []
    for league, sport, bets, wins, pnl, staked in rows:
        roi = (pnl / staked * 100) if staked else 0.0
        wr  = (wins / bets * 100)  if bets  else 0.0
        if not (math.isnan(roi) or math.isinf(roi)):
            result.append({
                "league": league, "sport": sport,
                "bets": bets, "wins": int(wins or 0),
                "win_rate": round(wr, 1),
                "pnl": round(pnl or 0, 0),
                "roi": round(roi, 2),
            })
    return jsonify(result)


@app.route("/api/bankroll_history")
def api_bankroll_history():
    from config import INITIAL_BANKROLL
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT updated_at, balance FROM bankroll ORDER BY id ASC"
        ).fetchall()
    data = [{"date": r[0][:10] if r[0] else "", "balance": r[1]} for r in rows]
    if not data:
        from datetime import date
        data = [{"date": str(date.today()), "balance": INITIAL_BANKROLL}]
    return jsonify(data)


@app.route("/api/daily_exposure")
def api_daily_exposure():
    from config import MAX_DAILY_STAKE_PCT
    today = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        row = conn.execute(f"""
            SELECT COALESCE(SUM(kelly_stake), 0) FROM predictions
            WHERE DATE(created_at) = {_ph} AND kelly_stake > 0
        """, (today,)).fetchone()
    staked_today = float(row[0]) if row else 0.0
    bankroll     = tracker.get_balance()
    daily_limit  = bankroll * MAX_DAILY_STAKE_PCT
    pct_used     = (staked_today / daily_limit * 100) if daily_limit > 0 else 0.0
    return jsonify({
        "staked_today": round(staked_today, 0),
        "daily_limit":  round(daily_limit, 0),
        "pct_used":     round(min(pct_used, 100), 1),
    })


@app.route("/api/roi_by_confidence")
def api_roi_by_confidence():
    return jsonify(tracker.get_roi_by_confidence())


@app.route("/api/models_status")
def api_models_status():
    """Vérifie quels fichiers modèles sont présents sur disque."""
    from config import MODELS_DIR
    def exists(fname):
        return os.path.exists(os.path.join(MODELS_DIR, fname))
    return jsonify({
        "football_ensemble": exists("football_ensemble_model.pkl"),
        "football_xgb":      exists("football_xgb_model.pkl"),
        "ou_football":       exists("ou_football_xgb_model.pkl"),
        "btts_football":     exists("btts_football_xgb_model.pkl"),
        "nba":               exists("nba_xgb_model.pkl"),
    })


@app.route("/api/market_stats")
def api_market_stats():
    """Statistiques (bets, win rate, ROI) par marché."""
    import math
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT COALESCE(market, '1X2') as market,
                   COUNT(*) as bets,
                   SUM(CASE WHEN pred_result = outcome THEN 1 ELSE 0 END) as wins,
                   SUM(COALESCE(pnl, 0)) as pnl,
                   SUM(COALESCE(kelly_stake, 0)) as staked
            FROM predictions
            WHERE kelly_stake > 0
            GROUP BY COALESCE(market, '1X2')
            ORDER BY bets DESC
        """).fetchall()
    result = []
    for market, bets, wins, pnl, staked in rows:
        wins = int(wins or 0)
        wr   = round(wins / bets * 100, 1) if bets else 0.0
        roi  = round(pnl / staked * 100, 2) if staked else 0.0
        if not (math.isnan(roi) or math.isinf(roi)):
            result.append({
                "market":   market,
                "bets":     bets,
                "wins":     wins,
                "win_rate": wr,
                "pnl":      round(pnl or 0, 0),
                "roi":      roi,
            })
    return jsonify(result)


@app.route("/api/brier_score")
def api_brier_score():
    """Brier Score sur tous les paris réglés (calibration du modèle).
    BS = mean((confidence - correct)^2). Meilleur score proche de 0."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT confidence, pred_result, outcome
            FROM predictions
            WHERE outcome IS NOT NULL AND confidence IS NOT NULL
        """).fetchall()
    if not rows:
        return jsonify({"brier_score": None, "n": 0})
    total = sum(
        (conf - (1.0 if pred == outcome else 0.0)) ** 2
        for conf, pred, outcome in rows
    )
    bs = total / len(rows)
    return jsonify({"brier_score": round(bs, 4), "n": len(rows)})


@app.route("/api/clv")
def api_clv():
    """Closing Line Value moyen — standard de l'industrie pour mesurer l'edge réel."""
    clv = tracker.get_avg_clv()
    if not clv:
        return jsonify({"avg_clv": None, "beat_rate": None, "n_bets": 0})
    return jsonify(clv)


@app.route("/api/sharp_money")
def api_sharp_money():
    """Statistiques de mouvement de cotes depuis l'ouverture.
    Positif = cote a baissé depuis nous (sharps confirment notre pari).
    Négatif = cote a monté (marché contre nous → paris annulé)."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT opening_movement_pct
            FROM predictions
            WHERE opening_movement_pct IS NOT NULL
        """).fetchall()
    if not rows:
        return jsonify({"confirmed": 0, "cancelled": 0, "neutral": 0, "avg_movement": 0.0})
    movements = [r[0] for r in rows]
    confirmed = sum(1 for m in movements if m > 0.05)
    cancelled = sum(1 for m in movements if m < -0.05)
    neutral   = sum(1 for m in movements if -0.05 <= m <= 0.05)
    avg_mvt   = round(sum(movements) / len(movements) * 100, 2)
    return jsonify({
        "confirmed":    confirmed,
        "cancelled":    cancelled,
        "neutral":      neutral,
        "avg_movement": avg_mvt,
    })


@app.route("/api/backtest")
def api_backtest():
    """
    Backtesting sur une période donnée.
    Params : from (YYYY-MM-DD), to (YYYY-MM-DD), sport?, market?
    Retourne : KPIs + courbe P&L cumulé jour par jour.
    """
    import math, sqlite3
    from config import DB_PATH

    date_from = request.args.get("from", "2020-01-01")
    date_to   = request.args.get("to",   "2099-12-31")
    sport     = request.args.get("sport",  "")
    market    = request.args.get("market", "")

    where  = ["outcome IS NOT NULL", "kelly_stake > 0"]
    params = []
    where.append(f"match_date >= {_ph}"); params.append(date_from)
    where.append(f"match_date <= {_ph}"); params.append(date_to)
    if sport:
        where.append(f"sport = {_ph}"); params.append(sport)
    if market:
        where.append(f"COALESCE(market,'1X2') = {_ph}"); params.append(market)

    sql_where = " AND ".join(where)

    with get_conn() as conn:
        # Agrégat global
        agg = conn.execute(f"""
            SELECT COUNT(*) AS bets,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                   COALESCE(SUM(pnl), 0) AS pnl,
                   COALESCE(SUM(kelly_stake), 0) AS staked,
                   AVG(
                     (confidence - CASE WHEN pred_result = outcome THEN 1.0 ELSE 0.0 END) *
                     (confidence - CASE WHEN pred_result = outcome THEN 1.0 ELSE 0.0 END)
                   ) AS brier
            FROM predictions WHERE {sql_where}
        """, params).fetchone()

        # Courbe P&L par date
        curve_rows = conn.execute(f"""
            SELECT DATE(match_date) AS d, SUM(pnl) AS daily_pnl
            FROM predictions
            WHERE {sql_where}
            GROUP BY DATE(match_date)
            ORDER BY DATE(match_date)
        """, params).fetchall()

    bets, wins, pnl, staked, brier = agg
    bets   = int(bets or 0)
    wins   = int(wins or 0)
    pnl    = float(pnl or 0)
    staked = float(staked or 0)
    roi    = round(pnl / staked * 100, 2) if staked else 0.0
    wr     = round(wins / bets * 100, 1) if bets else 0.0

    # Courbe cumulative
    cumul = 0.0
    curve = []
    for d, daily in curve_rows:
        cumul += float(daily or 0)
        curve.append({"date": d, "daily_pnl": round(float(daily or 0), 0),
                      "cumulative_pnl": round(cumul, 0)})

    return jsonify({
        "bets":     bets,
        "wins":     wins,
        "losses":   bets - wins,
        "win_rate": wr,
        "pnl":      round(pnl, 0),
        "roi":      roi,
        "brier":    round(float(brier), 4) if brier is not None else None,
        "curve":    curve,
    })


@app.route("/api/run")
def api_run():
    from predictor import run_all
    signals = run_all()
    return jsonify({"status": "ok", "signals": len(signals)})


@app.route("/api/model_history")
def api_model_history():
    try:
        from model_registry import get_all_history
        return jsonify(get_all_history())
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/xg_squad")
def api_xg_squad():
    import json as _json
    from config import DATA_DIR
    xg_path  = os.path.join(DATA_DIR, "understat_xg_history.json")
    sq_path  = os.path.join(DATA_DIR, "squad_values.json")

    # xG — compute rolling avg over last 8 matches for each team
    xg_rows = []
    if os.path.exists(xg_path):
        with open(xg_path) as f:
            xg_hist = _json.load(f)
        for team, matches in xg_hist.items():
            if not matches:
                continue
            recent = matches[-8:]
            xg_avg  = round(sum(m[1] for m in recent) / len(recent), 2)
            xga_avg = round(sum(m[2] for m in recent) / len(recent), 2)
            xg_rows.append({"team": team, "xg_avg": xg_avg, "xga_avg": xga_avg})
        xg_rows.sort(key=lambda r: r["xg_avg"], reverse=True)
        xg_rows = xg_rows[:20]

    # Squad values — top 20
    sq_rows = []
    if os.path.exists(sq_path):
        with open(sq_path) as f:
            sq_vals = _json.load(f)
        sq_rows = [{"team": k, "value_m": v} for k, v in sq_vals.items()]
        sq_rows.sort(key=lambda r: r["value_m"], reverse=True)
        sq_rows = sq_rows[:20]

    return jsonify({"xg": xg_rows, "squad": sq_rows})


@app.route("/api/account_health")
def api_account_health():
    """AU — Score de santé du compte bookmaker."""
    try:
        from account_health import compute_account_health
        report = compute_account_health("all", lookback_days=14)
        return jsonify(report)
    except Exception as e:
        return jsonify({"health_score": 1.0, "error": str(e)})


if __name__ == "__main__":
    from data_fetcher import init_db
    init_db()
    app.run(port=FLASK_PORT, debug=FLASK_DEBUG)
