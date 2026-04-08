# ============================================================
# dashboard.py — Flask Dashboard
# ============================================================

import json
import logging
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from data_fetcher import get_all_predictions
from bankroll import BankrollTracker
from config import FLASK_PORT, FLASK_DEBUG

app = Flask(__name__)
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
    display: flex; align-items: center; gap: 16px;
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

  /* KPI Cards */
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }
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
  .kpi-sub { font-size: 12px; color: var(--muted); margin-top: 4px; font-family: 'JetBrains Mono'; }

  /* Charts */
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 32px; }
  .chart-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
  }
  .chart-title { font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
                 color: var(--muted); margin-bottom: 16px; }

  /* Table */
  .table-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
  }
  .table-header {
    padding: 16px 20px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
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
  .badge-value  { background: rgba(245,200,66,.15); color: var(--gold); }
  .badge-signal { background: rgba(0,212,255,.1);   color: var(--accent); }
  .badge-win    { background: rgba(0,230,118,.12);  color: var(--green); }
  .badge-loss   { background: rgba(255,71,87,.12);  color: var(--red); }
  .badge-h      { background: rgba(0,212,255,.1);   color: var(--accent); }
  .badge-d      { background: rgba(90,122,153,.2);  color: var(--muted); }
  .badge-a      { background: rgba(255,107,43,.12); color: var(--accent2); }
  .prob-bar { display: flex; gap: 3px; align-items: center; }
  .prob-bar span { height: 6px; border-radius: 3px; display: inline-block; min-width: 2px; }
  .refresh-btn {
    font-family: 'JetBrains Mono'; font-size: 11px;
    background: var(--surface2); color: var(--accent);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 12px; cursor: pointer; transition: all .2s;
  }
  .refresh-btn:hover { background: var(--accent); color: var(--bg); }
  @media (max-width: 768px) {
    .charts-row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">BetMind</div>
  <span class="tag">AI Prediction Agent v1.0</span>
  <span class="tag">⚽ Football | 🏀 NBA</span>
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
  </div>

  <!-- Charts -->
  <div class="charts-row">
    <div class="chart-card">
      <div class="chart-title">Répartition prédictions</div>
      <canvas id="chartResult" height="180"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">Confiance distribution</div>
      <canvas id="chartConf" height="180"></canvas>
    </div>
  </div>

  <!-- Table -->
  <div class="table-card">
    <div class="table-header">
      <span class="table-title">Prédictions récentes</span>
      <button class="refresh-btn" onclick="loadData()">⟳ Actualiser</button>
    </div>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Date</th><th>Sport</th><th>Match</th><th>Ligue</th>
            <th>Prédiction</th><th>Probas</th><th>Confiance</th>
            <th>Type</th><th>Mise (FCFA)</th><th>Résultat</th><th>P&L</th>
          </tr>
        </thead>
        <tbody id="pred-tbody"></tbody>
      </table>
    </div>
  </div>
</main>

<script>
let chartResult = null, chartConf = null;

async function loadData() {
  const [statsRes, predsRes] = await Promise.all([
    fetch('/api/stats'), fetch('/api/predictions')
  ]);
  const stats = await statsRes.json();
  const preds = await predsRes.json();

  document.getElementById('updated').textContent =
    new Date().toLocaleTimeString('fr-FR');

  // KPIs
  document.getElementById('kpi-balance').textContent =
    (stats.balance || 0).toLocaleString('fr-FR');
  document.getElementById('kpi-total').textContent = stats.total_bets || 0;
  document.getElementById('kpi-sub-total').textContent =
    `${stats.wins || 0}W / ${stats.losses || 0}L`;
  document.getElementById('kpi-winrate').textContent =
    `${(stats.win_rate || 0).toFixed(1)}%`;
  document.getElementById('kpi-sub-wr').textContent = 'sur paris réglés';
  const roi = stats.roi || 0;
  const roiEl = document.getElementById('kpi-roi');
  roiEl.textContent = `${roi > 0 ? '+' : ''}${roi.toFixed(2)}%`;
  roiEl.style.color = roi >= 0 ? 'var(--green)' : 'var(--red)';
  const pnl = stats.total_pnl || 0;
  document.getElementById('kpi-pnl').textContent =
    `${pnl > 0 ? '+' : ''}${pnl.toLocaleString('fr-FR')} FCFA`;
  document.getElementById('kpi-vb').textContent =
    preds.filter(p => p.is_value_bet).length;

  buildCharts(preds);
  buildTable(preds);
}

function buildCharts(preds) {
  const results = {H: 0, D: 0, A: 0};
  const confBuckets = {'>80%': 0, '70-80%': 0, '60-70%': 0, '<60%': 0};
  preds.forEach(p => {
    results[p.pred_result] = (results[p.pred_result] || 0) + 1;
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
      datasets: [{
        data: [results.H, results.D, results.A],
        backgroundColor: ['#00d4ff', '#5a7a99', '#ff6b2b'],
        borderWidth: 0
      }]
    },
    options: {
      plugins: { legend: { labels: { color: '#e8eef5', font: { size: 12 } } } }
    }
  });

  if (chartConf) chartConf.destroy();
  chartConf = new Chart(document.getElementById('chartConf'), {
    type: 'bar',
    data: {
      labels: Object.keys(confBuckets),
      datasets: [{
        data: Object.values(confBuckets),
        backgroundColor: ['#00e676', '#00d4ff', '#f5c842', '#ff4757'],
        borderWidth: 0, borderRadius: 4
      }]
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

function buildTable(preds) {
  const NAMES = { H: 'Domicile', D: 'Nul', A: 'Extérieur' };
  const BADGE = { H: 'badge-h', D: 'badge-d', A: 'badge-a' };
  const tbody = document.getElementById('pred-tbody');
  tbody.innerHTML = preds.slice(0, 50).map(p => {
    const conf = (p.confidence * 100).toFixed(1);
    const confColor = p.confidence >= 0.7 ? 'var(--green)' : p.confidence >= 0.6 ? 'var(--gold)' : 'var(--muted)';
    const ph = Math.round(p.prob_home * 100);
    const pd_ = Math.round((p.prob_draw || 0) * 100);
    const pa = Math.round(p.prob_away * 100);
    const pnlStr = p.pnl != null
      ? `<span style="color:${p.pnl >= 0 ? 'var(--green)' : 'var(--red)'}; font-family:'JetBrains Mono'; font-size:12px;">
           ${p.pnl >= 0 ? '+' : ''}${p.pnl.toLocaleString('fr-FR')}</span>`
      : '<span style="color:var(--muted)">—</span>';
    const outcomeStr = p.outcome
      ? `<span class="badge ${p.outcome === p.pred_result ? 'badge-win' : 'badge-loss'}">${NAMES[p.outcome] || p.outcome}</span>`
      : '<span style="color:var(--muted);font-size:12px;">En attente</span>';

    return `<tr>
      <td style="font-family:'JetBrains Mono';font-size:11px;color:var(--muted)">${p.created_at?.split(' ')[0] || '—'}</td>
      <td>${p.sport === 'football' ? '⚽' : '🏀'}</td>
      <td><b>${p.home_team}</b><br><span style="color:var(--muted);font-size:11px">vs ${p.away_team}</span></td>
      <td style="color:var(--muted);font-size:12px">${p.league || '—'}</td>
      <td><span class="badge ${BADGE[p.pred_result]}">${NAMES[p.pred_result] || p.pred_result}</span></td>
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


@app.route("/api/run")
def api_run():
    """Déclenche manuellement un cycle de prédiction (via dashboard)."""
    from predictor import run_all
    signals = run_all()
    return jsonify({"status": "ok", "signals": len(signals)})


if __name__ == "__main__":
    from data_fetcher import init_db
    init_db()
    app.run(port=FLASK_PORT, debug=FLASK_DEBUG)