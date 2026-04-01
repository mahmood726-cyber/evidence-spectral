"""
EvidenceSpectral: Build interactive dashboard HTML from pipeline results.
"""

from __future__ import annotations

import json
import math
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# -----------------------------------------------------------------------
# Load / run pipeline
# -----------------------------------------------------------------------

RESULTS_JSON = Path(r"C:\Models\EvidenceSpectral\results.json")
OUT_HTML = Path(r"C:\Models\EvidenceSpectral\dashboard.html")

from spectral_engine import run_pipeline, COMPONENT_COLS

print("Running pipeline...")
results = run_pipeline()

with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {RESULTS_JSON}")

# -----------------------------------------------------------------------
# Derived display values
# -----------------------------------------------------------------------

eigenvalues = results["eigenvalues"]                 # list of 5
corr_mat = results["correlation_matrix"]             # 5x5 list
eigenvectors = results["eigenvectors"]               # 5x5 list (columns = eigenvectors)
prs = results["participation_ratios"]
n_signal = results["n_signal_eigenvalues"]
tw_s = results["tw_s"]
tw_p = results["tw_p_approx"]
lp = results["mp_lambda_plus"]
lm = results["mp_lambda_minus"]
n_obs = results["n_obs"]
first_ev = results["first_eigenvector"]

component_labels = ["Audit", "Consistency", "Robustness", "Stability", "Power"]
component_short = ["AUD", "CON", "ROB", "STA", "POW"]

# Format TW p-value
if tw_p < 1e-100:
    tw_p_str = "< 10\u207b\u00b9\u2070\u2070"
else:
    exp = int(math.floor(math.log10(tw_p))) if tw_p > 0 else -300
    tw_p_str = f"< 10<sup>{exp}</sup>"

# Eigenvector grid data (transposed: each column of eigenvectors matrix = one eigenvector)
# eigenvectors[component_idx][ev_idx]
ev_grid = []
for comp_i in range(5):
    row = []
    for ev_j in range(5):
        val = eigenvectors[comp_i][ev_j]
        row.append(round(val, 4))
    ev_grid.append(row)

# Build JS data objects
eigenvalues_js = json.dumps([round(v, 4) for v in eigenvalues])
corr_mat_js = json.dumps([[round(v, 4) for v in row] for row in corr_mat])
ev_grid_js = json.dumps([[round(v, 4) for v in row] for row in ev_grid])
prs_js = json.dumps([round(v, 3) for v in prs])
labels_js = json.dumps(component_labels)
short_js = json.dumps(component_short)

# -----------------------------------------------------------------------
# Build HTML
# -----------------------------------------------------------------------

html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EvidenceSpectral Dashboard</title>
  <style>
    :root {{
      --bg: #f6f3ee;
      --paper: rgba(255,255,255,0.92);
      --paper-strong: #fbfaf7;
      --ink: #111111;
      --muted: #5f5a53;
      --line: #ddd5ca;
      --line-strong: #b8aea2;
      --accent: #326891;
      --accent-soft: #e8f0f6;
      --good: #216c53;
      --warn: #a06a12;
      --bad: #922b21;
      --shadow: 0 16px 38px rgba(17,17,17,0.045);
      --radius: 8px;
      --serif: "Iowan Old Style","Palatino Linotype","Book Antiqua",Palatino,Georgia,serif;
      --sans: "Segoe UI","Helvetica Neue",Arial,sans-serif;
      --mono: "SFMono-Regular",Consolas,"Liberation Mono",monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: radial-gradient(circle at top, rgba(50,104,145,0.08), transparent 28%),
                  linear-gradient(180deg, #fcfbf8 0%, var(--bg) 100%);
      font-family: var(--serif);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      width: min(1200px, calc(100vw - 48px));
      margin: 18px auto 72px;
      display: grid;
      gap: 24px;
    }}
    .masthead {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 18px;
      padding: 10px 0 18px;
      border-top: 1px solid var(--line-strong);
      border-bottom: 3px double var(--line);
    }}
    .masthead-brand {{
      font-family: var(--serif);
      font-size: clamp(28px, 4vw, 44px);
      font-weight: 700;
      letter-spacing: -0.035em;
    }}
    .masthead-meta {{
      font-family: var(--sans);
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.14em;
      text-align: right;
      line-height: 1.6;
    }}
    /* Hero */
    .hero {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-top: 4px solid var(--ink);
      box-shadow: var(--shadow);
      padding: 36px clamp(20px, 4vw, 50px) 30px;
      backdrop-filter: blur(10px);
    }}
    .eyebrow {{
      color: var(--accent);
      font-family: var(--sans);
      font-size: 11px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-weight: 700;
      margin-bottom: 12px;
    }}
    h1 {{
      margin: 0 0 14px;
      font-family: var(--serif);
      font-size: clamp(36px, 5vw, 64px);
      line-height: 0.97;
      letter-spacing: -0.05em;
      font-weight: 700;
    }}
    .lede {{
      color: #464038;
      max-width: 52rem;
      margin: 0;
      font-size: clamp(17px, 2vw, 22px);
      line-height: 1.62;
    }}
    /* KPI rail */
    .kpi-rail {{
      margin-top: 26px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .kpi {{
      padding: 14px 16px;
      border-top: 2px solid var(--line-strong);
      background: rgba(255,255,255,0.62);
      font-family: var(--sans);
    }}
    .kpi-label {{
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .kpi-value {{
      font-size: 32px;
      font-weight: 800;
      line-height: 1;
      color: var(--ink);
    }}
    .kpi-value.signal {{ color: var(--good); }}
    .kpi-sub {{
      margin-top: 6px;
      font-size: 11px;
      color: var(--muted);
    }}
    /* Card grid */
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 24px;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 24px clamp(16px, 2vw, 28px);
    }}
    .card-title {{
      margin: 0 0 4px;
      font-family: var(--sans);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
      border-bottom: 1px solid var(--line);
      padding-bottom: 10px;
      margin-bottom: 16px;
    }}
    /* Eigenvalue bar chart */
    .bar-chart {{ display: grid; gap: 10px; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 80px 1fr 60px;
      align-items: center;
      gap: 10px;
      font-family: var(--sans);
      font-size: 12px;
    }}
    .bar-label {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.10em; }}
    .bar-track {{
      height: 22px;
      background: #eee8e0;
      border-radius: 3px;
      overflow: hidden;
      position: relative;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 3px;
      transition: width 0.4s ease;
    }}
    .bar-signal {{ background: var(--good); }}
    .bar-noise  {{ background: #b8b0a6; }}
    .bar-val {{ text-align: right; font-weight: 700; color: var(--ink); font-size: 13px; }}
    /* MP line marker */
    .mp-line {{
      position: absolute;
      top: 0; bottom: 0;
      width: 2px;
      background: var(--bad);
      opacity: 0.8;
    }}
    .mp-label {{
      font-family: var(--sans);
      font-size: 10px;
      color: var(--bad);
      margin-top: 6px;
      text-align: right;
    }}
    /* Heatmap grid */
    .heatmap-wrap {{ overflow-x: auto; }}
    .heatmap-table {{ border-collapse: collapse; width: 100%; }}
    .hm-head {{
      font-family: var(--sans);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      padding: 6px 8px;
      text-align: center;
    }}
    .hm-row-label {{
      font-family: var(--sans);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--muted);
      padding: 6px 8px;
      white-space: nowrap;
    }}
    .hm-cell {{
      width: 60px;
      height: 42px;
      text-align: center;
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 600;
      border: 1px solid var(--line);
      transition: opacity 0.2s;
      cursor: default;
    }}
    .hm-cell:hover {{ opacity: 0.8; outline: 2px solid var(--accent); }}
    /* PR table */
    .pr-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .pr-table th, .pr-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      font-family: var(--sans);
    }}
    .pr-table th {{
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      text-align: left;
      background: #f8f5ef;
    }}
    .pr-bar-wrap {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .pr-bar {{
      flex: 1;
      height: 10px;
      background: #eee8e0;
      border-radius: 3px;
      overflow: hidden;
    }}
    .pr-bar-fill {{
      height: 100%;
      background: var(--accent);
      border-radius: 3px;
    }}
    /* Legend */
    .legend {{
      display: flex;
      gap: 20px;
      margin-bottom: 14px;
      font-family: var(--sans);
      font-size: 11px;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .legend-swatch {{ width: 14px; height: 14px; border-radius: 2px; }}
    /* Footer */
    .footer {{
      font-family: var(--sans);
      font-size: 11px;
      color: var(--muted);
      text-align: center;
      border-top: 1px solid var(--line);
      padding-top: 18px;
    }}
    @media (max-width: 760px) {{
      .card-grid {{ grid-template-columns: 1fr; }}
      .kpi-rail {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
<div class="page">

  <!-- MASTHEAD -->
  <div class="masthead">
    <div class="masthead-brand">EvidenceSpectral</div>
    <div class="masthead-meta">
      Random Matrix Theory<br>
      Cochrane Trust Components<br>
      n = {n_obs:,} meta-analyses
    </div>
  </div>

  <!-- HERO -->
  <div class="hero">
    <div class="eyebrow">Signal Dimensions Analysis</div>
    <h1>Two Latent<br>Dimensions Drive<br>Evidence Trust</h1>
    <p class="lede">
      Random matrix theory applied to 5&thinsp;&times;&thinsp;5 trust component correlations across
      {n_obs:,} Cochrane systematic reviews reveals {n_signal}&nbsp;signal dimensions
      (eigenvalues above the Marchenko&ndash;Pastur bulk edge &lambda;<sub>+</sub>&thinsp;=&thinsp;{lp:.3f}).
      The largest eigenvalue ({eigenvalues[0]:.3f}) is rejected as random noise with a
      Tracy&ndash;Widom statistic of s&thinsp;=&thinsp;{tw_s:.1f}, p&thinsp;{tw_p_str}.
    </p>
    <div class="kpi-rail">
      <div class="kpi">
        <div class="kpi-label">Signal Dimensions</div>
        <div class="kpi-value signal">{n_signal}</div>
        <div class="kpi-sub">Eigenvalues &gt; &lambda;<sub>+</sub> = {lp:.3f}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Largest Eigenvalue</div>
        <div class="kpi-value">{eigenvalues[0]:.3f}</div>
        <div class="kpi-sub">Above MP bound by {eigenvalues[0]-lp:.3f}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Tracy-Widom s</div>
        <div class="kpi-value">{tw_s:.1f}</div>
        <div class="kpi-sub">p {tw_p_str}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Meta-analyses</div>
        <div class="kpi-value">{n_obs:,}</div>
        <div class="kpi-sub">&gamma; = p/n = {results['mp_gamma']:.5f}</div>
      </div>
    </div>
  </div>

  <!-- ROW 1: Eigenvalue spectrum + Participation ratios -->
  <div class="card-grid">

    <!-- Eigenvalue bar chart -->
    <div class="card">
      <div class="card-title">Eigenvalue Spectrum &mdash; Marchenko&ndash;Pastur Bounds</div>
      <div class="legend">
        <div class="legend-item">
          <div class="legend-swatch" style="background:var(--good)"></div>
          <span>Signal (&gt;&thinsp;&lambda;<sub>+</sub>)</span>
        </div>
        <div class="legend-item">
          <div class="legend-swatch" style="background:#b8b0a6"></div>
          <span>Noise bulk</span>
        </div>
        <div class="legend-item">
          <div class="legend-swatch" style="background:var(--bad); height:4px; width:14px; border-radius:0;"></div>
          <span>MP upper bound (&lambda;<sub>+</sub> = {lp:.3f})</span>
        </div>
      </div>
      <div class="bar-chart" id="eigenBarChart"></div>
      <p class="mp-label">MP &lambda;<sub>+</sub> = {lp:.3f} &nbsp;|&nbsp; MP &lambda;<sub>&minus;</sub> = {lm:.3f}</p>
    </div>

    <!-- Participation ratio table -->
    <div class="card">
      <div class="card-title">Participation Ratios &mdash; Eigenvector Spread (range 1 to 5)</div>
      <table class="pr-table" id="prTable">
        <thead>
          <tr>
            <th>Eigenvector</th>
            <th>&lambda;</th>
            <th>PR</th>
            <th>Spread</th>
          </tr>
        </thead>
        <tbody id="prTbody"></tbody>
      </table>
    </div>

  </div>

  <!-- ROW 2: Eigenvector heatmap + Correlation heatmap -->
  <div class="card-grid">

    <!-- Eigenvector heatmap (5x5) -->
    <div class="card">
      <div class="card-title">Eigenvector Loadings Heatmap (components &times; eigenvectors)</div>
      <div class="heatmap-wrap">
        <table class="heatmap-table" id="evTable"></table>
      </div>
    </div>

    <!-- Correlation matrix heatmap -->
    <div class="card">
      <div class="card-title">Trust Component Correlation Matrix</div>
      <div class="heatmap-wrap">
        <table class="heatmap-table" id="corrTable"></table>
      </div>
    </div>

  </div>

  <div class="footer">
    EvidenceSpectral &mdash; Random Matrix Theory analysis of Cochrane trust components &mdash;
    {n_obs:,} meta-analyses, p=5 components, &gamma;={results['mp_gamma']:.5f}
  </div>

</div>

<script>
(function() {{
  'use strict';

  const EIGENVALUES = {eigenvalues_js};
  const CORR_MAT    = {corr_mat_js};
  const EV_GRID     = {ev_grid_js};   // [comp][ev]
  const PRS         = {prs_js};
  const LABELS      = {labels_js};
  const SHORT       = {short_js};
  const MP_PLUS     = {lp:.6f};
  const N_SIGNAL    = {n_signal};

  // --- Colour helpers ---
  function heatColor(v, vmin, vmax) {{
    // Blue (negative) → white (0) → red (positive)
    var mid = (vmin + vmax) / 2;
    var t;
    var r, g, b;
    if (v < 0) {{
      t = (v - vmin) / (0 - vmin + 1e-9);
      t = Math.max(0, Math.min(1, t));
      r = Math.round(50 + 150 * t);
      g = Math.round(104 + 140 * t);
      b = Math.round(145 + 110 * (1 - t) + 145 * t);
      // Simplify: blue to white
      r = Math.round(255 * t + 50 * (1 - t));
      g = Math.round(255 * t + 104 * (1 - t));
      b = Math.round(255 * t + 200 * (1 - t));
    }} else {{
      t = v / (vmax + 1e-9);
      t = Math.max(0, Math.min(1, t));
      r = Math.round(255 * (1 - t) + 146 * t);
      g = Math.round(255 * (1 - t) + 43 * t);
      b = Math.round(255 * (1 - t) + 33 * t);
    }}
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }}

  function textColor(bg) {{
    // Parse rgb components
    var m = bg.match(/rgb\\((\\d+),(\\d+),(\\d+)\\)/);
    if (!m) return '#000';
    var lum = 0.299 * +m[1] + 0.587 * +m[2] + 0.114 * +m[3];
    return lum > 140 ? '#111' : '#fff';
  }}

  // --- Eigenvalue bar chart ---
  (function buildBarChart() {{
    var maxVal = Math.max.apply(null, EIGENVALUES) * 1.08;
    var container = document.getElementById('eigenBarChart');
    EIGENVALUES.forEach(function(ev, i) {{
      var isSignal = ev > MP_PLUS;
      var pct = (ev / maxVal * 100).toFixed(1);
      var mpPct = (MP_PLUS / maxVal * 100).toFixed(1);
      var row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML =
        '<div class="bar-label">EV' + (i + 1) + '</div>' +
        '<div class="bar-track">' +
          '<div class="bar-fill ' + (isSignal ? 'bar-signal' : 'bar-noise') +
            '" style="width:' + pct + '%"></div>' +
          '<div class="mp-line" style="left:' + mpPct + '%"></div>' +
        '</div>' +
        '<div class="bar-val">' + ev.toFixed(3) + '</div>';
      container.appendChild(row);
    }});
  }})();

  // --- Participation ratio table ---
  (function buildPRTable() {{
    var tbody = document.getElementById('prTbody');
    PRS.forEach(function(pr, i) {{
      var isSignal = EIGENVALUES[i] > MP_PLUS;
      var pct = ((pr / 5) * 100).toFixed(1);
      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td style="font-family:var(--mono);font-size:12px;">' +
          (isSignal ? '<strong>' : '') +
          'EV' + (i + 1) +
          (isSignal ? '</strong>' : '') +
        '</td>' +
        '<td style="font-family:var(--mono);font-size:12px;">' + EIGENVALUES[i].toFixed(3) +
          (isSignal ? ' <span style="color:var(--good);font-size:9px;text-transform:uppercase;letter-spacing:0.1em;font-family:var(--sans)">signal</span>' : '') +
        '</td>' +
        '<td style="font-family:var(--mono);font-size:13px;font-weight:700;">' + pr.toFixed(3) + '</td>' +
        '<td>' +
          '<div class="pr-bar-wrap">' +
            '<div class="pr-bar"><div class="pr-bar-fill" style="width:' + pct + '%"></div></div>' +
          '</div>' +
        '</td>';
      tbody.appendChild(tr);
    }});
  }})();

  // --- Generic heatmap builder ---
  function buildHeatmap(tableId, data, rowLabels, colLabels, vmin, vmax) {{
    var table = document.getElementById(tableId);
    // Header row
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    headRow.appendChild(document.createElement('th'));
    colLabels.forEach(function(lbl) {{
      var th = document.createElement('th');
      th.className = 'hm-head';
      th.textContent = lbl;
      headRow.appendChild(th);
    }});
    thead.appendChild(headRow);
    table.appendChild(thead);

    // Body rows
    var tbody = document.createElement('tbody');
    data.forEach(function(row, ri) {{
      var tr = document.createElement('tr');
      var th = document.createElement('th');
      th.className = 'hm-row-label';
      th.textContent = rowLabels[ri];
      tr.appendChild(th);
      row.forEach(function(val) {{
        var td = document.createElement('td');
        td.className = 'hm-cell';
        var bg = heatColor(val, vmin, vmax);
        td.style.background = bg;
        td.style.color = textColor(bg);
        td.textContent = val.toFixed(2);
        td.title = val.toFixed(4);
        tr.appendChild(td);
      }});
      tbody.appendChild(tr);
    }});
    table.appendChild(tbody);
  }}

  // Eigenvector heatmap: rows = components, cols = EV1..EV5
  var evColLabels = EIGENVALUES.map(function(v, i) {{ return 'EV' + (i+1); }});
  buildHeatmap('evTable', EV_GRID, SHORT, evColLabels, -1.0, 1.0);

  // Correlation heatmap: rows = cols = components
  buildHeatmap('corrTable', CORR_MAT, SHORT, SHORT, -1.0, 1.0);

}})();
</script>
</body>
</html>"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Dashboard written: {OUT_HTML}")
print(f"Lines: {html.count(chr(10))}")
