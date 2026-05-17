"""Étape 12 — Génération de index.html

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 12 — Générer index.html dans le dossier du projet
# ══════════════════════════════════════════════════════════

html_code = '''<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AgroShield</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family:system-ui,sans-serif; background:#f0f4f0; color:#1e293b; }
    nav  { background:#1a3a1a; padding:0 24px; display:flex; align-items:center;
           justify-content:space-between; height:56px; }
    .logo { color:#fff; font-size:20px; font-weight:700; }
    .logo span { color:#4ade80; }
    .tabs { display:flex; gap:4px; }
    .tab  { background:none; border:none; color:#94a3b8; padding:8px 16px;
            border-radius:6px; cursor:pointer; font-size:14px; transition:all .2s; }
    .tab.active { background:#2d5a2d; color:#4ade80; }
    .mode-badge { background:#f59e0b; color:#78350f; font-size:11px;
                  font-weight:700; padding:3px 10px; border-radius:20px; }
    .page { display:none; padding:24px; max-width:1100px; margin:0 auto; }
    .page.active { display:block; }
    .card { background:#fff; border-radius:12px; padding:20px;
            box-shadow:0 1px 4px rgba(0,0,0,.08); margin-bottom:16px; }
    .card-title { font-size:15px; font-weight:600; color:#475569; margin-bottom:14px; }
    .sensor-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
    .sensor-card { background:#fff; border-radius:10px; padding:16px;
                   box-shadow:0 1px 4px rgba(0,0,0,.08); border-left:4px solid #e2e8f0; }
    .sensor-card.alert { border-left-color:#ef4444; background:#fff5f5; }
    .s-icon  { font-size:28px; margin-bottom:6px; }
    .s-label { font-size:12px; color:#64748b; }
    .s-value { font-size:28px; font-weight:700; }
    .sensor-card.alert .s-value { color:#ef4444; }
    .alert-tag { display:inline-block; background:#fef2f2; color:#ef4444;
                 font-size:10px; font-weight:700; padding:2px 8px; border-radius:10px; margin-top:4px; }
    .servo-row  { display:flex; align-items:center; justify-content:space-between;
                  padding:10px 0; border-bottom:1px solid #f1f5f9; }
    .servo-row:last-child { border-bottom:none; }
    .servo-name  { font-size:14px; color:#475569; }
    .servo-badge { padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }
    .servo-A { background:#dcfce7; color:#166534; }
    .servo-B { background:#f1f5f9; color:#64748b; }
    .upload-zone { border:2px dashed #4ade80; border-radius:12px; padding:40px;
                   text-align:center; cursor:pointer; background:#f0fdf4; transition:all .2s; }
    .upload-zone:hover { background:#dcfce7; border-color:#16a34a; }
    .upload-zone input { display:none; }
    .u-icon { font-size:48px; margin-bottom:12px; }
    .u-text { font-size:16px; color:#475569; margin-bottom:6px; }
    .u-hint { font-size:13px; color:#94a3b8; }
    .preview-img { max-width:300px; max-height:300px; border-radius:10px;
                   display:block; margin:16px auto; box-shadow:0 2px 8px rgba(0,0,0,.15); }
    .btn { padding:12px 28px; border-radius:8px; border:none; cursor:pointer;
           font-size:15px; font-weight:600; transition:all .2s; }
    .btn-primary { background:#16a34a; color:#fff; }
    .btn-primary:hover { background:#15803d; }
    .btn-primary:disabled { background:#94a3b8; cursor:not-allowed; }
    .result-box  { border-radius:10px; padding:20px; margin-top:16px; }
    .r-healthy   { background:#f0fdf4; border-left:5px solid #22c55e; }
    .r-moderate  { background:#fffbeb; border-left:5px solid #f59e0b; }
    .r-critical  { background:#fff5f5; border-left:5px solid #ef4444; }
    .r-plant     { font-size:20px; font-weight:700; margin-bottom:4px; }
    .r-disease   { font-size:16px; font-weight:600; margin-bottom:10px; }
    .r-meta      { font-size:13px; color:#64748b; }
    .agree-badge { display:inline-block; padding:4px 12px; border-radius:20px;
                   font-size:13px; font-weight:700; margin-right:8px; }
    .agree-ok    { background:#dcfce7; color:#166534; }
    .agree-no    { background:#fef3c7; color:#92400e; }
    .models-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
    .model-card  { background:#f8fafc; border-radius:8px; padding:12px; }
    .m-name      { font-size:12px; color:#64748b; margin-bottom:4px; }
    .m-pred      { font-size:14px; font-weight:600; }
    .conf-bar    { background:#e2e8f0; border-radius:4px; height:6px; margin-top:6px; overflow:hidden; }
    .conf-fill   { height:100%; border-radius:4px; background:#16a34a; transition:width .5s; }
    .spread-info { background:#fffbeb; border:1px solid #f59e0b; border-radius:8px;
                   padding:12px; margin-top:10px; font-size:14px; color:#92400e; }
    .det-list    { margin-top:16px; }
    .det-item    { display:flex; align-items:center; gap:12px; padding:12px;
                   background:#fff; border-radius:8px; margin-bottom:8px;
                   box-shadow:0 1px 3px rgba(0,0,0,.06); }
    .det-dot     { width:14px; height:14px; border-radius:50%; flex-shrink:0; }
    .det-info    { flex:1; }
    .det-title   { font-size:14px; font-weight:600; }
    .det-meta    { font-size:12px; color:#64748b; margin-top:2px; }
    .det-conf    { font-size:13px; font-weight:700; }
    .stats-row   { display:flex; gap:12px; margin-bottom:16px; }
    .stat-pill   { background:#fff; border-radius:8px; padding:10px 16px;
                   font-size:13px; font-weight:600; box-shadow:0 1px 3px rgba(0,0,0,.06); }
    .spinner     { display:inline-block; width:18px; height:18px;
                   border:3px solid #fff; border-top-color:transparent;
                   border-radius:50%; animation:spin .7s linear infinite; vertical-align:middle; }
    @keyframes spin { to { transform:rotate(360deg); } }
    .empty-state { text-align:center; padding:60px 20px; color:#94a3b8; }
    .empty-state .e-icon { font-size:48px; margin-bottom:12px; }
    .hidden { display:none !important; }
    .zone-input { flex:1; padding:10px; border:1px solid #e2e8f0;
                  border-radius:8px; font-size:14px; }
    .row { display:flex; gap:10px; align-items:center; margin-top:14px; }
  </style>
</head>
<body>

<nav>
  <div class="logo">Agro<span>Shield</span></div>
  <div class="tabs">
    <button class="tab active" onclick="showPage(\'dashboard\', this)">📊 Capteurs</button>
    <button class="tab" onclick="showPage(\'scanner\', this)">📷 Scanner</button>
    <button class="tab" onclick="showPage(\'map\', this)">🗺 Détections</button>
  </div>
  <span class="mode-badge" id="mode-badge">⚡ Simulation</span>
</nav>

<!-- PAGE CAPTEURS -->
<div class="page active" id="page-dashboard">
  <div class="sensor-grid" id="sensor-grid">
    <div class="sensor-card" id="c-temp">
      <div class="s-icon">🌡️</div>
      <div class="s-label">Température</div>
      <div class="s-value" id="v-temp">--</div>
    </div>
    <div class="sensor-card" id="c-hum">
      <div class="s-icon">💧</div>
      <div class="s-label">Humidité</div>
      <div class="s-value" id="v-hum">--</div>
    </div>
    <div class="sensor-card" id="c-lux">
      <div class="s-icon">☀️</div>
      <div class="s-label">Luminosité</div>
      <div class="s-value" id="v-lux">--</div>
    </div>
  </div>
  <div class="card" style="margin-top:16px">
    <div class="card-title">État des plaques</div>
    <div id="servo-list"></div>
  </div>
  <p style="color:#94a3b8;font-size:12px;text-align:center;margin-top:8px">
    Rafraîchissement toutes les 5s — mode simulation (les valeurs varient aléatoirement)
  </p>
</div>

<!-- PAGE SCANNER -->
<div class="page hidden" id="page-scanner">
  <div class="card">
    <div class="card-title">Analyser une feuille</div>
    <label class="upload-zone" id="upload-zone">
      <input type="file" id="file-input" accept="image/*">
      <div class="u-icon">🌿</div>
      <div class="u-text">Cliquez ou glissez une photo de feuille</div>
      <div class="u-hint">JPG, PNG — une seule feuille bien éclairée et centrée</div>
    </label>
    <img id="preview" class="preview-img hidden" alt="Aperçu">
    <div class="row">
      <input class="zone-input" id="zone-id" placeholder="Identifiant zone (ex: zone_nord_A1)">
      <button class="btn btn-primary" id="analyze-btn" onclick="analyzeImage()" disabled>Analyser</button>
    </div>
    <div id="result-area" class="hidden"></div>
  </div>
</div>

<!-- PAGE CARTE DÉTECTIONS -->
<div class="page hidden" id="page-map">
  <div class="stats-row" id="map-stats">
    <div class="stat-pill">🔍 0 zones scannées</div>
    <div class="stat-pill" style="color:#ef4444">⚠ 0 infectées</div>
  </div>
  <div class="card">
    <div class="card-title">Historique des diagnostics</div>
    <div id="det-list">
      <div class="empty-state">
        <div class="e-icon">🗺</div>
        <div>Aucune détection — utilisez le Scanner</div>
      </div>
    </div>
  </div>
  <button class="btn" style="background:#ef4444;color:#fff" onclick="clearDetections()">🗑 Effacer tout</button>
</div>

<script>
// ════════════════════════════════════════════════
// CONFIG — UNE SEULE LIGNE à changer pour le RPi
// ════════════════════════════════════════════════
const API = \'http://localhost:5000/api\';
// Sur Raspberry Pi :
// const API = \'http://192.168.1.100:5000/api\';

// ── Navigation ──────────────────────────────────
function showPage(name, btn) {
  document.querySelectorAll(\'.page\').forEach(p => { p.classList.remove(\'active\'); p.classList.add(\'hidden\'); });
  document.querySelectorAll(\'.tab\').forEach(t => t.classList.remove(\'active\'));
  const pg = document.getElementById(\'page-\' + name);
  pg.classList.remove(\'hidden\'); pg.classList.add(\'active\');
  btn.classList.add(\'active\');
  if (name === \'map\') loadMap();
}

// ── Capteurs ────────────────────────────────────
async function fetchSensors() {
  try {
    const r = await fetch(API + \'/sensors\');
    const d = await r.json();
    document.getElementById(\'v-temp\').textContent = d.temperature + \'°C\';
    document.getElementById(\'v-hum\').textContent  = d.humidity    + \' %\';
    document.getElementById(\'v-lux\').textContent  = d.luminosity  + \'lux\';
    [[\'temp\',\'temperature\'],[\'hum\',\'humidity\'],[\'lux\',\'luminosity\']].forEach(([k,key]) => {
      const card = document.getElementById(\'c-\'+k);
      const isA  = d.alerts[key];
      card.classList.toggle(\'alert\', isA);
      const ex = card.querySelector(\'.alert-tag\');
      if (isA && !ex) { const t=document.createElement(\'span\'); t.className=\'alert-tag\'; t.textContent=\'⚠ ALERTE\'; card.appendChild(t); }
      else if (!isA && ex) ex.remove();
    });
    const badge = document.getElementById(\'mode-badge\');
    badge.textContent = d.mode === \'simulation\' ? \'⚡ Simulation\' : \'🍓 Raspberry Pi\';
    badge.style.background = d.mode === \'simulation\' ? \'#f59e0b\' : \'#22c55e\';
    document.getElementById(\'servo-list\').innerHTML =
      [[\'servo1_pos\',\'Plaque 1\'],[\'servo2_pos\',\'Plaque 2\'],[\'servo3_pos\',\'Plaque 3\']].map(([k,n]) =>
        `<div class="servo-row"><span class="servo-name">${n}</span>
         <span class="servo-badge servo-${d[k]}">${d[k]===\'A\' ? \'✅ Déployée 90°\' : \'⬜ Repos 0°\'}</span></div>`
      ).join(\'\');
  } catch(e) {
    document.getElementById(\'mode-badge\').textContent = \'❌ Serveur hors ligne\';
    document.getElementById(\'mode-badge\').style.background = \'#ef4444\';
  }
}
fetchSensors();
setInterval(fetchSensors, 5000);

// ── Scanner ──────────────────────────────────────
const fileInput = document.getElementById(\'file-input\');
fileInput.addEventListener(\'change\', e => {
  if (!e.target.files[0]) return;
  const reader = new FileReader();
  reader.onload = ev => {
    const img = document.getElementById(\'preview\');
    img.src = ev.target.result;
    img.classList.remove(\'hidden\');
    document.getElementById(\'analyze-btn\').disabled = false;
    document.getElementById(\'result-area\').classList.add(\'hidden\');
  };
  reader.readAsDataURL(e.target.files[0]);
});

const dropZone = document.getElementById(\'upload-zone\');
dropZone.addEventListener(\'dragover\', e => { e.preventDefault(); dropZone.style.background=\'#dcfce7\'; });
dropZone.addEventListener(\'dragleave\', () => dropZone.style.background=\'#f0fdf4\');
dropZone.addEventListener(\'drop\', e => {
  e.preventDefault(); dropZone.style.background=\'#f0fdf4\';
  if (e.dataTransfer.files[0]) { fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event(\'change\')); }
});

function getSeverity(disease) {
  if (!disease || disease.toLowerCase().includes(\'healthy\')) return \'healthy\';
  const crit = [\'Late blight\',\'Bacterial spot\',\'Yellow Leaf Curl\',\'mosaic virus\'];
  return crit.some(c => disease.includes(c)) ? \'critical\' : \'moderate\';
}

async function analyzeImage() {
  const file = fileInput.files[0];
  if (!file) return;
  const btn = document.getElementById(\'analyze-btn\');
  btn.innerHTML = \'<span class="spinner"></span> Analyse en cours...\';
  btn.disabled = true;
  const form = new FormData();
  form.append(\'image\', file);
  form.append(\'zone_id\', document.getElementById(\'zone-id\').value || \'zone_\'+Date.now());
  form.append(\'lat\', \'33.5731\');
  form.append(\'lng\', \'-7.5898\');
  try {
    const res  = await fetch(API+\'/analyze\', {method:\'POST\', body:form});
    const data = await res.json();
    const sev  = getSeverity(data.disease);
    const colors = {healthy:\'#22c55e\',moderate:\'#f59e0b\',critical:\'#ef4444\'};
    const color  = colors[sev];
    const conf   = (data.confidence*100).toFixed(1);
    const eff    = data.efficientnet;
    const rnet   = data.resnet;
    let spreadHtml = \'\';
    if (data.zone && data.zone.spread_radius_meters > 0) {
      spreadHtml = `<div class="spread-info">⚠ Zone de propagation estimée : <strong>${data.zone.spread_radius_meters}m</strong> — traitez les zones adjacentes.</div>`;
    }
    document.getElementById(\'result-area\').innerHTML = `
      <div class="result-box r-${sev}">
        <div class="r-plant">${data.plant}</div>
        <div class="r-disease" style="color:${color}">${data.disease}</div>
        <div class="r-meta">
          <span class="agree-badge ${data.agreement?\'agree-ok\':\'agree-no\'}">
            Accord ${data.agreement_score}
          </span>
          Confiance : <strong>${conf}%</strong>
        </div>
        <div class="models-grid">
          <div class="model-card">
            <div class="m-name">EfficientNet-B0</div>
            <div class="m-pred">${eff.prediction.replace(/__/g,\' — \').replace(/_/g,\' \')}</div>
            <div class="conf-bar"><div class="conf-fill" style="width:${(eff.confidence*100).toFixed(0)}%"></div></div>
            <div style="font-size:11px;color:#64748b;margin-top:3px">${(eff.confidence*100).toFixed(1)}%</div>
          </div>
          <div class="model-card">
            <div class="m-name">ResNet-50</div>
            <div class="m-pred">${rnet.prediction.replace(/__/g,\' — \').replace(/_/g,\' \')}</div>
            <div class="conf-bar"><div class="conf-fill" style="width:${(rnet.confidence*100).toFixed(0)}%"></div></div>
            <div style="font-size:11px;color:#64748b;margin-top:3px">${(rnet.confidence*100).toFixed(1)}%</div>
          </div>
        </div>
        ${spreadHtml}
      </div>`;
    document.getElementById(\'result-area\').classList.remove(\'hidden\');
  } catch(e) {
    document.getElementById(\'result-area\').innerHTML =
      `<div class="result-box" style="background:#fff5f5;border-left:5px solid #ef4444;margin-top:16px">
        ❌ Erreur : ${e.message}<br>Vérifiez que server.py tourne sur localhost:5000
      </div>`;
    document.getElementById(\'result-area\').classList.remove(\'hidden\');
  }
  btn.textContent = \'Analyser\';
  btn.disabled = false;
}

// ── Carte des détections ─────────────────────────────
async function loadMap() {
  try {
    const res  = await fetch(API+\'/disease_map\');
    const data = await res.json();
    const det  = data.detections || [];
    document.getElementById(\'map-stats\').innerHTML =
      `<div class="stat-pill">🔍 ${data.total_scanned} zones scannées</div>
       <div class="stat-pill" style="color:#ef4444">⚠ ${data.infected_zones} infectées</div>`;
    const colors = {healthy:\'#22c55e\',moderate:\'#f59e0b\',critical:\'#ef4444\'};
    document.getElementById(\'det-list\').innerHTML = det.length === 0
      ? \'<div class="empty-state"><div class="e-icon">🗺</div><div>Aucune détection — utilisez le Scanner</div></div>\'  
      : det.slice().reverse().map(d => {
          const sev   = getSeverity(d.disease);
          const color = colors[sev];
          const t     = new Date(d.timestamp).toLocaleTimeString(\'fr-FR\');
          return `<div class="det-item">
            <div class="det-dot" style="background:${color}"></div>
            <div class="det-info">
              <div class="det-title">${d.plant} — ${d.disease}</div>
              <div class="det-meta">Zone : ${d.zone_id} | ${t}
                ${d.spread_radius_meters > 0
                  ? ` | <span style="color:#f59e0b">⚠ propagation ${d.spread_radius_meters}m</span>` : \'\'}
              </div>
            </div>
            <div class="det-conf" style="color:${color}">${(d.confidence*100).toFixed(1)}%</div>
          </div>`;
        }).join(\'\')
  } catch(e) {}
}

async function clearDetections() {
  if (!confirm(\'Effacer toutes les détections ?\')) return;
  await fetch(API+\'/detections/clear\', {method:\'POST\'});
  loadMap();
}
</script>
</body>
</html>'''

html_path = PROJECT_DIR / 'index.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_code)

print(f'✅ index.html généré : {html_path}')
print()
print('═' * 55)
print('RÉCAPITULATIF — COMMENT LANCER L\'APPLICATION')
print('═' * 55)
print()
print('1. Ouvrez un terminal VS Code   (Ctrl + `)')
print(f'2. cd "{PROJECT_DIR}"')
print('3. python server.py')
print('4. Ouvrez index.html dans votre navigateur')
print('   (double-clic ou Ctrl+O dans le navigateur)')
print()
print('Structure finale du projet :')
for item in sorted(PROJECT_DIR.iterdir()):
    if item.is_dir():
        print(f'  {item.name}/')
        for sub in sorted(item.iterdir()):
            print(f'    {sub.name}')
    else:
        print(f'  {item.name}')