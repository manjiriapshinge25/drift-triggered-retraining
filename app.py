
import sys
import json
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.core import load_production_model


app = FastAPI(
    title="DriftGuard | MLOps Control Center",
    description="""
## ✨ DriftGuard — Smart MLOps

Drift-triggered model retraining and customer churn prediction.

- 🟢 **Health:** Check the active production model.
- 🔮 **Prediction:** Test churn predictions using customer features.
- 📚 **Documentation:** Explore and execute API endpoints.
- 🔄 **Model lifecycle:** Production model is resolved from the registry.
""",
    version="1.0",
    docs_url=None,
    redoc_url="/redoc",
)


@app.get("/docs", include_in_schema=False)
def custom_docs():
    page = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="DriftGuard | MLOps Control Center",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "displayRequestDuration": True,
            "filter": True,
            "docExpansion": "list",
            "persistAuthorization": True,
        },
    )

    html = page.body.decode("utf-8")

    style = """

<style>
body {
    background: #f8fafc !important;
    color: #1e293b !important;
}

.swagger-ui {
    font-family: "Segoe UI", Arial, sans-serif;
}

.swagger-ui .topbar {
    background: linear-gradient(100deg,#6d28d9,#9333ea,#0891b2) !important;
    padding: 20px 0 !important;
    border-bottom: 3px solid #67e8f9;
}

.swagger-ui .topbar-wrapper .link span {
    color: white !important;
    font-size: 25px;
    font-weight: 900;
}

.swagger-ui .info h1,
.swagger-ui .info h2,
.swagger-ui .info h3,
.swagger-ui .info p,
.swagger-ui .info li,
.swagger-ui .opblock-tag {
    color: #1e293b !important;
}

.swagger-ui .info {
    margin: 35px 0;
}

.swagger-ui .info .title {
    color: #6d28d9 !important;
    font-size: 36px;
}

.swagger-ui .scheme-container,
.swagger-ui section.models {
    background: white !important;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    box-shadow: 0 8px 25px #64748b18;
}

.swagger-ui .opblock {
    border-radius: 14px !important;
    overflow: hidden;
    margin: 18px 0 !important;
    box-shadow: 0 6px 20px #64748b20;
    transition: transform .2s, box-shadow .2s;
}

.swagger-ui .opblock:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 26px #64748b30;
}

.swagger-ui .opblock.opblock-get {
    background: #ecfeff !important;
    border: 1px solid #06b6d4 !important;
}

.swagger-ui .opblock.opblock-post {
    background: #ecfdf5 !important;
    border: 1px solid #10b981 !important;
}

.swagger-ui .opblock .opblock-summary {
    border-color: #dbeafe !important;
    padding: 12px;
}

.swagger-ui .opblock .opblock-summary-method {
    border-radius: 8px !important;
    min-width: 82px;
    font-weight: 900;
}

.swagger-ui .opblock-summary-description,
.swagger-ui .opblock .opblock-summary-path,
.swagger-ui .opblock .opblock-summary-path__deprecated {
    color: #334155 !important;
}

.swagger-ui .opblock .opblock-section-header {
    background: #f1f5f9 !important;
}

.swagger-ui .opblock-description-wrapper p,
.swagger-ui .response-col_description,
.swagger-ui .parameter__name,
.swagger-ui .parameter__type,
.swagger-ui label,
.swagger-ui .tab li,
.swagger-ui .response-col_status {
    color: #334155 !important;
}

.swagger-ui .btn {
    border-radius: 9px !important;
}

.swagger-ui .btn.execute {
    background: linear-gradient(100deg,#7c3aed,#0891b2) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 15px #7c3aed30;
}

.swagger-ui input,
.swagger-ui textarea,
.swagger-ui select {
    background: white !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 7px;
}

.swagger-ui a {
    color: #0891b2 !important;
}

.swagger-ui table thead tr td,
.swagger-ui table thead tr th {
    color: #6d28d9 !important;
}

.swagger-ui .model-box,
.swagger-ui .model {
    color: #334155 !important;
}

.swagger-ui .filter input {
    background: white !important;
    color: #1e293b !important;
}

.swagger-ui .responses-inner h4,
.swagger-ui .responses-inner h5 {
    color: #6d28d9 !important;
}

@media (max-width: 600px) {
    .swagger-ui .info .title {
        font-size: 27px;
    }

    .swagger-ui .wrapper {
        padding: 0 12px;
    }
}
</style>

    """

    html = html.replace("</head>", style + "</head>")
    return HTMLResponse(html)


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def home():
    return HTMLResponse(r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DriftGuard | Churn Prediction</title>
<style>
:root{--green:#66856e;--green-dark:#45634d;--mint:#e9f1e9;--ink:#243128;--muted:#6d7b70;--line:#e2e9e2;--bg:#f7f9f6;--white:#fff;--amber:#f4ead3}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Segoe UI",Arial,sans-serif}
.topbar{height:74px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 clamp(20px,5vw,72px)}
.brand{display:flex;align-items:center;gap:12px;font-size:19px;font-weight:800;letter-spacing:-.4px}.logo{width:38px;height:38px;border-radius:12px;background:var(--green);display:grid;place-items:center;color:white;font-size:20px}
.status{display:flex;align-items:center;gap:8px;color:var(--green-dark);font-size:13px;font-weight:700;background:var(--mint);padding:9px 13px;border-radius:999px}.dot{width:8px;height:8px;border-radius:50%;background:#5a9b67}
main{max-width:1120px;margin:0 auto;padding:44px 24px 60px}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:800;color:var(--green);margin-bottom:12px}
.hero{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;margin-bottom:30px}.hero h1{font-size:clamp(30px,4vw,44px);letter-spacing:-1.6px;margin:0 0 12px;line-height:1.1}.hero p{color:var(--muted);font-size:15px;line-height:1.7;margin:0;max-width:650px}.version{background:#fff;border:1px solid var(--line);padding:15px 18px;border-radius:16px;min-width:160px}.version span{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:7px}.version strong{font-size:19px}
.layout{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(290px,.8fr);gap:22px;align-items:start}.card{background:var(--white);border:1px solid var(--line);border-radius:22px;box-shadow:0 8px 28px #24312808;overflow:hidden}.card-head{padding:24px 26px 18px;border-bottom:1px solid #edf1ed}.card-head h2{font-size:18px;margin:0 0 7px;letter-spacing:-.4px}.card-head p{margin:0;color:var(--muted);font-size:13px;line-height:1.5}.form-body{padding:23px 26px 26px}.section-label{font-size:11px;text-transform:uppercase;letter-spacing:1.3px;font-weight:800;color:var(--green);margin:0 0 15px}.fields{display:grid;grid-template-columns:1fr 1fr;gap:17px 15px;margin-bottom:25px}.field label{display:block;font-size:12px;font-weight:700;color:#465449;margin-bottom:8px}.field input,.field select{width:100%;border:1px solid #dce5dc;background:#fbfcfa;border-radius:11px;padding:12px 12px;font-size:14px;color:var(--ink);outline:none;min-height:44px}.field input:focus,.field select:focus{border-color:var(--green);box-shadow:0 0 0 3px #66856e20}.submit{width:100%;border:0;background:var(--green);color:#fff;font-weight:800;font-size:14px;border-radius:12px;padding:15px;cursor:pointer;transition:.2s}.submit:hover{background:var(--green-dark);transform:translateY(-1px)}.submit:disabled{opacity:.65;cursor:wait;transform:none}.small-note{font-size:11px;color:var(--muted);line-height:1.6;margin:12px 0 0;text-align:center}
.side{display:grid;gap:18px}.result-card{padding:24px;background:linear-gradient(145deg,#f0f5ef,#fff 70%)}.result-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}.result-top h2{margin:0;font-size:17px}.pill{font-size:11px;font-weight:800;color:var(--green-dark);background:#dfeade;border-radius:999px;padding:7px 10px}.empty{border:1px dashed #cbd8cb;border-radius:16px;padding:26px 18px;text-align:center;background:#ffffffb5}.empty-icon{font-size:29px;margin-bottom:9px}.empty strong{display:block;font-size:14px;margin-bottom:7px}.empty p{font-size:12px;color:var(--muted);line-height:1.6;margin:0}.result{display:none}.result-label{font-size:11px;text-transform:uppercase;letter-spacing:1.2px;color:var(--muted);font-weight:800}.result-value{font-size:28px;font-weight:850;letter-spacing:-.7px;margin:7px 0 16px}.meter{height:9px;background:#e4eae3;border-radius:99px;overflow:hidden}.meter-fill{height:100%;width:0;background:var(--green);border-radius:99px;transition:width .4s}.meter-meta{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-top:8px}.result-box{background:white;border:1px solid var(--line);border-radius:15px;padding:16px;margin-top:18px}.result-box p{font-size:12px;color:var(--muted);margin:8px 0 0;line-height:1.5}.error{display:none;background:#fff0ed;border:1px solid #f3c8bf;color:#8d3326;padding:12px;border-radius:11px;font-size:12px;line-height:1.5;margin-top:14px;white-space:pre-wrap}.info-card{padding:22px 24px}.info-card h3{font-size:14px;margin:0 0 14px}.info-row{display:flex;gap:12px;margin:13px 0;align-items:flex-start}.info-icon{width:30px;height:30px;flex:0 0 30px;border-radius:10px;background:var(--mint);display:grid;place-items:center;color:var(--green-dark)}.info-row strong{font-size:12px;display:block;margin:1px 0 4px}.info-row span{font-size:11px;color:var(--muted);line-height:1.5;display:block}.links{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}.links a{font-size:12px;color:var(--green-dark);font-weight:700;text-decoration:none;border:1px solid var(--line);background:white;padding:10px 13px;border-radius:10px}.links a:hover{border-color:var(--green)}
footer{max-width:1120px;margin:auto;padding:0 24px 28px;color:#89958b;font-size:11px;display:flex;justify-content:space-between;gap:15px}

.dashboard-intro,.dashboard-section{margin-top:38px}.section-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin:0 0 18px}.section-heading h2{font-size:24px;letter-spacing:-.7px;margin:0 0 7px}.section-heading p{margin:0;color:var(--muted);font-size:13px;line-height:1.6}.refresh-btn{border:1px solid var(--line);background:#fff;color:var(--green-dark);font-weight:800;padding:10px 14px;border-radius:11px;cursor:pointer}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.metric-card{background:#fff;border:1px solid var(--line);border-radius:17px;padding:19px;box-shadow:0 8px 24px #24312806}.metric-icon{width:35px;height:35px;border-radius:11px;background:var(--mint);color:var(--green-dark);display:grid;place-items:center;font-size:18px;margin-bottom:15px}.metric-card span{display:block;font-size:12px;color:var(--muted);font-weight:650}.metric-card strong{display:block;font-size:25px;letter-spacing:-.8px;margin:7px 0}.metric-card small{font-size:11px;color:var(--muted)}.section-tag{white-space:nowrap;background:var(--mint);color:var(--green-dark);padding:8px 11px;border-radius:99px;font-size:11px;font-weight:800}.table-card{border-radius:17px}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:700px;text-align:left;font-size:12px}th{background:#f6f8f5;color:#718074;text-transform:uppercase;letter-spacing:.7px;font-size:10px;padding:14px 16px}td{padding:14px 16px;border-top:1px solid #edf1ed;color:#405044;white-space:nowrap}tbody tr:hover{background:#fbfcfa}.status-chip{display:inline-block;padding:5px 8px;border-radius:99px;font-size:10px;font-weight:800;background:#e9f1e9;color:#45634d}.status-chip.high{background:#fff0e7;color:#9b542b}.status-chip.muted{background:#f0f2ef;color:#657166}.note-banner{margin-top:14px;padding:14px 16px;border-radius:13px;background:#eef4ed;color:#586b5b;font-size:12px;line-height:1.6}.disclaimer{text-align:left!important}.nav-strip{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 30px}.nav-strip a{border:1px solid var(--line);background:#fff;border-radius:99px;padding:9px 13px;text-decoration:none;color:var(--green-dark);font-size:12px;font-weight:750}.nav-strip a:hover{background:var(--mint)}

@media(max-width:800px){.hero{align-items:flex-start;flex-direction:column}.version{min-width:0}.layout{grid-template-columns:1fr}.topbar{padding:0 18px}main{padding:30px 16px 40px}.card-head,.form-body{padding-left:19px;padding-right:19px}footer{padding:0 16px 24px;flex-direction:column}}
@media(max-width:430px){.fields{grid-template-columns:1fr}.status{font-size:11px;padding:8px 10px}.brand{font-size:16px}}
@media(max-width:800px){.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.section-heading{align-items:flex-start;flex-direction:column}}
@media(max-width:430px){.metric-grid{grid-template-columns:1fr 1fr}.metric-card{padding:14px}.metric-card strong{font-size:21px}}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand"><div class="logo">D</div><span>DriftGuard</span></div>
  <div class="status"><span class="dot"></span> Prediction API</div>
</header>
<main>
  <nav class="nav-strip"><a href="#overview">Overview</a><a href="#predictionForm">Churn prediction</a><a href="#monitoring">Drift monitoring</a><a href="#registry">Model registry</a><a href="#retraining">Run history</a></nav>
  <section class="hero">
    <div><div class="eyebrow">MLOps · Customer intelligence</div><h1>Customer churn prediction</h1><p>Estimate the likelihood that a customer may leave. Enter the customer details below to generate a prediction using the currently registered production model.</p></div>
    <div class="version"><span>Serving model</span><strong id="modelVersion">Checking…</strong></div>
  </section>
  <div class="layout">
    <section class="card">
      <div class="card-head"><h2>Customer details</h2><p>Fill in the customer profile and run a prediction.</p></div>
      <form class="form-body" id="predictionForm">
        <div class="section-label">Account overview</div>
        <div class="fields">
          <div class="field"><label for="tenure">Tenure (months)</label><input id="tenure" type="number" min="0" value="12" required></div>
          <div class="field"><label for="monthly">Monthly charges (₹ / units)</label><input id="monthly" type="number" min="0" step="0.01" value="79.99" required></div>
          <div class="field"><label for="total">Total charges</label><input id="total" type="number" min="0" step="0.01" value="959.88" required></div>
          <div class="field"><label for="contract">Contract type</label><select id="contract"><option>Month-to-month</option><option>One year</option><option>Two year</option></select></div>
          <div class="field"><label for="internet">Internet service</label><select id="internet"><option>Fiber optic</option><option>DSL</option><option>No</option></select></div>
          <div class="field"><label for="usage">Average data usage (GB)</label><input id="usage" type="number" min="0" step="0.1" value="5.2" required></div>
        </div>
        <div class="section-label">Customer support & payments</div>
        <div class="fields">
          <div class="field"><label for="late">Late payments</label><input id="late" type="number" min="0" step="1" value="1" required></div>
          <div class="field"><label for="calls">Support calls last month</label><input id="calls" type="number" min="0" step="1" value="2" required></div>
        </div>
        <button class="submit" id="submitBtn" type="submit">Run churn prediction&nbsp; →</button>
        <p class="small-note">The result is generated by the live prediction API. Example values are prefilled and can be edited.</p>
        <div class="error" id="errorBox"></div>
      </form>
    </section>
    <aside class="side">
      <section class="card result-card">
        <div class="result-top"><h2>Prediction result</h2><span class="pill">Live API</span></div>
        <div class="empty" id="emptyState"><div class="empty-icon">✳</div><strong>Your result will appear here</strong><p>Enter the customer details and select “Run churn prediction” to view the model output.</p></div>
        <div class="result" id="resultState">
          <div class="result-label">Churn probability</div><div class="result-value" id="probability">—</div>
          <div class="meter"><div class="meter-fill" id="meterFill"></div></div>
          <div class="meter-meta"><span>Lower likelihood</span><span>Higher likelihood</span></div>
          <div class="result-box"><div class="result-label">Model assessment</div><div class="result-value" id="predictionText" style="font-size:20px;margin:8px 0">—</div><p id="assessmentText"></p></div>
          <div class="small-note" style="text-align:left;margin-top:14px">Model version: <strong id="resultVersion">—</strong></div>
        </div>
      </section>
      <section class="card info-card"><h3>How DriftGuard helps</h3>
        <div class="info-row"><div class="info-icon">↗</div><div><strong>Prediction on demand</strong><span>Send a customer record to the active production model.</span></div></div>
        <div class="info-row"><div class="info-icon">◉</div><div><strong>Version-aware serving</strong><span>Each result identifies the model version used.</span></div></div>
        <div class="info-row"><div class="info-icon">✓</div><div><strong>Controlled model updates</strong><span>Candidate models should pass evaluation before promotion.</span></div></div>
      </section>
    </aside>
  </div>
  <section class="dashboard-intro" id="overview">
    <div class="section-heading"><div><div class="eyebrow">MLOPS CONTROL CENTER</div><h2>Model operations overview</h2><p>Registry and experiment evidence available with this deployment.</p></div><button class="refresh-btn" onclick="loadDashboard()">↻ Refresh data</button></div>
    <div class="metric-grid">
      <article class="metric-card"><div class="metric-icon">◉</div><span>API health</span><strong id="apiMetric">Checking…</strong><small id="apiSub">Live endpoint check</small></article>
      <article class="metric-card"><div class="metric-icon">◆</div><span>Production model</span><strong id="productionMetric">—</strong><small id="modelCount">Registry data</small></article>
      <article class="metric-card"><div class="metric-icon">↻</div><span>Recorded weekly runs</span><strong id="runsMetric">—</strong><small>Saved experiment history</small></article>
      <article class="metric-card"><div class="metric-icon">⚑</div><span>Promotion events</span><strong id="promotionsMetric">—</strong><small>From recorded run history</small></article>
    </div>
  </section>
  <section class="dashboard-section" id="monitoring">
    <div class="section-heading"><div><div class="eyebrow">DRIFT MONITORING</div><h2>Feature drift across evaluation weeks</h2><p>Recorded PSI values from the project reports. This is experiment history, not a live streaming monitor.</p></div><span class="section-tag">Recorded evidence</span></div>
    <div class="card table-card"><div class="table-scroll"><table><thead><tr><th>Week</th><th>Mean PSI</th><th>Maximum PSI</th><th>Features with high drift</th><th>Retraining</th><th>Outcome</th></tr></thead><tbody id="driftRows"><tr><td colspan="6">Loading recorded results…</td></tr></tbody></table></div></div>
    <div class="note-banner"><strong>How to read this:</strong> PSI is a distribution-shift indicator. Drift alone does not prove that model performance has declined.</div>
  </section>
  <section class="dashboard-section" id="registry">
    <div class="section-heading"><div><div class="eyebrow">MODEL LIFECYCLE</div><h2>Model registry</h2><p>Versions and evaluation metrics read from the deployed registry.</p></div><span class="section-tag">Registry-backed</span></div>
    <div class="card table-card"><div class="table-scroll"><table><thead><tr><th>Version</th><th>Status</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 score</th><th>ROC AUC</th></tr></thead><tbody id="modelRows"><tr><td colspan="7">Loading registry…</td></tr></tbody></table></div></div>
  </section>
  <section class="dashboard-section" id="retraining">
    <div class="section-heading"><div><div class="eyebrow">RETRAINING & GOVERNANCE</div><h2>Workflow run history</h2><p>Recorded decisions, candidate promotions, and escalations from the sequential experiment.</p></div><span class="section-tag">Experiment history</span></div>
    <div class="card table-card"><div class="table-scroll"><table><thead><tr><th>Week</th><th>Policy</th><th>Drifted features</th><th>Max PSI</th><th>Decision</th><th>Model change</th><th>Escalation</th></tr></thead><tbody id="runRows"><tr><td colspan="7">Loading run history…</td></tr></tbody></table></div></div>
    <p class="small-note disclaimer">This page displays persisted project artifacts. It does not claim that a continuously scheduled cloud retraining service is running.</p>
  </section>

  <nav class="links"><a href="/health" target="_blank" rel="noreferrer">API health ↗</a><a href="/docs" target="_blank" rel="noreferrer">Technical API docs ↗</a><a href="/redoc" target="_blank" rel="noreferrer">ReDoc ↗</a></nav>
</main>
<footer><span>DriftGuard · Drift-triggered retraining</span><span>Prediction support interface</span></footer>
<script>
const form=document.getElementById('predictionForm');
const errorBox=document.getElementById('errorBox');
async function refreshHealth(){
  try{const r=await fetch('/health');const d=await r.json();document.getElementById('modelVersion').textContent=r.ok?(d.model_version||'Online'):'Unavailable';}
  catch(e){document.getElementById('modelVersion').textContent='API unavailable';}
}
refreshHealth();
form.addEventListener('submit',async(e)=>{
  e.preventDefault();errorBox.style.display='none';
  const btn=document.getElementById('submitBtn');btn.disabled=true;btn.textContent='Generating prediction…';
  const payload={features:{
    Tenure_Months:Number(document.getElementById('tenure').value),
    Monthly_Charges:Number(document.getElementById('monthly').value),
    Total_Charges:Number(document.getElementById('total').value),
    Contract_Type:document.getElementById('contract').value,
    Internet_Service:document.getElementById('internet').value,
    Avg_Data_Usage_GB:Number(document.getElementById('usage').value),
    Late_Payments:Number(document.getElementById('late').value),
    Support_Calls_Last_Month:Number(document.getElementById('calls').value)
  }};
  try{
    const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await response.json();
    if(!response.ok) throw new Error(data.detail||'The prediction request failed.');
    const probability=Number(data.churn_probability);
    const pct=Math.round(probability*100);
    document.getElementById('emptyState').style.display='none';
    document.getElementById('resultState').style.display='block';
    document.getElementById('probability').textContent=pct+'%';
    document.getElementById('meterFill').style.width=pct+'%';
    document.getElementById('predictionText').textContent=Number(data.prediction)===1?'Higher churn risk':'Lower churn risk';
    document.getElementById('assessmentText').textContent=Number(data.prediction)===1?'The model classified this customer as likely to churn.':'The model classified this customer as unlikely to churn.';
    document.getElementById('resultVersion').textContent=data.model_version||'Unknown';
    document.getElementById('modelVersion').textContent=data.model_version||'Online';
  }catch(err){errorBox.textContent='Could not generate prediction: '+err.message+' If the feature names or values do not match the active model, check the API documentation.';errorBox.style.display='block';}
  finally{btn.disabled=false;btn.textContent='Run churn prediction  →';}
});
const fmt=(v,d=3)=>v===null||v===undefined||v===''?'—':(Number.isFinite(Number(v))?Number(v).toFixed(d):String(v));
const chip=(text,kind='')=>`<span class="status-chip ${kind}">${text}</span>`;
async function loadDashboard(){
  try{
    const [healthResp,dataResp]=await Promise.all([fetch('/health'),fetch('/api/dashboard-data')]);
    const health=await healthResp.json(); const d=await dataResp.json();
    document.getElementById('apiMetric').textContent=healthResp.ok?'Online':'Unavailable';
    document.getElementById('apiSub').textContent=healthResp.ok?'Health endpoint responding':'Health check failed';
    document.getElementById('productionMetric').textContent=d.production_version||health.model_version||'—';
    document.getElementById('modelCount').textContent=`${(d.models||[]).length} registered versions`;
    document.getElementById('runsMetric').textContent=(d.runs||[]).length;
    document.getElementById('promotionsMetric').textContent=(d.runs||[]).filter(x=>x.promoted).length;
    document.getElementById('modelRows').innerHTML=(d.models||[]).map(m=>`<tr><td><strong>${m.version||'—'}</strong></td><td>${chip(m.status||'unknown',m.status==='production'?'': 'muted')}</td><td>${fmt(m.metrics?.accuracy*100,1)}${m.metrics?.accuracy==null?'':'%'}</td><td>${fmt(m.metrics?.precision,3)}</td><td>${fmt(m.metrics?.recall,3)}</td><td>${fmt(m.metrics?.f1,3)}</td><td>${fmt(m.metrics?.roc_auc,3)}</td></tr>`).join('')||'<tr><td colspan="7">No registry records were found.</td></tr>';
    document.getElementById('runRows').innerHTML=(d.runs||[]).map(r=>`<tr><td>Week ${r.week??'—'}</td><td>${String(r.policy||'—').replaceAll('_',' ')}</td><td>${r.drifting_features??'—'}</td><td>${fmt(r.max_psi)}</td><td>${chip(String(r.decision||r.status||'not recorded').replaceAll('_',' '),r.escalation?'high':'')}</td><td>${r.promoted?chip(`${r.promoted_from||r.production_before||'—'} → ${r.promoted_to||r.production_after||'—'}`):chip(r.retraining_triggered?'Candidate not promoted':'No model change','muted')}</td><td>${r.escalation?chip('Escalated','high'):chip('No','muted')}</td></tr>`).join('')||'<tr><td colspan="7">No recorded runs found.</td></tr>';
    document.getElementById('driftRows').innerHTML=(d.drift_weeks||[]).map(w=>`<tr><td>Week ${w.week}</td><td>${fmt(w.mean_psi)}</td><td>${fmt(w.max_psi)}</td><td>${w.drifting_features??'—'}</td><td>${w.retraining_triggered?chip('Triggered','high'):chip('Not triggered','muted')}</td><td>${chip(String(w.decision||'not recorded').replaceAll('_',' '),w.escalation?'high':'')}</td></tr>`).join('')||'<tr><td colspan="6">No weekly drift evidence found in the deployment.</td></tr>';
  }catch(e){document.getElementById('apiMetric').textContent='Unavailable';document.getElementById('apiSub').textContent='Could not load dashboard data';}
}
loadDashboard();
</script>
</body>
</html>
""")


@app.get("/api/dashboard-data", include_in_schema=False)
def dashboard_data():
    """Return persisted registry and experiment evidence for the UI; no fabricated metrics."""
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    reports = artifacts / "reports"

    def read_json(path, fallback):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback
        except (OSError, json.JSONDecodeError):
            return fallback

    registry = read_json(artifacts / "registry.json", {"production_version": None, "models": []})
    runs = read_json(reports / "final_eight_week_run_history.json", None)
    if not isinstance(runs, list):
        runs = read_json(artifacts / "run_history.json", [])
    drift_weeks = []
    drift_file = reports / "feature_drift_all_weeks.csv"
    if drift_file.exists():
        try:
            grouped = {}
            with drift_file.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    week = int(float(row.get("week", 0)))
                    if week <= 0:
                        continue
                    item = grouped.setdefault(week, {"week": week, "psi": [], "drifting_features": 0})
                    psi = float(row.get("psi", 0) or 0)
                    item["psi"].append(psi)
                    if str(row.get("status", "")).lower() == "high":
                        item["drifting_features"] += 1
            for week, item in sorted(grouped.items()):
                vals = item.pop("psi")
                item["mean_psi"] = round(sum(vals) / len(vals), 6) if vals else None
                item["max_psi"] = round(max(vals), 6) if vals else None
                run = next((r for r in runs if int(r.get("week", -1)) == week), {})
                item["retraining_triggered"] = bool(run.get("retraining_triggered", False))
                item["decision"] = run.get("decision", "not recorded")
                item["escalation"] = bool(run.get("escalation", False))
                drift_weeks.append(item)
        except (OSError, ValueError, TypeError):
            drift_weeks = []
    models = registry.get("models", [])
    if isinstance(models, dict):
        models = list(models.values())
    return {
        "production_version": registry.get("production_version"),
        "models": models if isinstance(models, list) else [],
        "runs": runs if isinstance(runs, list) else [],
        "drift_weeks": drift_weeks,
        "data_source": "persisted project artifacts",
    }


class Customer(BaseModel):
    features: dict


@app.get("/health")
def health():
    try:
        _, entry = load_production_model()
        return {
            "status": "ok",
            "model_version": entry["version"],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict")
def predict(customer: Customer):
    try:
        model, entry = load_production_model()
        X = pd.DataFrame([customer.features])

        # Reject missing features and preserve the trained feature order.
        expected = entry["features"]
        missing = set(expected) - set(X.columns)

        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")

        X = X[expected]
        probability = float(model.predict_proba(X)[0, 1])
        prediction = int(probability >= 0.5)

        return {
            "prediction": prediction,
            "churn_probability": round(probability, 6),
            "model_version": entry["version"],
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
