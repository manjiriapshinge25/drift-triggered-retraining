
import sys
import json
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



DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DriftGuard | MLOps Dashboard</title>
<style>
:root{--sage:#718b75;--sage-dark:#49644f;--sage-light:#edf3ed;--ink:#202a23;--muted:#778178;--line:#e5ebe5;--paper:#f7f9f6;--white:#fff;--amber:#b7791f;--red:#b45353}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 Inter,"Segoe UI",Arial,sans-serif}
.shell{display:flex;min-height:100vh}.sidebar{width:245px;background:#fff;border-right:1px solid var(--line);padding:28px 18px;display:flex;flex-direction:column;gap:28px;flex-shrink:0}
.brand{display:flex;gap:11px;align-items:center;padding:0 9px}.logo{height:39px;width:39px;border-radius:13px;background:var(--sage-light);display:grid;place-items:center;color:var(--sage-dark);font-size:20px;font-weight:800}.brand strong{display:block;font-size:17px;letter-spacing:-.4px}.brand small{color:var(--muted);font-size:11px}
.nav-label{font-size:10px;text-transform:uppercase;letter-spacing:1.4px;color:#9aa39a;padding:0 12px;margin-bottom:8px}.nav button{display:flex;align-items:center;gap:12px;width:100%;border:0;background:transparent;color:#6c766d;padding:12px;border-radius:10px;text-align:left;cursor:pointer;font:inherit;margin:3px 0}.nav button.active,.nav button:hover{background:var(--sage-light);color:var(--sage-dark);font-weight:650}.nav .ico{width:20px;text-align:center;font-size:16px}
.side-bottom{margin-top:auto;padding:14px;border-radius:13px;background:#f6f8f5;border:1px solid var(--line)}.side-bottom small{color:var(--muted)}.status-dot{display:inline-block;width:8px;height:8px;background:#5e9b70;border-radius:50%;margin-right:7px}
main{padding:34px clamp(18px,3.5vw,48px);flex:1;min-width:0;max-width:1600px;margin:0 auto}.topline{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:26px}.eyebrow{color:var(--sage-dark);text-transform:uppercase;font-size:10px;font-weight:750;letter-spacing:1.7px}.topline h1{font-size:29px;letter-spacing:-1px;margin:5px 0 4px}.sub{color:var(--muted);margin:0}.top-actions{display:flex;align-items:center;gap:10px}.pill{background:#e7f2e8;color:#3f7650;border:1px solid #d3e5d5;border-radius:100px;padding:7px 11px;font-size:12px;font-weight:650;white-space:nowrap}.button{background:var(--sage-dark);color:#fff;border:0;padding:10px 15px;border-radius:9px;cursor:pointer;font:inherit;font-weight:650}.button:hover{background:#3d5743}.button.secondary{background:#fff;color:var(--sage-dark);border:1px solid #cbd8cb}
.grid{display:grid;gap:16px}.kpis{grid-template-columns:repeat(4,minmax(0,1fr));margin:22px 0}.card{background:#fff;border:1px solid var(--line);border-radius:15px;padding:20px;box-shadow:0 3px 12px #283a2805;min-width:0}.kpi-head{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:12px}.kpi-icon{width:33px;height:33px;border-radius:10px;display:grid;place-items:center;background:var(--sage-light);color:var(--sage-dark);font-size:16px}.value{font-size:27px;font-weight:750;letter-spacing:-.8px;margin:13px 0 3px}.hint{font-size:11px;color:var(--muted)}.two{grid-template-columns:minmax(0,1.3fr) minmax(280px,.7fr)}.section-title{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:15px}.section-title h2{font-size:15px;margin:0;letter-spacing:-.2px}.section-title p{color:var(--muted);font-size:11px;margin:4px 0 0}.link{color:var(--sage-dark);text-decoration:none;font-weight:650;font-size:12px}.bars{height:185px;display:flex;align-items:end;gap:12px;padding:15px 8px 0;border-bottom:1px solid var(--line)}.bar-wrap{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:end;height:100%;gap:8px}.bar{width:min(34px,75%);border-radius:7px 7px 2px 2px;background:#b9cbb9;min-height:8px;position:relative}.bar.promoted{background:var(--sage)}.bar-label{font-size:10px;color:var(--muted)}.legend{display:flex;gap:16px;color:var(--muted);font-size:11px;margin-top:14px}.legend i{display:inline-block;width:8px;height:8px;border-radius:3px;margin-right:5px;background:var(--sage)}.legend .other{background:#b9cbb9}
.health-row{display:flex;justify-content:space-between;gap:12px;padding:13px 0;border-bottom:1px solid #eff2ee}.health-row:last-child{border:0}.health-row span{color:var(--muted)}.health-row strong{font-size:12px}.healthy{color:#4d8a5d}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:left;color:var(--muted);font-weight:600;background:#fafbf9}th,td{padding:12px 10px;border-bottom:1px solid #edf0ec;white-space:nowrap}.status{padding:4px 8px;border-radius:99px;font-size:10px;background:#edf5ed;color:#4d8055}.status.failed{background:#fff1ed;color:#ad5748}.page{display:none}.page.active{display:block}.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.field label{display:block;font-size:12px;font-weight:650;margin-bottom:6px}.field input,.field select{width:100%;padding:11px 12px;border:1px solid #dbe3da;border-radius:9px;background:#fff;color:var(--ink);font:inherit}.field input:focus,.field select:focus{outline:2px solid #d5e5d6;border-color:var(--sage)}.result{margin-top:18px;padding:18px;border-radius:12px;background:var(--sage-light);display:none}.result strong{font-size:18px}.result .prob{font-size:30px;font-weight:800;color:var(--sage-dark)}.error{color:var(--red);white-space:pre-wrap}.footnote{font-size:11px;color:var(--muted);margin-top:14px}
@media(max-width:950px){.sidebar{width:200px}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.two{grid-template-columns:1fr}}
@media(max-width:650px){.shell{display:block}.sidebar{width:100%;padding:15px;gap:14px;border-right:0;border-bottom:1px solid var(--line)}.brand{padding:0}.nav{display:flex;overflow-x:auto}.nav-label,.side-bottom{display:none}.nav button{white-space:nowrap;width:auto;padding:9px}.topline{align-items:flex-start;flex-direction:column}.topline h1{font-size:24px}.kpis{gap:10px}.card{padding:15px}.value{font-size:23px}.form-grid{grid-template-columns:1fr}main{padding:22px 14px}}
</style>
</head>
<body>
<div class="shell">
<aside class="sidebar">
  <div class="brand"><div class="logo">D</div><div><strong>DriftGuard</strong><small>MLOps control center</small></div></div>
  <div><div class="nav-label">Workspace</div><nav class="nav">
    <button class="active" data-page="overview"><span class="ico">◫</span>Overview</button>
    <button data-page="predict"><span class="ico">✳</span>Prediction playground</button>
    <button data-page="monitoring"><span class="ico">⌁</span>Model monitoring</button>
    <button data-page="registry"><span class="ico">▤</span>Model registry</button>
  </nav></div>
  <div class="side-bottom"><div><span class="status-dot"></span><strong>API connection</strong></div><small id="side-health">Checking service…</small></div>
</aside>
<main>
  <div class="topline"><div><div class="eyebrow">Machine learning operations</div><h1 id="page-title">Overview</h1><p class="sub" id="page-sub">A clear view of your production model and retraining activity.</p></div><div class="top-actions"><span class="pill" id="live-pill">● Checking status</span><button class="button secondary" onclick="refreshData()">↻ Refresh</button></div></div>

  <section class="page active" id="page-overview">
    <div class="grid kpis">
      <div class="card"><div class="kpi-head">Production model <span class="kpi-icon">◈</span></div><div class="value" id="kpi-version">—</div><div class="hint">Currently registered production version</div></div>
      <div class="card"><div class="kpi-head">Registered models <span class="kpi-icon">▤</span></div><div class="value" id="kpi-models">—</div><div class="hint">Versions recorded in the registry</div></div>
      <div class="card"><div class="kpi-head">Weekly run records <span class="kpi-icon">⌁</span></div><div class="value" id="kpi-runs">—</div><div class="hint">Available experiment history</div></div>
      <div class="card"><div class="kpi-head">Promotions recorded <span class="kpi-icon">↗</span></div><div class="value" id="kpi-promotions">—</div><div class="hint">Recorded successful model promotions</div></div>
    </div>
    <div class="grid two">
      <div class="card"><div class="section-title"><div><h2>Weekly workflow activity</h2><p>Promotion records from saved experiment history</p></div><a class="link" href="#" onclick="showPage('monitoring');return false">View monitoring →</a></div><div id="weekly-bars" class="bars"><span class="hint">Loading recorded runs…</span></div><div class="legend"><span><i></i>Promotion recorded</span><span><i class="other"></i>Other run record</span></div></div>
      <div class="card"><div class="section-title"><div><h2>Service health</h2><p>Live API check</p></div><span class="pill">Live</span></div>
        <div class="health-row"><span>FastAPI service</span><strong id="api-status">Checking…</strong></div>
        <div class="health-row"><span>Active model version</span><strong id="health-version">—</strong></div>
        <div class="health-row"><span>Prediction route</span><strong class="healthy">/predict</strong></div>
        <div class="health-row"><span>API documentation</span><strong><a class="link" href="/docs">Open /docs ↗</a></strong></div>
      </div>
    </div>
    <div class="card" style="margin-top:16px"><div class="section-title"><div><h2>Production model metrics</h2><p>Metrics stored in the model registry for the active production version</p></div></div><div class="grid kpis" style="margin:0" id="metrics-cards"></div><p class="footnote">Metrics shown here are recorded registry values, not a real-time evaluation on newly arriving data.</p></div>
  </section>

  <section class="page" id="page-predict">
    <div class="card"><div class="section-title"><div><h2>Customer churn prediction</h2><p>Submit customer features to the live production API.</p></div><span class="pill">POST /predict</span></div>
      <form id="predict-form"><div class="form-grid">
        <div class="field"><label for="Tenure_Months">Tenure (months)</label><input id="Tenure_Months" name="Tenure_Months" type="number" min="0" value="12" required></div>
        <div class="field"><label for="Contract_Type">Contract type</label><select id="Contract_Type" name="Contract_Type"><option>Month-to-month</option><option>One year</option><option>Two year</option></select></div>
        <div class="field"><label for="Internet_Service">Internet service</label><select id="Internet_Service" name="Internet_Service"><option>Fiber optic</option><option>DSL</option><option>No</option></select></div>
        <div class="field"><label for="Monthly_Charges">Monthly charges</label><input id="Monthly_Charges" name="Monthly_Charges" type="number" min="0" step=".01" value="79.99" required></div>
        <div class="field"><label for="Total_Charges">Total charges</label><input id="Total_Charges" name="Total_Charges" type="number" min="0" step=".01" value="959.88" required></div>
        <div class="field"><label for="Support_Calls_Last_Month">Support calls last month</label><input id="Support_Calls_Last_Month" name="Support_Calls_Last_Month" type="number" min="0" value="2" required></div>
        <div class="field"><label for="Avg_Data_Usage_GB">Average data usage (GB)</label><input id="Avg_Data_Usage_GB" name="Avg_Data_Usage_GB" type="number" min="0" step=".1" value="5.2" required></div>
        <div class="field"><label for="Late_Payments">Late payments</label><input id="Late_Payments" name="Late_Payments" type="number" min="0" value="1" required></div>
      </div><div style="display:flex;gap:10px;align-items:center;margin-top:20px"><button class="button" type="submit" id="predict-btn">Run prediction →</button><button class="button secondary" type="reset">Reset form</button></div></form>
      <div class="result" id="prediction-result"><div id="prediction-label"></div><div class="prob" id="prediction-prob"></div><div id="prediction-model" class="hint"></div></div><p class="footnote">This prediction is generated by the currently deployed production model. It is a demonstration, not a decision about a real customer.</p>
    </div>
  </section>

  <section class="page" id="page-monitoring">
    <div class="grid two">
      <div class="card"><div class="section-title"><div><h2>Weekly workflow records</h2><p>Saved run records, not continuous real-time monitoring</p></div></div><div class="table-wrap"><table><thead><tr><th>Week</th><th>Policy</th><th>Status</th><th>Promoted</th></tr></thead><tbody id="runs-table"><tr><td colspan="4">Loading…</td></tr></tbody></table></div></div>
      <div class="card"><div class="section-title"><div><h2>Production performance</h2><p>Stored model evaluation metrics</p></div></div><div id="monitor-metrics"></div><p class="footnote">Drift and workflow records are loaded from the repository artifacts bundled with this service. They update only when those records are updated and redeployed.</p></div>
    </div>
  </section>

  <section class="page" id="page-registry">
    <div class="card"><div class="section-title"><div><h2>Model registry</h2><p>Production version and registered model inventory</p></div></div><div class="grid kpis" style="margin:0 0 18px"><div><div class="hint">Production version</div><div class="value" id="reg-version">—</div></div><div><div class="hint">Registered versions</div><div class="value" id="reg-count">—</div></div></div><div class="table-wrap"><table><thead><tr><th>Version</th><th>Role</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody id="registry-table"><tr><td colspan="4">Loading…</td></tr></tbody></table></div><p class="footnote">Registry information is read from the deployed artifacts/registry.json file.</p></div>
  </section>
</main>
</div>
<script>
const $=id=>document.getElementById(id);
const titles={overview:["Overview","A clear view of your production model and retraining activity."],predict:["Prediction playground","Test a customer record against the deployed production API."],monitoring:["Model monitoring","Review saved weekly workflow records and model performance."],registry:["Model registry","Inspect the registered versions and the active production model."]};
function showPage(name){document.querySelectorAll(".page").forEach(p=>p.classList.remove("active"));$("page-"+name).classList.add("active");document.querySelectorAll(".nav button").forEach(b=>b.classList.toggle("active",b.dataset.page===name));$("page-title").textContent=titles[name][0];$("page-sub").textContent=titles[name][1];}
document.querySelectorAll(".nav button").forEach(b=>b.addEventListener("click",()=>showPage(b.dataset.page)));
let data=null;
async function refreshData(){
  try{
    const h=await fetch("/health",{cache:"no-store"});
    const health=await h.json();
    $("api-status").textContent=h.ok?"Healthy":"Unavailable";$("api-status").className=h.ok?"healthy":"error";
    $("health-version").textContent=health.model_version||"—";$("side-health").textContent=h.ok?"Connected · model "+(health.model_version||"active"):"API returned an error";
    $("live-pill").textContent=h.ok?"● API connected":"● API issue";$("live-pill").style.background=h.ok?"#e7f2e8":"#fff0ed";$("live-pill").style.color=h.ok?"#3f7650":"#ad5748";
    const r=await fetch("/dashboard-data",{cache:"no-store"});data=await r.json();
    $("kpi-version").textContent=data.production_version;$("kpi-models").textContent=data.registered_models;$("kpi-runs").textContent=data.weekly_runs;$("kpi-promotions").textContent=data.promotions;
    $("reg-version").textContent=data.production_version;$("reg-count").textContent=data.registered_models;
    const metrics=data.metrics||{};const keys=[["f1","F1 score"],["roc_auc","ROC-AUC"],["accuracy","Accuracy"],["precision","Precision"],["recall","Recall"]];
    $("metrics-cards").innerHTML=keys.map(([k,label])=>`<div class="card" style="padding:14px;background:#fafbf9"><div class="hint">${label}</div><div class="value" style="font-size:22px">${metrics[k]==null?"—":(Number(metrics[k])*100).toFixed(1)+"%"}</div></div>`).join("");
    $("monitor-metrics").innerHTML=keys.map(([k,label])=>`<div class="health-row"><span>${label}</span><strong>${metrics[k]==null?"—":(Number(metrics[k])*100).toFixed(2)+"%"}</strong></div>`).join("");
    const runs=data.runs||[];const max=Math.max(1,...runs.map(x=>Number(x.week)||0));
    $("weekly-bars").innerHTML=runs.length?runs.map((x,i)=>`<div class="bar-wrap"><div class="bar ${x.promoted?"promoted":""}" style="height:${Math.max(10,((Number(x.week)||i+1)/max)*125)}px" title="${x.promoted?"Promotion recorded":"No promotion recorded"}"></div><div class="bar-label">W${x.week||i+1}</div></div>`).join(""):`<span class="hint">No saved weekly records found.</span>`;
    $("runs-table").innerHTML=runs.length?runs.map((x,i)=>`<tr><td>Week ${x.week||i+1}</td><td>${x.policy||"—"}</td><td><span class="status">${x.status||"recorded"}</span></td><td>${x.promoted?"Yes":"No"}</td></tr>`).join(""):`<tr><td colspan="4">No saved run records.</td></tr>`;
    const models=Array.isArray(data.models)?data.models:[];
    $("registry-table").innerHTML=models.length?models.map(m=>`<tr><td>${m.version}</td><td>${m.version===data.production_version?'<span class="status">Production</span>':'Registered'}</td><td>${m.metrics?.f1==null?"—":(m.metrics.f1*100).toFixed(1)+"%"}</td><td>${m.metrics?.roc_auc==null?"—":(m.metrics.roc_auc*100).toFixed(1)+"%"}</td></tr>`).join(""):`<tr><td colspan="4">Registry model list is not available from the dashboard summary.</td></tr>`;
  }catch(e){$("side-health").textContent="Could not connect";$("api-status").textContent="Unavailable";$("live-pill").textContent="● Connection issue";}
}
$("predict-form").addEventListener("submit",async e=>{
  e.preventDefault();const btn=$("predict-btn");btn.disabled=true;btn.textContent="Predicting…";$("prediction-result").style.display="none";
  const f=new FormData(e.target),features={};
  for(const [k,v] of f.entries()){features[k]=["Contract_Type","Internet_Service"].includes(k)?v:Number(v);}
  try{
    const res=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({features})});
    const out=await res.json();if(!res.ok)throw new Error(out.detail||"Prediction request failed");
    $("prediction-label").textContent=out.prediction===1?"Higher churn risk":"Lower churn risk";
    $("prediction-prob").textContent=(Number(out.churn_probability)*100).toFixed(1)+"%";
    $("prediction-model").textContent="Churn probability · Model "+out.model_version;
    $("prediction-result").style.display="block";$("prediction-result").style.background=out.prediction===1?"#fff5e9":"#edf5ed";
  }catch(err){$("prediction-result").style.display="block";$("prediction-result").className="result error";$("prediction-label").textContent="Prediction failed";$("prediction-prob").textContent="";$("prediction-model").textContent=err.message;}
  finally{btn.disabled=false;btn.textContent="Run prediction →";}
});
refreshData();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    """Serve the DriftGuard dashboard from the same FastAPI service."""
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/dashboard-data", include_in_schema=False)
def dashboard_data():
    """Return registry and recorded workflow summaries for dashboard widgets."""
    root = Path(__file__).resolve().parents[1]
    registry_path = root / "artifacts" / "registry.json"
    runs_path = root / "artifacts" / "run_history.json"
    registry = {}
    runs = []
    try:
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if runs_path.exists():
            runs = json.loads(runs_path.read_text(encoding="utf-8"))
    except Exception:
        # Keep the live prediction API available even if optional dashboard
        # summaries cannot be read.
        pass
    models = registry.get("models", [])
    production_version = registry.get("production_version")
    production = next((m for m in models if m.get("version") == production_version), {})
    metrics = production.get("metrics", {})
    return {
        "production_version": production_version or "Unavailable",
        "registered_models": len(models),
        "weekly_runs": len(runs),
        "promotions": sum(bool(r.get("promoted")) for r in runs),
        "metrics": metrics,
        "models": models,
        "runs": [
            {
                "week": r.get("week", r.get("week_number", i + 1)),
                "status": r.get("status", "recorded"),
                "promoted": bool(r.get("promoted")),
                "policy": r.get("policy", ""),
            }
            for i, r in enumerate(runs[-8:])
        ],
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
