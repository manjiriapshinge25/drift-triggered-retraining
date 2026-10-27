import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core import (
    ART, DATA, REPORTS, RUNS, ensure_baseline, load_registry,
    load_production_model, prepare_xy, metrics_for, drift_report,
    run_week, reset_artifacts,
)

# ---------- Page setup and visual system ----------
st.set_page_config(
    page_title="DriftGuard | MLOps Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAGE = "#66856E"
DARK = "#23352B"
MUTED = "#718078"
PALE = "#EEF4EF"
BORDER = "#E2E9E3"

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; color: {DARK}; }}
      .stApp {{ background: #F7F9F7; }}
      [data-testid="stHeader"] {{ background: rgba(247,249,247,.92); }}
      [data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {BORDER}; }}
      [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.25rem; }}
      .block-container {{ padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1500px; }}
      h1, h2, h3 {{ font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; color: {DARK}; }}
      h1 {{ font-size: 2rem !important; font-weight: 800 !important; }}
      h2 {{ font-size: 1.35rem !important; font-weight: 750 !important; }}
      h3 {{ font-size: 1.02rem !important; font-weight: 700 !important; }}
      [data-testid="stMetric"] {{ background: white; border: 1px solid {BORDER}; padding: 18px 20px; border-radius: 16px; box-shadow: 0 3px 12px rgba(35,53,43,.025); }}
      [data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: .83rem; }}
      [data-testid="stMetricValue"] {{ color: {DARK}; font-family: 'Manrope',sans-serif; font-weight: 800; }}
      [data-testid="stMetricDelta"] {{ font-size: .78rem; }}
      div.stButton > button, div.stDownloadButton > button, [data-testid="stFormSubmitButton"] button {{
        background: {SAGE}; color: white; border: 1px solid {SAGE}; border-radius: 10px;
        padding: .55rem 1rem; font-weight: 700; transition: all .15s ease;
      }}
      div.stButton > button:hover, div.stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{
        background: #526E59; color: white; border-color: #526E59; transform: translateY(-1px);
      }}
      [data-testid="stTabs"] button {{ font-weight: 600; }}
      [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}
      div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{ border-radius: 10px; border-color: {BORDER}; }}
      .brand {{ display:flex; align-items:center; gap:11px; padding: 4px 3px 24px; }}
      .brand-mark {{ width:42px;height:42px;border-radius:13px;background:{PALE};display:flex;align-items:center;justify-content:center;font-size:22px; }}
      .brand-name {{ font-family:'Manrope',sans-serif;font-size:20px;font-weight:800;color:{DARK};line-height:1.1; }}
      .brand-sub {{ color:{MUTED};font-size:11px;letter-spacing:.08em;text-transform:uppercase;margin-top:4px; }}
      .eyebrow {{ color:{SAGE};font-size:11px;letter-spacing:.13em;text-transform:uppercase;font-weight:800;margin-bottom:7px; }}
      .hero {{ background:linear-gradient(115deg,#FFFFFF 0%,#F0F5F0 100%);border:1px solid {BORDER};border-radius:20px;padding:25px 28px;margin-bottom:22px; }}
      .hero-title {{ font-family:'Manrope',sans-serif;font-size:27px;font-weight:800;letter-spacing:-.045em;color:{DARK};line-height:1.2; }}
      .hero-copy {{ color:{MUTED};font-size:14px;line-height:1.6;margin-top:9px;max-width:720px; }}
      .status-pill {{ display:inline-block;padding:6px 10px;border-radius:999px;background:#E5F2E8;color:#41694A;font-size:11px;font-weight:800;letter-spacing:.04em; }}
      .section-head {{ display:flex;justify-content:space-between;align-items:center;margin:26px 0 12px; }}
      .section-note {{ color:{MUTED};font-size:12px; }}
      .soft-card {{ background:white;border:1px solid {BORDER};border-radius:16px;padding:18px 20px;height:100%; }}
      .soft-label {{ color:{MUTED};font-size:12px;font-weight:600;margin-bottom:9px; }}
      .soft-value {{ color:{DARK};font-family:'Manrope',sans-serif;font-size:25px;font-weight:800; }}
      .soft-foot {{ color:{MUTED};font-size:12px;margin-top:5px; }}
      .small-label {{ color:{MUTED};font-size:12px; }}
      footer {{ visibility:hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="brand-mark">🌿</div><div><div class="brand-name">DriftGuard</div><div class="brand-sub">Model operations</div></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("CHURN PREDICTION PLATFORM")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Prediction playground", "Model monitoring", "Retraining workflow", "Policy comparison", "Model registry & logs"],
        label_visibility="collapsed",
        key="navigation",
    )
    st.markdown("---")
    st.markdown("**Environment**")
    st.markdown(f'<span class="status-pill">● LOCAL DEMO</span>', unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.caption("Drift-triggered retraining · Safe model promotion")

if not (DATA / "churn_reference_train.csv").exists():
    st.error("Dataset files are missing. Extract the supplied churn CSV files into the project's data/ folder.")
    st.stop()

try:
    ensure_baseline()
    registry = load_registry()
except Exception as exc:
    st.error(f"Could not initialize the baseline model: {exc}")
    st.stop()

history = json.loads(RUNS.read_text()) if RUNS.exists() else []
production_version = registry.get("production_version") or "Not set"
model_count = len(registry.get("models", {}))
promotions = sum(bool(row.get("promoted", False)) for row in history)

# ---------- Shared helpers ----------
def page_header(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div style="color:{MUTED};font-size:14px;margin-top:-7px;margin-bottom:22px">{subtitle}</div>', unsafe_allow_html=True)


def card(label, value, foot=""):
    st.markdown(
        f'<div class="soft-card"><div class="soft-label">{label}</div><div class="soft-value">{value}</div><div class="soft-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def load_weekly_drift():
    ref = pd.read_csv(DATA / "churn_reference_train.csv")
    rows = []
    for week in range(1, 9):
        path = DATA / f"churn_week_{week}.csv"
        if path.exists():
            try:
                result = drift_report(ref, pd.read_csv(path))
                result["week"] = f"Week {week}"
                rows.append(result)
            except Exception:
                continue
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_weekly_performance():
    model, _ = load_production_model()
    rows = []
    for week in range(1, 9):
        path = DATA / f"churn_week_{week}.csv"
        if path.exists():
            try:
                X, y = prepare_xy(pd.read_csv(path))
                metrics = metrics_for(model, X, y)
                metrics["week"] = f"Week {week}"
                rows.append(metrics)
            except Exception:
                continue
    return pd.DataFrame(rows)

# ---------- Overview ----------
if page == "Overview":
    st.markdown(
        f'''<div class="hero"><div class="eyebrow">YOUR MLOPS WORKSPACE</div>
        <div class="hero-title">Model health, at a glance.</div>
        <div class="hero-copy">Monitor customer churn predictions, explore feature drift, and run a controlled retraining workflow — all from one workspace.</div>
        <div style="margin-top:17px"><span class="status-pill">● Registry connected</span></div></div>''',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Production model", production_version, help="Currently selected model from the registry")
    with c2: st.metric("Registered models", model_count)
    with c3: st.metric("Weekly runs recorded", len(history))
    with c4: st.metric("Successful promotions", promotions)

    st.markdown('<div class="section-head"><h3>Workspace</h3><span class="section-note">Choose a workflow to continue</span></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="soft-card"><div style="font-size:22px">◉</div><h3>Prediction playground</h3><div class="small-label">Test a customer record and see the churn score from the active model.</div></div>', unsafe_allow_html=True)
        st.button("Open prediction →", key="go_predict", use_container_width=True,
                  on_click=lambda: st.session_state.update({"navigation": "Prediction playground"}))
    with c2:
        st.markdown('<div class="soft-card"><div style="font-size:22px">⌁</div><h3>Model monitoring</h3><div class="small-label">Review PSI drift signals and weekly model performance.</div></div>', unsafe_allow_html=True)
        st.button("Open monitoring →", key="go_monitor", use_container_width=True,
                  on_click=lambda: st.session_state.update({"navigation": "Model monitoring"}))
    with c3:
        st.markdown('<div class="soft-card"><div style="font-size:22px">↻</div><h3>Retraining workflow</h3><div class="small-label">Run the eight-week simulation with configurable retraining policy.</div></div>', unsafe_allow_html=True)
        st.button("Open workflow →", key="go_workflow", use_container_width=True,
                  on_click=lambda: st.session_state.update({"navigation": "Retraining workflow"}))

    drift = load_weekly_drift()
    if not drift.empty:
        st.markdown('<div class="section-head"><h3>Drift snapshot</h3><span class="section-note">Preloaded eight-week dataset</span></div>', unsafe_allow_html=True)
        d1, d2 = st.columns([1.35, 1])
        with d1:
            heat = drift.pivot_table(index="feature", columns="week", values="psi", aggfunc="mean")
            st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale=[[0,"#E8F0E9"],[0.5,"#AFC5B2"],[1,"#66856E"]], title="Feature drift · PSI by week", labels={"color":"PSI"}), use_container_width=True)
        with d2:
            summary = drift.groupby("week", as_index=False).agg(drifted_features=("psi", lambda values: int((values >= 0.25).sum())))
            st.plotly_chart(px.bar(summary, x="week", y="drifted_features", title="Features at high drift (PSI ≥ 0.25)", color_discrete_sequence=[SAGE]), use_container_width=True)

# ---------- Prediction ----------
elif page == "Prediction playground":
    page_header("INFERENCE", "Prediction playground", "Enter customer details to score churn risk using the current production model.")
    model, entry = load_production_model()
    ref = pd.read_csv(DATA / "churn_reference_train.csv")
    target = next((col for col in ["Churn", "churn", "Exited", "Exited_Flag", "target", "Target"] if col in ref.columns), None)
    if target is None:
        st.error("Could not identify the target column in the reference dataset.")
        st.stop()
    drop_cols = [target] + [col for col in ["Customer_ID", "customer_id", "CustomerID", "customerID", "ID", "id"] if col in ref.columns]
    Xref = ref.drop(columns=drop_cols)
    sample = Xref.iloc[0].to_dict()
    st.markdown(f'<div class="soft-card"><div class="soft-label">ACTIVE PRODUCTION MODEL</div><div class="soft-value">{entry["version"]}</div><div class="soft-foot">Features are prefilled from a sample record. Adjust values before running a prediction.</div></div>', unsafe_allow_html=True)
    st.markdown("### Customer features")
    with st.form("predict_form", border=True):
        inputs = {}
        cols = st.columns(3)
        for i, col in enumerate(entry["features"]):
            value = sample.get(col)
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(Xref[col]):
                    inputs[col] = st.number_input(col.replace("_", " "), value=float(value) if pd.notna(value) else 0.0, help=f"Model input: {col}")
                else:
                    options = sorted(Xref[col].dropna().astype(str).unique().tolist())
                    selected = str(value) if pd.notna(value) else (options[0] if options else "")
                    inputs[col] = st.selectbox(col.replace("_", " "), options, index=options.index(selected) if selected in options else 0)
        submitted = st.form_submit_button("Run churn prediction", type="primary", use_container_width=True)
    if submitted:
        frame = pd.DataFrame([inputs])[entry["features"]]
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0, 1])
        left, right = st.columns([1, 1])
        with left:
            if prediction:
                st.error("Higher churn risk · model predicts churn")
            else:
                st.success("Lower churn risk · model predicts no churn")
            st.caption(f"Prediction generated by model {entry['version']}.")
        with right:
            st.metric("Churn probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0), text="Estimated probability")
        st.dataframe(frame, use_container_width=True, hide_index=True)

# ---------- Monitoring ----------
elif page == "Model monitoring":
    page_header("OBSERVABILITY", "Model monitoring", "Inspect feature-level drift and weekly classification performance.")
    drift = load_weekly_drift()
    if drift.empty:
        st.info("No drift results were available.")
    else:
        high = int((drift["psi"] >= 0.25).sum())
        max_psi = float(drift["psi"].max())
        a, b, c = st.columns(3)
        with a: st.metric("Feature-week checks", len(drift))
        with b: st.metric("High-drift signals", high)
        with c: st.metric("Maximum PSI", f"{max_psi:.3f}")
        heat = drift.pivot_table(index="feature", columns="week", values="psi", aggfunc="mean")
        st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale=[[0,"#E8F0E9"],[0.5,"#AFC5B2"],[1,"#66856E"]], title="Feature drift heatmap · Population Stability Index", labels={"color":"PSI"}), use_container_width=True)
        st.dataframe(drift.sort_values("psi", ascending=False), use_container_width=True, hide_index=True)
    perf = load_weekly_performance()
    if not perf.empty:
        st.markdown("### Weekly model performance")
        metrics = [m for m in ["f1", "roc_auc", "accuracy", "precision", "recall"] if m in perf.columns]
        if metrics:
            st.plotly_chart(px.line(perf, x="week", y=metrics, markers=True, title="Performance by week", color_discrete_sequence=[SAGE, "#A6BCA9", "#B59D78", "#7D9AA0", "#9A8CB8"]), use_container_width=True)
        st.dataframe(perf, use_container_width=True, hide_index=True)

# ---------- Workflow ----------
elif page == "Retraining workflow":
    page_header("ORCHESTRATION", "Retraining workflow", "Run the eight-week simulation sequentially and inspect model promotion decisions.")
    st.info("This runs the project's local simulation. It does not start a continuously running cloud scheduler.")
    with st.container(border=True):
        policy = st.selectbox("Retraining policy", ["drift_low", "drift_high", "never", "weekly"], format_func=lambda value: {"drift_low":"Drift-triggered · lower threshold", "drift_high":"Drift-triggered · higher threshold", "never":"Never retrain", "weekly":"Retrain every week"}[value])
        threshold = st.slider("PSI trigger threshold", 0.05, 0.50, 0.10 if policy == "drift_low" else 0.25, 0.05)
        fault = st.selectbox("Fault injection", ["None", "Week 3 · invalid data", "Week 5 · reject candidate"])
        reset = st.checkbox("Reset registry before running", value=False)
        run = st.button("Run all 8 weeks", type="primary", use_container_width=True)
    if run:
        if reset:
            reset_artifacts()
        ensure_baseline()
        results = []
        progress = st.progress(0, text="Starting weekly workflow…")
        for week in range(1, 9):
            scenario = None
            if fault.startswith("Week 3") and week == 3:
                scenario = "missing_column"
            if fault.startswith("Week 5") and week == 5:
                scenario = "candidate_reject"
            results.append(run_week(week, policy, threshold, scenario))
            progress.progress(week / 8, text=f"Completed week {week} of 8")
        st.session_state["last_results"] = results
        st.success("Workflow simulation completed.")
    if "last_results" in st.session_state:
        results = pd.DataFrame(st.session_state["last_results"])
        st.markdown("### Run results")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("Download workflow results", json.dumps(st.session_state["last_results"], indent=2, default=str), "driftguard_workflow_results.json")

# ---------- Policy comparison ----------
elif page == "Policy comparison":
    page_header("EXPERIMENTS", "Policy comparison", "Compare retraining frequency, promotion outcomes, performance, and modeled cost.")
    st.caption("Cost is reported in the project's abstract cost units, not actual cloud billing.")
    if st.button("Run all four policies", type="primary", use_container_width=True):
        all_results = []
        progress = st.progress(0, text="Preparing policy simulations…")
        policies = [("never", 0.10), ("weekly", 0.10), ("drift_low", 0.10), ("drift_high", 0.25)]
        for index, (policy, threshold) in enumerate(policies):
            reset_artifacts()
            ensure_baseline()
            for week in range(1, 9):
                result = run_week(week, policy, threshold)
                result["policy"] = policy
                all_results.append(result)
            progress.progress((index + 1) / len(policies), text=f"Completed {policy} policy")
        st.session_state["policy_results"] = all_results
        (REPORTS / "policy_results.json").write_text(json.dumps(all_results, indent=2, default=str))
        st.success("All four policy simulations completed.")
    if "policy_results" in st.session_state:
        records = pd.DataFrame(st.session_state["policy_results"])
        summaries = []
        for policy, group in records.groupby("policy"):
            attempts = int(group.get("retraining_triggered", pd.Series(False, index=group.index)).fillna(False).sum())
            promoted = int(group.get("promoted", pd.Series(False, index=group.index)).fillna(False).sum())
            scores = []
            if "production_metrics" in group:
                for item in group["production_metrics"].dropna():
                    if isinstance(item, dict) and item.get("f1") is not None:
                        scores.append(float(item["f1"]))
            summaries.append({"Policy": policy, "Weeks": len(group), "Failed runs": int((group.get("status", pd.Series(index=group.index, dtype=str)) == "failed").sum()), "Retraining attempts": attempts, "Promotions": promoted, "Mean weekly F1": sum(scores) / len(scores) if scores else None, "Modeled cost units": attempts * 10 + promoted * 2})
        summary = pd.DataFrame(summaries)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        if "Mean weekly F1" in summary and summary["Mean weekly F1"].notna().any():
            st.plotly_chart(px.bar(summary, x="Policy", y="Mean weekly F1", color_discrete_sequence=[SAGE], title="Mean weekly F1 by policy"), use_container_width=True)
        st.download_button("Download policy results", json.dumps(st.session_state["policy_results"], indent=2, default=str), "driftguard_policy_results.json")
    else:
        st.markdown('<div class="soft-card"><h3>Policy study ready</h3><div class="small-label">Run the four-policy comparison to generate a fresh side-by-side evaluation across all eight weeks.</div></div>', unsafe_allow_html=True)

# ---------- Registry and logs ----------
elif page == "Model registry & logs":
    page_header("MODEL LIFECYCLE", "Model registry & logs", "Review registered model versions, run history, and workflow diagnostics.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Registry")
        st.json(load_registry(), expanded=2)
    with c2:
        st.markdown("### Run history")
        if RUNS.exists():
            run_text = RUNS.read_text()
            try:
                run_df = pd.DataFrame(json.loads(run_text))
                st.dataframe(run_df, use_container_width=True, hide_index=True)
            except Exception:
                st.code(run_text[-10000:])
            st.download_button("Download run history", run_text, "run_history.json")
        else:
            st.info("No run history has been recorded yet.")
    log = ART / "logs" / "workflow.log"
    if log.exists():
        st.markdown("### Workflow log")
        st.code(log.read_text()[-12000:], language="text")
