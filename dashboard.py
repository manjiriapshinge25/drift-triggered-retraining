import sys, json, subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.core import (DATA, ART, REPORTS, ensure_baseline, load_registry, load_production_model,
                      prepare_xy, metrics_for, drift_report, run_week, RUNS, reset_artifacts)

st.set_page_config(page_title="Drift-Triggered Retraining",page_icon="📈",layout="wide")
st.title("Drift-Triggered Retraining")
st.caption("Local MLOps demo • Churn prediction • Weekly drift • Safe model promotion")
if not (DATA/"churn_reference_train.csv").exists():
    st.error("CSV files missing. Extract the supplied churn CSV files into the project's data/ folder.")
    st.stop()
try: baseline=ensure_baseline()
except Exception as e:
    st.error(f"Baseline initialization failed: {e}")
    st.stop()

reg=load_registry()
history=json.loads(RUNS.read_text()) if RUNS.exists() else []
a,b,c,d=st.columns(4)
a.metric("Production model",reg.get("production_version") or "None")
b.metric("Registered models",len(reg["models"]))
b2=sum(x.get("promoted",False) for x in history)
c.metric("Recorded weekly runs",len(history))
d.metric("Promotions recorded",b2)

tab1,tab2,tab3,tab4,tab5=st.tabs(["Predict","Weekly workflow","Drift analysis","Policy comparison","Registry & logs"])
with tab1:
    st.subheader("Customer churn prediction")
    model,entry=load_production_model()
    ref=pd.read_csv(DATA/"churn_reference_train.csv")
    target=next(c for c in ["Churn","churn","Exited","Exited_Flag","target","Target"] if c in ref.columns)
    Xref=ref.drop(columns=[target]+[c for c in ["Customer_ID","customer_id","CustomerID","customerID","ID","id"] if c in ref.columns])
    sample=Xref.iloc[0].to_dict()
    with st.form("predict_form"):
        inputs={}
        cols=st.columns(3)
        for i,col in enumerate(entry["features"]):
            val=sample.get(col)
            with cols[i%3]:
                if pd.api.types.is_numeric_dtype(Xref[col]):
                    inputs[col]=st.number_input(col,value=float(val) if pd.notna(val) else 0.0)
                else:
                    options=sorted(Xref[col].dropna().astype(str).unique().tolist())
                    sval=str(val) if pd.notna(val) else (options[0] if options else "")
                    inputs[col]=st.selectbox(col,options,index=options.index(sval) if sval in options else 0)
        submitted=st.form_submit_button("Predict churn")
    if submitted:
        pred=int(model.predict(pd.DataFrame([inputs])[entry["features"]])[0])
        prob=float(model.predict_proba(pd.DataFrame([inputs])[entry["features"]])[0,1])
        st.success(f"Prediction: {'Likely to churn' if pred else 'Unlikely to churn'}")
        st.metric("Churn probability",f"{prob:.1%}")
        st.caption(f"Model version: {entry['version']}")

with tab2:
    st.subheader("Run the eight-week workflow")
    policy=st.selectbox("Policy",["drift_low","drift_high","never","weekly"],format_func=lambda x:{"drift_low":"Drift-triggered (lower threshold)","drift_high":"Drift-triggered (higher threshold)","never":"Never retrain","weekly":"Retrain every week"}[x])
    threshold=st.slider("PSI trigger threshold",0.05,0.50,0.10 if policy=="drift_low" else 0.25,0.05)
    fault_week=st.selectbox("Fault injection (optional)",["None","Week 3: invalid data","Week 5: reject candidate"])
    if st.button("Run all 8 weeks",type="primary"):
        with st.spinner("Running sequential weekly workflow..."):
            if st.checkbox("Reset registry before run",value=False):
                reset_artifacts()
            ensure_baseline()
            results=[]
            for w in range(1,9):
                fault=None
                if fault_week.startswith("Week 3") and w==3: fault="missing_column"
                if fault_week.startswith("Week 5") and w==5: fault="candidate_reject"
                results.append(run_week(w,policy,threshold,fault))
            st.session_state["last_results"]=results
        st.success("Workflow finished. Review results below.")
    if "last_results" in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state["last_results"]),use_container_width=True)

with tab3:
    st.subheader("Feature drift across weeks")
    ref=pd.read_csv(DATA/"churn_reference_train.csv")
    rows=[]
    for w in range(1,9):
        p=DATA/f"churn_week_{w}.csv"
        if p.exists():
            try:
                wk=pd.read_csv(p); dr=drift_report(ref,wk); dr["week"]=f"Week {w}"; rows.append(dr)
            except Exception as e: st.warning(f"Week {w}: {e}")
    if rows:
        drift=pd.concat(rows,ignore_index=True)
        st.dataframe(drift,use_container_width=True)
        heat=drift.pivot(index="feature",columns="week",values="psi")
        st.plotly_chart(px.imshow(heat,aspect="auto",color_continuous_scale="RdYlBu_r",title="PSI by feature and week"),use_container_width=True)
        st.plotly_chart(px.bar(drift.groupby("week",as_index=False).agg(high_drift=("psi",lambda x:(x>=0.25).sum())),
                               x="week",y="high_drift",title="Features with PSI ≥ 0.25 per week"),use_container_width=True)
    # Evaluate the baseline on each week for comparison.
    model,entry=load_production_model()
    perf=[]
    for w in range(1,9):
        p=DATA/f"churn_week_{w}.csv"
        if p.exists():
            try:
                wk=pd.read_csv(p); X,y=prepare_xy(wk)
                mm=metrics_for(model,X,y); mm["week"]=f"Week {w}"; perf.append(mm)
            except Exception as e: st.warning(f"Performance Week {w}: {e}")
    if perf:
        pf=pd.DataFrame(perf); st.dataframe(pf,use_container_width=True)
        st.plotly_chart(px.line(pf,x="week",y=["f1","roc_auc"],markers=True,title="Weekly performance (current production model)"),use_container_width=True)

with tab4:
    st.subheader("Retraining policy comparison")
    if st.button("Run all four policies (fresh registry per policy)"):
        all_results=[]
        for pol,thr in [("never",0.10),("weekly",0.10),("drift_low",0.10),("drift_high",0.25)]:
            reset_artifacts(); ensure_baseline()
            for w in range(1,9):
                r=run_week(w,pol,thr)
                r["policy"]=pol
                all_results.append(r)
        st.session_state["policy_results"]=all_results
        # Export raw records; run history will also have current final policy records.
        (REPORTS/"policy_results.json").write_text(json.dumps(all_results,indent=2))
    if "policy_results" in st.session_state:
        rr=pd.DataFrame(st.session_state["policy_results"])
        summary=[]
        for pol,g in rr.groupby("policy"):
            summary.append({"policy":pol,"weeks":len(g),"failed_runs":int((g.status=="failed").sum()),
                            "retraining_attempts":int(g.get("retraining_triggered",pd.Series(False,index=g.index)).fillna(False).sum()),
                            "promotions":int(g.get("promoted",pd.Series(False,index=g.index)).fillna(False).sum()),
                            "mean_weekly_f1":float(g.production_metrics.dropna().map(lambda x:x.get("f1",0)).mean()) if "production_metrics" in g else None,
                            "total_cost_units":int(g.get("retraining_triggered",pd.Series(False,index=g.index)).fillna(False).sum())*10})
        sm=pd.DataFrame(summary)
        st.dataframe(sm,use_container_width=True)
        st.plotly_chart(px.line(rr,x="week",y="production_metrics",color="policy",title="Run records (see F1 table for numeric comparison)"),use_container_width=True)
        st.download_button("Download policy results JSON",json.dumps(all_results,indent=2),"policy_results.json")
    else:
        st.info("Click the button to run all policies. This runs 32 sequential weekly simulations.")

with tab5:
    st.subheader("Registry")
    st.json(load_registry())
    st.subheader("Run history")
    if RUNS.exists():
        rh=pd.DataFrame(json.loads(RUNS.read_text()))
        st.dataframe(rh,use_container_width=True)
        st.download_button("Download run history",RUNS.read_text(),"run_history.json")
    log=ART/"logs"/"workflow.log"
    if log.exists(): st.code(log.read_text()[-12000:])
