from __future__ import annotations
import hashlib, json, os, time, uuid, logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
MODELS = ART / "models"
LOGS = ART / "logs"
REPORTS = ART / "reports"
for p in (ART, MODELS, LOGS, REPORTS): p.mkdir(parents=True, exist_ok=True)
REGISTRY = ART / "registry.json"
RUNS = ART / "run_history.json"
RUNLOG = LOGS / "workflow.log"
logger = logging.getLogger("drift_project")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    h = logging.FileHandler(RUNLOG, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def load_registry():
    if not REGISTRY.exists():
        return {"production_version": None, "models": []}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))

def save_registry(reg):
    tmp=REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg,indent=2),encoding="utf-8")
    tmp.replace(REGISTRY)

def save_run(run):
    records=json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else []
    records.append(run)
    RUNS.write_text(json.dumps(records,indent=2),encoding="utf-8")

def find_target(df):
    candidates=["Churn","churn","Exited","Exited_Flag","target","Target"]
    for c in candidates:
        if c in df.columns: return c
    raise ValueError(f"Target column not found. Columns: {list(df.columns)}")

def prepare_xy(df):
    target=find_target(df)
    drop=[target, "Week"]
    for c in ["Customer_ID","customer_id","CustomerID","customerID","ID","id"]:
        if c in df.columns: drop.append(c)
    X=df.drop(columns=drop, errors="ignore").copy()
    y=df[target].copy()
    # Normalize common binary string targets.
    if y.dtype == "object":
        mapped=y.astype(str).str.strip().str.lower().map({
            "yes":1,"no":0,"true":1,"false":0,"churn":1,"not churn":0,
            "1":1,"0":0,"1.0":1,"0.0":0
        })
        if mapped.notna().all(): y=mapped.astype(int)
    if y.nunique() != 2:
        raise ValueError(f"Expected binary target, found {y.nunique()} classes.")
    if not set(pd.Series(y).dropna().unique()).issubset({0,1}):
        # Convert stable two-class labels to 0/1
        vals=sorted(pd.Series(y).dropna().unique(), key=str)
        y=y.map({vals[0]:0,vals[1]:1}).astype(int)
    return X, y.astype(int)

def build_model(X, seed=42):
    numeric=X.select_dtypes(include=np.number).columns.tolist()
    categorical=[c for c in X.columns if c not in numeric]
    num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                  ("onehot",OneHotEncoder(handle_unknown="ignore"))])
    prep=ColumnTransformer([("num",num,numeric),("cat",cat,categorical)])
    clf=RandomForestClassifier(n_estimators=200,random_state=seed,class_weight="balanced")
    return Pipeline([("prep",prep),("clf",clf)])

def metrics_for(model, X, y):
    pred=model.predict(X)
    prob=model.predict_proba(X)[:,1]
    out={
        "accuracy":float(accuracy_score(y,pred)),
        "precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0))
    }
    out["roc_auc"]=float(roc_auc_score(y,prob)) if len(set(y))==2 else None
    return out

def train_and_save(df, version, source_weeks, seed=42):
    X,y=prepare_xy(df)
    Xtr,Xval,ytr,yval=train_test_split(X,y,test_size=0.2,random_state=seed,stratify=y)
    model=build_model(Xtr,seed)
    model.fit(Xtr,ytr)
    m=metrics_for(model,Xval,yval)
    path=MODELS/f"{version}.joblib"
    joblib.dump(model,path)
    entry={
        "version":version,"model_path":str(path.relative_to(ROOT)),
        "sha256":sha256(path),"metrics":m,"source_weeks":source_weeks,
        "created_at":now(),"status":"candidate","seed":seed,
        "features":list(X.columns)
    }
    return model, entry


def train_candidate_fixed_validation(training_history, version, source_weeks, seed=42):
    """Train on reference-train plus prior labelled weeks; evaluate on fixed reference holdout.

    The current week's rows are deliberately excluded from training and candidate promotion metrics.
    """
    reference = pd.read_csv(DATA / "churn_reference_train.csv")
    X_ref, y_ref = prepare_xy(reference)
    X_train_ref, X_valid_ref, y_train_ref, y_valid_ref = train_test_split(
        X_ref, y_ref, test_size=0.2, random_state=seed, stratify=y_ref
    )
    prior_X, prior_y = [], []
    for df in training_history:
        Xp, yp = prepare_xy(df)
        prior_X.append(Xp); prior_y.append(yp)
    if prior_X:
        X_fit = pd.concat([X_train_ref, *prior_X], ignore_index=True)
        y_fit = pd.concat([y_train_ref, *prior_y], ignore_index=True)
    else:
        X_fit, y_fit = X_train_ref, y_train_ref
    model = build_model(X_fit, seed)
    model.fit(X_fit, y_fit)
    metrics = metrics_for(model, X_valid_ref, y_valid_ref)
    path = MODELS / f"{version}.joblib"
    joblib.dump(model, path)
    entry = {
        "version": version, "model_path": str(path.relative_to(ROOT)),
        "sha256": sha256(path), "metrics": metrics, "source_weeks": source_weeks,
        "created_at": now(), "status": "candidate", "seed": seed,
        "features": list(X_ref.columns),
        "validation_protocol": "fixed 20% reference holdout; current week excluded"
    }
    return model, entry

def ensure_baseline():
    reg=load_registry()
    if reg["production_version"]:
        entry=next((m for m in reg["models"] if m["version"]==reg["production_version"]),None)
        if entry and (ROOT/entry["model_path"]).exists(): return entry
    ref=DATA/"churn_reference_train.csv"
    if not ref.exists(): raise FileNotFoundError(f"Missing {ref}. Extract the supplied CSV files into data/.")
    df=pd.read_csv(ref)
    model,entry=train_and_save(df,"v1",["reference"],seed=42)
    entry["status"]="production"
    reg={"production_version":"v1","models":[entry]}
    save_registry(reg)
    (REPORTS/"baseline_metrics.json").write_text(json.dumps(entry["metrics"],indent=2))
    (REPORTS/"training_config.json").write_text(json.dumps({
        "seed":42,"test_size":0.2,"model":"RandomForestClassifier",
        "n_estimators":200,"reference_file":ref.name,"source_weeks":["reference"]
    },indent=2))
    logger.info("BASELINE_REGISTERED version=v1 metrics=%s",entry["metrics"])
    return entry

def production_entry():
    reg=load_registry()
    if not reg["production_version"]: return ensure_baseline()
    return next(m for m in reg["models"] if m["version"]==reg["production_version"])

def load_production_model():
    entry=production_entry()
    path=ROOT/entry["model_path"]
    if sha256(path)!=entry["sha256"]: raise RuntimeError("Production model checksum mismatch")
    return joblib.load(path),entry

def psi_numeric(ref, cur, bins=10, eps=1e-6):
    ref=pd.to_numeric(pd.Series(ref),errors="coerce").dropna()
    cur=pd.to_numeric(pd.Series(cur),errors="coerce").dropna()
    if ref.empty or cur.empty: return 0.0
    edges=np.unique(np.quantile(ref,np.linspace(0,1,bins+1)))
    if len(edges)<2: return 0.0
    edges[0],edges[-1]=-np.inf,np.inf
    rp=np.histogram(ref,bins=edges)[0].astype(float); cp=np.histogram(cur,bins=edges)[0].astype(float)
    rp=(rp+eps)/(rp.sum()+eps*len(rp)); cp=(cp+eps)/(cp.sum()+eps*len(cp))
    return float(np.sum((cp-rp)*np.log(cp/rp)))

def psi_categorical(ref,cur,eps=1e-6):
    r=pd.Series(ref).fillna("__MISSING__").astype(str)
    c=pd.Series(cur).fillna("__MISSING__").astype(str)
    cats=sorted(set(r.unique())|set(c.unique()))
    if not cats: return 0.0
    rp=r.value_counts(normalize=True).reindex(cats,fill_value=0).to_numpy()+eps
    cp=c.value_counts(normalize=True).reindex(cats,fill_value=0).to_numpy()+eps
    rp=rp/rp.sum(); cp=cp/cp.sum()
    return float(np.sum((cp-rp)*np.log(cp/rp)))

def drift_report(reference, current):
    target_ref=find_target(reference); target_cur=find_target(current)
    drop_ref=[target_ref]; drop_cur=[target_cur]
    for c in ["Customer_ID","customer_id","CustomerID","customerID","ID","id"]:
        if c in reference.columns: drop_ref.append(c)
        if c in current.columns: drop_cur.append(c)
    Xr=reference.drop(columns=drop_ref,errors="ignore")
    Xc=current.drop(columns=drop_cur,errors="ignore")
    rows=[]
    for col in Xr.columns:
        if col not in Xc.columns:
            rows.append({"feature":col,"psi":None,"status":"MISSING"})
            continue
        if pd.api.types.is_numeric_dtype(Xr[col]):
            score=psi_numeric(Xr[col],Xc[col])
        else:
            score=psi_categorical(Xr[col],Xc[col])
        rows.append({"feature":col,"psi":score,
                     "status":"high" if score>=0.25 else ("moderate" if score>=0.10 else "low")})
    return pd.DataFrame(rows)

def validate_week(df, reference):
    required=set(reference.columns)
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty: raise ValueError("Weekly dataset is empty")
    if df.isnull().all().any(): raise ValueError("A column contains only missing values")
    target=find_target(df)
    if df[target].nunique(dropna=True)<2: raise ValueError("Week needs both target classes for evaluation")
    return True

def next_version(reg):
    nums=[]
    for m in reg["models"]:
        try: nums.append(int(m["version"].lstrip("v")))
        except Exception: pass
    return f"v{max(nums+[0])+1}"

def candidate_passes(candidate_metrics, production_metrics, min_f1=0.0, tolerance=0.02):
    """Require a meaningful F1/AUC gain and no material regression on either metric."""
    cf=float(candidate_metrics.get("f1",0) or 0); pf=float(production_metrics.get("f1",0) or 0)
    ca=float(candidate_metrics.get("roc_auc",0) or 0); pa=float(production_metrics.get("roc_auc",0) or 0)
    meaningful_gain = (cf >= pf + 0.01) or (ca >= pa + 0.01)
    return cf >= min_f1 and meaningful_gain and cf >= pf-tolerance and ca >= pa-0.03

def promote_candidate(entry):
    reg=load_registry()
    current=reg.get("production_version")
    old=next(m for m in reg["models"] if m["version"]==current)
    # Revalidate artifact integrity before any production pointer change.
    path=ROOT/entry["model_path"]
    if not path.exists() or sha256(path)!=entry["sha256"]:
        raise RuntimeError("Candidate artifact missing or checksum mismatch")
    # Atomic registry write; keep history and only switch pointer after all checks.
    for m in reg["models"]:
        if m["version"]==current: m["status"]="archived"
    found=next((m for m in reg["models"] if m["version"]==entry["version"]),None)
    if found is None: reg["models"].append(entry.copy()); found=reg["models"][-1]
    found["status"]="production"
    reg["production_version"]=entry["version"]
    save_registry(reg)
    logger.info("PROMOTED from=%s to=%s",current,entry["version"])
    return current,entry["version"]

def run_week(week_num, policy="drift_low", threshold=0.10, inject_fault=None):
    started=time.perf_counter()
    week_path=DATA/f"churn_week_{week_num}.csv"
    run={"run_id":str(uuid.uuid4())[:8],"week":week_num,"policy":policy,
         "started_at":now(),"status":"started","production_before":None,
         "production_after":None,"promoted":False,"retry_count":0}
    reg_before=load_registry()
    run["production_before"]=reg_before.get("production_version")
    try:
        reference=pd.read_csv(DATA/"churn_reference_train.csv")
        if not week_path.exists(): raise FileNotFoundError(week_path)
        # Bounded retry applies only to transient ingestion timeouts/connections.
        attempts=0; retry_log=[]
        while True:
            attempts += 1
            try:
                if inject_fault=="transient_ingest" and attempts < 3:
                    raise TimeoutError("simulated transient ingestion timeout")
                week=pd.read_csv(week_path)
                break
            except (TimeoutError, ConnectionError) as exc:
                retry_log.append({"attempt":attempts,"error":f"{type(exc).__name__}: {exc}"})
                if attempts >= 3: raise
                time.sleep(0.05*(2**(attempts-1)))
        run["retry_count"]=attempts-1; run["ingestion_attempts"]=attempts; run["retry_log"]=retry_log
        # Fault injection is deliberately explicit and recorded.
        if inject_fault=="missing_column":
            week=week.drop(columns=[week.columns[0]])
        if inject_fault=="candidate_reject":
            # Process data normally, then force candidate gate rejection below.
            run["forced_candidate_reject"]=True
        validate_week(week,reference)
        drift=drift_report(reference,week)
        drift.to_csv(REPORTS/f"week_{week_num}_drift.csv",index=False)
        high_count=int((drift["psi"].fillna(0)>=threshold).sum())
        model,prod=load_production_model()
        Xw,yw=prepare_xy(week)
        weekly_metrics=metrics_for(model,Xw,yw)
        run.update({"drift_threshold":threshold,"drifting_features":high_count,
                    "mean_psi":float(drift["psi"].dropna().mean()),
                    "production_metrics":weekly_metrics})
        if policy=="never":
            trigger=False
        elif policy=="weekly":
            trigger=True
        elif policy in ("drift_low","drift_high"):
            trigger=bool((drift["psi"].fillna(0)>=threshold).any())
        else: raise ValueError(f"Unknown policy {policy}")
        run["retraining_triggered"]=trigger
        if trigger:
            # Candidate training may use only reference data and weeks strictly earlier than this week.
            prior_weeks=[]; source_weeks=["reference"]
            for w in range(1,week_num):
                prior_path=DATA/f"churn_week_{w}.csv"
                if prior_path.exists():
                    prior_df=pd.read_csv(prior_path)
                    validate_week(prior_df,reference)
                    prior_weeks.append(prior_df); source_weeks.append(f"week_{w}")
            version=next_version(load_registry())
            candidate_model,entry=train_candidate_fixed_validation(prior_weeks,version,source_weeks,seed=42)
            run["candidate_version"]=version
            run["candidate_metrics"]=entry["metrics"]
            run["candidate_evaluation"]="fixed reference holdout; current and future weeks excluded"
            # Compare candidate and production on the same fixed reference holdout.
            X_ref, y_ref = prepare_xy(reference)
            _, X_valid_ref, _, y_valid_ref = train_test_split(
                X_ref, y_ref, test_size=0.2, random_state=42, stratify=y_ref
            )
            production_reference_metrics=metrics_for(model,X_valid_ref,y_valid_ref)
            candidate_reference_metrics=metrics_for(candidate_model,X_valid_ref,y_valid_ref)
            pass_gate=candidate_passes(candidate_reference_metrics,production_reference_metrics,min_f1=0.0,tolerance=0.02)
            run["production_reference_metrics"]=production_reference_metrics
            run["candidate_metrics"]=candidate_reference_metrics
            if inject_fault=="candidate_reject": pass_gate=False
            run["candidate_passed"]=bool(pass_gate)
            if pass_gate:
                # Store validation results in registry entry; promotion only via workflow.
                entry["metrics"]=candidate_week_metrics
                entry["status"]="candidate"
                entry["evaluated_on_week"]=week_num
                old,new=promote_candidate(entry)
                run["promoted"]=True
                run["promoted_from"]=old
                run["promoted_to"]=new
            else:
                run["decision"]="candidate_rejected"
                logger.warning("CANDIDATE_REJECTED week=%s version=%s",week_num,version)
        else:
            run["decision"]="no_retraining"
        # Feature-only escalation rule: no current-week labels used for the decision.
        severe_features=int((drift["psi"].fillna(0)>=0.25).sum())
        mean_psi=run["mean_psi"]
        escalation=bool(severe_features>=2 or mean_psi>=0.25)
        run["severe_drift_features"]=severe_features
        run["escalation"]=escalation
        if escalation:
            run["escalation_reason"]=f"feature-only rule: severe_PSI_features={severe_features}; mean_PSI={mean_psi:.3f}"
            logger.warning("ESCALATION week=%s %s",week_num,run["escalation_reason"])
        run["status"]="success"
    except Exception as e:
        run["status"]="failed"
        run["error"]=f"{type(e).__name__}: {e}"
        run["retry_count"]=0
        logger.exception("RUN_FAILED week=%s policy=%s",week_num,policy)
    finally:
        run["duration_seconds"]=round(time.perf_counter()-started,4)
        run["finished_at"]=now()
        run["production_after"]=load_registry().get("production_version")
        if run["status"]=="failed" and run["production_before"]!=run["production_after"]:
            # This is a safety violation; log loudly. Normal failure path should never change it.
            logger.error("SAFETY_VIOLATION production changed after failed run")
        save_run(run)
    return run

def reset_artifacts():
    """Reset registry and model artifacts for reproducible fresh experiments."""
    for p in MODELS.glob("*.joblib"): p.unlink()
    for p in [REGISTRY,RUNS]:
        if p.exists(): p.unlink()
    for p in REPORTS.glob("*.csv"): p.unlink()
    return ensure_baseline()
