# Five-minute demonstration script

## 0:00–0:30 — Problem and setup
“This project studies how a churn model can detect changing customer data, retrain only when justified, and promote a candidate safely. The reference dataset is the baseline and the eight weekly CSVs simulate incoming data.” Show the project folder and notebook.

## 0:30–1:10 — Baseline and registry
Show the baseline metrics and `artifacts/registry_evidence.json`. Explain that each model has a version, validation metrics, source weeks and SHA-256 checksum. The registry production pointer is the only model selector used by the endpoint.

## 1:10–2:00 — Drift and weekly workflow
Show `feature_drift_heatmap.png` and the eight rows in `artifacts/run_history.json`. Explain PSI bands (below 0.10 low, 0.10–0.25 moderate, at least 0.25 high). Clarify that current-week labels are used for retrospective performance charts only; they are not used to train or promote a candidate during that week.

## 2:00–2:45 — Policies and escalation
Show `policy_comparison.csv` and `policy_timeline.png`. Explain the cost assumptions (10 units per retraining attempt, 2 per promotion). Show an escalation entry from week 6 or later: at least two features have PSI ≥0.25 or mean PSI ≥0.25.

## 2:45–3:35 — Inference and switching
Show `endpoint_model_switch_evidence.json`; the same prediction function resolves the current production model from the registry. For a live local API demo, run `uvicorn endpoint.app:app --host 127.0.0.1 --port 8000` and open `/docs`. Mention the notebook measured 30 local cold model loads and 30 warm cached calls; do not call these cloud cold starts.

## 3:35–4:25 — Fault handling
Show `fault_injection_evidence.json`: invalid schema fails validation and leaves registry/endpoint version unchanged; forced candidate rejection prevents promotion; transient ingestion timeout is retried twice and then succeeds. Retry policy is bounded and only transient errors are retried.

## 4:25–5:00 — Conclusion and limitation
“The selected policy is drift-triggered high threshold because it achieved near-identical mean F1 to low-threshold retraining at lower illustrative cost. The notebook and local endpoint run on a PC. A real AWS Lambda deployment requires account access and separate provider cold/warm evidence, which is documented separately.”
