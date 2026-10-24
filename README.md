# Drift-Triggered Retraining with an Orchestrated Serverless Endpoint

A reproducible MLOps coursework project for churn prediction using eight simulated weekly datasets. The primary deliverable is `00_end_to_end_project.ipynb`; it creates the evidence files used in the report and demonstration.

## Tasks covered

1. **Baseline and registry:** reproducible Random Forest pipeline, fixed random seed, baseline validation metrics, model checksum and JSON registry.
2. **Inference endpoint:** registry-driven prediction logic in `endpoint/app.py` (local FastAPI) and `endpoint/lambda_function.py` (AWS Lambda/API Gateway-compatible handler). The notebook measures 30 cold model-load and 30 warm cached handler calls. These are local measurements, not hosted-provider cold starts.
3. **Drift investigation:** PSI for each model feature in each of eight weeks; weekly retrospective F1/ROC-AUC/accuracy; drift and performance are reported separately.
4. **Orchestrated workflow:** ingest, schema validation, PSI detection, policy trigger, candidate training, quality gate, promotion through the registry, run history and escalation. Candidate training uses reference data and labelled weeks strictly before the current week. Current-week labels are only used for retrospective performance reporting, not candidate promotion or escalation.
5. **Policy study:** never, weekly, drift-triggered PSI ≥0.10, and drift-triggered PSI ≥0.25; transparent illustrative costs; comparison table and timeline chart.
6. **Failure handling:** invalid-schema fault, forced candidate rejection and transient timeout with bounded retries. Evidence includes registry and endpoint version before/after checks.
7. **Escalation:** feature-only condition: at least two features with PSI ≥0.25, or mean PSI ≥0.25. Actual firing weeks and reasons are logged.

## Run on Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook
```

Open `00_end_to_end_project.ipynb` in the project root and **Run All**. Keep the `data` folder beside the notebook. The notebook should finish with `FINAL CHECKS PASSED` and writes outputs into `artifacts/` and `artifacts/reports/`.

## Run the local HTTP endpoint

In a terminal at the project root:

```powershell
uvicorn endpoint.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. `GET /health` reports the current model version. `POST /predict` accepts JSON like:

```json
{"features":{"Tenure_Months":12,"Contract_Type":"Month-to-month","Internet_Service":"Fiber optic","Monthly_Charges":75.0,"Total_Charges":900.0,"Support_Calls_Last_Month":2,"Avg_Data_Usage_GB":25.0,"Late_Payments":1}}
```

The endpoint reads the production pointer from `artifacts/registry.json`; promotion changes the model version without changing prediction code.

## Run workflow from a terminal

```powershell
python orchestrator.py --reset --policy drift_high
python orchestrator.py --reset --policy never
python orchestrator.py --reset --policy weekly
python orchestrator.py --reset --policy drift_low
```

The notebook is the recommended end-to-end path because it preserves policy-specific evidence and writes the final eight-week history. A terminal policy run intentionally resets the local experiment state.

## Evidence files

- `artifacts/registry_evidence.json` and `artifacts/reports/final_registry_evidence.json`: baseline and promoted versions, checksums and metrics.
- `artifacts/run_history.json`: exactly eight records for the final sequential workflow.
- `artifacts/reports/feature_drift_all_weeks.csv`, `baseline_weekly_performance.csv`, and `feature_drift_heatmap.png`.
- `artifacts/reports/orchestration_graph.png`.
- `artifacts/reports/policy_comparison.csv`, `policy_timeline.png`, and `policy_weekly_results.csv`.
- `artifacts/reports/fault_injection_evidence.json`, `retry_policy_evidence.json`, and `endpoint_model_switch_evidence.json`.
- `artifacts/reports/final_escalation_evidence.json`.
- `report.pdf` (five pages or fewer) and `DEMO_SCRIPT.md`.

## Policy decision and assumptions

Retraining cost = 10 abstract cost units per attempt; promotion cost = 2 units per successful promotion. The supplied data's observed policy comparison slightly favors the low threshold on mean weekly F1, but the high threshold has nearly identical mean F1 at lower total cost; see `artifacts/reports/policy_comparison.csv` and the report for the actual figures. Cost units are illustrative, not cloud billing quotes.

## Cloud deployment limitation

`endpoint/lambda_function.py` and `template.yaml` are deployment-ready scaffolding for AWS SAM. A real AWS deployment and provider-side cold/warm benchmark require an AWS account, credentials, deployment permissions, and packaging the model/dependencies. This environment cannot deploy to your cloud account. Do not present the notebook's local load/cache timings as cloud cold starts. Follow `AWS_DEPLOYMENT.md` if your professor requires hosted serverless evidence.

## Tests

```powershell
pytest -q
```

## AI-use disclosure

AI assistance from OpenAI ChatGPT was used to help plan, scaffold, debug and document parts of this project. The student team must review and understand the code, run it on the demonstration PC, verify the outputs, and disclose any changes made. Do not claim a cloud deployment unless you have actually deployed and tested it.
