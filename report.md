# Drift-Triggered Retraining with an Orchestrated Serverless Endpoint

## Objective and method
This project evaluates a reproducible churn model and a controlled weekly retraining workflow on one reference dataset and eight simulated weekly datasets. A Random Forest pipeline uses fixed seed 42 and preprocessing for numeric/categorical variables. Each model version is saved with validation metrics, source-week metadata and a SHA-256 checksum. Promotion changes the registry's production pointer; prediction code does not hard-code a model filename.

Feature drift is measured using Population Stability Index (PSI) per feature and week. PSI below 0.10 is treated as low, 0.10–0.25 as moderate, and at least 0.25 as high. Model performance is measured separately on each week's labelled data for retrospective analysis. The operational retraining/promotion decision uses current-week feature drift and candidate evaluation on reference/prior labelled data only; current and future week labels are not used to decide promotion for that week.

## Results
The baseline model's held-out reference metrics are recorded in `artifacts/reports/baseline_metrics.json`. The baseline model's retrospective weekly F1 varies from 0.271 to 0.504. Mean feature PSI rises sharply in later weeks: week 6 mean PSI is 0.310, week 7 is 0.501, and week 8 is 0.341. This shows that feature drift and performance loss are distinct signals; weekly performance is reported separately rather than assuming drift automatically means a model has failed.

### Policy comparison
Retraining cost is assumed to be 10 abstract units per attempt and promotion cost 2 units per successful promotion. These are comparison assumptions, not provider billing estimates.

| Policy | Retraining attempts | Promotions | Mean weekly F1 | Total cost units |
|---|---:|---:|---:|---:|
| drift_high | 4 | 4 | 0.401 | 48 |
| drift_low | 5 | 5 | 0.402 | 60 |
| never | 0 | 0 | 0.383 | 0 |
| weekly | 8 | 5 | 0.401 | 90 |

The high-threshold policy produced mean weekly F1 0.401 at total cost 48 units, versus 0.402 F1 and 60 units for the low threshold. The low threshold's F1 advantage is only about 0.0008, while the high threshold costs 20% less. Therefore, the high threshold is recommended as the more economical balance for this dataset, while acknowledging that the F1 differences are small and validation results are dataset-specific.

## Safety, retry and escalation
Three fault cases are recorded: a missing required column fails validation; a forced candidate rejection does not promote; and a transient timeout is retried twice before succeeding. The invalid-data and candidate-rejection cases preserve the production registry version and the endpoint-resolved model version. The retry policy is bounded to two retries with exponential backoff and applies only to transient timeout/connection errors, not schema errors.

Escalation is feature-only to avoid current-week label leakage: escalate if at least two features have PSI ≥0.25, or mean feature PSI ≥0.25. In the final eight-week run this condition fires in weeks 6, 7, 8. The run history stores the PSI evidence and reason.

## Endpoint and limitations
The notebook measures 30 local cold model-load calls and 30 warm cached calls; the recorded means are 34.3 ms and 10.7 ms in the preparation environment. These are not cloud-provider cold starts. A local FastAPI endpoint and an AWS Lambda-compatible handler are included. A hosted serverless deployment still requires the student's cloud account, packaging, credentials, and provider-side measurements; this report does not claim that cloud deployment has already occurred.

## Reproducibility and artifacts
Run `00_end_to_end_project.ipynb` top-to-bottom. It creates the registry evidence, eight-week run history, per-feature drift CSV and heatmap, orchestration graph, policy comparison/timeline, fault/retry logs, escalation evidence and endpoint model-switch proof. The selected final registry includes baseline `v1` and promoted versions; the endpoint responds with the active production version after promotion. AI assistance was used for planning, code scaffolding and documentation; the team must understand and verify all code and results before submission.
