Create these notebooks in Jupyter if your instructor expects `.ipynb` files:
1. `01_baseline_model.ipynb`: call `ensure_baseline()` and inspect baseline metrics/config.
2. `02_drift_investigation.ipynb`: load reference and weekly CSVs; use `drift_report()` and `metrics_for()`, then plot heatmaps and performance.
3. `03_policy_comparison.ipynb`: reset registry and run all eight weeks for each of `never`, `weekly`, `drift_low`, `drift_high`; compare cost and metrics.
The dashboard and reusable Python modules are provided so you can run the project quickly without writing all logic inside notebooks.
