import json
import pandas as pd
from src.core import DATA, reset_artifacts, run_week, load_registry, RUNS

def test_invalid_week_does_not_promote():
    reset_artifacts()
    before=load_registry()["production_version"]
    result=run_week(3,"weekly",0.10,inject_fault="missing_column")
    after=load_registry()["production_version"]
    assert result["status"]=="failed"
    assert before==after
    assert result["production_before"]==result["production_after"]

def test_candidate_rejection_does_not_promote():
    reset_artifacts()
    before=load_registry()["production_version"]
    result=run_week(5,"weekly",0.10,inject_fault="candidate_reject")
    after=load_registry()["production_version"]
    assert result["status"]=="success"
    assert result.get("promoted") is False
    assert before==after
