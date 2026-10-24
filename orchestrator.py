import argparse
from src.core import ensure_baseline, run_week, reset_artifacts, ART, RUNS
import json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reset",action="store_true",help="Reset registry and models before run")
    parser.add_argument("--policy",default="drift_low",choices=["never","weekly","drift_low","drift_high"])
    parser.add_argument("--threshold",type=float,default=None)
    parser.add_argument("--fault-week",type=int,default=None,choices=range(1,9))
    parser.add_argument("--fault",choices=["missing_column","candidate_reject","transient_ingest"],default=None)
    args=parser.parse_args()
    if args.reset: reset_artifacts()
    else: ensure_baseline()
    threshold=args.threshold if args.threshold is not None else (0.10 if args.policy=="drift_low" else 0.25)
    results=[]
    for week in range(1,9):
        fault=args.fault if args.fault_week==week else None
        result=run_week(week,args.policy,threshold,fault)
        results.append(result)
        print(f"Week {week}: {result['status']} | decision={result.get('decision', 'n/a')} | production={result.get('production_after')}")
    print(f"Run history saved to {RUNS}")
    print(json.dumps({"weeks":len(results),"failed":sum(r["status"]=="failed" for r in results),
                      "promotions":sum(r.get("promoted",False) for r in results),
                      "escalations":sum(r.get("escalation",False) for r in results)},indent=2))
if __name__=="__main__": main()
