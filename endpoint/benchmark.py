"""Basic local HTTP latency benchmark. Process cold starts are approximated by restarting the server externally."""
import argparse, time, json, statistics
from pathlib import Path
import httpx
from src.core import DATA, find_target

def sample_payload():
    import pandas as pd
    df=pd.read_csv(DATA/"churn_reference_train.csv")
    target=find_target(df)
    row=df.drop(columns=[target]+[c for c in ["Customer_ID","customer_id","CustomerID","customerID","ID","id"] if c in df.columns]).iloc[0]
    return {"features":{k:(v.item() if hasattr(v,"item") else v) for k,v in row.to_dict().items()}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--url",default="http://127.0.0.1:8000/predict")
    p.add_argument("--n",type=int,default=30)
    p.add_argument("--label",default="warm")
    args=p.parse_args()
    payload=sample_payload()
    lat=[]
    with httpx.Client(timeout=20) as client:
        for _ in range(args.n):
            t=time.perf_counter()
            r=client.post(args.url,json=payload); r.raise_for_status()
            lat.append((time.perf_counter()-t)*1000)
    result={"label":args.label,"count":len(lat),"mean_ms":statistics.mean(lat),
            "median_ms":statistics.median(lat),"max_ms":max(lat),
            "p95_ms":sorted(lat)[min(len(lat)-1,int(.95*len(lat)))] ,"latencies_ms":lat}
    out=Path("artifacts/reports/latency_"+args.label+".json")
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="latencies_ms"},indent=2))
if __name__=="__main__": main()
