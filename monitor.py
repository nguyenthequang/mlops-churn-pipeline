"""Data-drift monitoring with Evidently.

Compares a reference batch (the data the champion was trained on) against a
"current" batch of incoming data, and writes an HTML report flagging features
whose distribution has shifted. This is the post-deployment half of MLOps:
knowing *when* the world has moved away from what the model learned.

Target/id columns are dropped so this reflects input-feature drift only (in
production you'd score incoming data before its labels exist).

Usage:
    python monitor.py                                   # batch_0 vs batch_1
    python monitor.py --current data/raw/batch_2.csv    # batch_0 vs batch_2
"""
import argparse

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

DROP_COLS = ["customerID", "Churn"]


def load(path):
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/raw/batch_0.csv")
    ap.add_argument("--current", default="data/raw/batch_1.csv")
    ap.add_argument("--out", default="drift_report.html")
    args = ap.parse_args()

    ref = load(args.reference)
    cur = load(args.current)

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=cur, reference_data=ref)
    snapshot.save_html(args.out)

    # Pull the headline numbers out of the snapshot for a console summary.
    metrics = snapshot.dict().get("metrics", [])
    overall = None
    columns = []
    for m in metrics:
        name = m.get("metric_name", "")
        value = m.get("value")
        if name.startswith("DriftedColumnsCount"):
            overall = value  # {"count": ..., "share": ...}
        elif name.startswith("ValueDrift"):
            cfg = m.get("config", {})
            col = cfg.get("column", "?")
            threshold = cfg.get("threshold", 0.1)
            columns.append((col, value, value > threshold))

    print("Drift summary:")
    if overall:
        print(
            f"  drifted columns: {int(overall['count'])}/{len(columns)} "
            f"(share={overall['share']:.2f})"
        )
    for col, score, drifted in sorted(columns, key=lambda c: -c[1]):
        print(f"  [{'DRIFT' if drifted else ' ok  '}] {col}: {score:.4f}")

    print(f"\nReference: {args.reference}  ({len(ref)} rows)")
    print(f"Current:   {args.current}  ({len(cur)} rows)")
    print(f"Wrote drift report -> {args.out}")


if __name__ == "__main__":
    main()
