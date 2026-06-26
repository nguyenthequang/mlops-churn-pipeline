"""Create a deliberately drifted batch to demonstrate drift detection.

The real batches are random splits of one CSV, so their distributions are
identical -- a drift monitor correctly reports "no drift" on them. To show the
monitor *catching* drift (and to produce a meaningful report screenshot), this
script simulates a plausible real-world shift in incoming customers:

  * a ~25% price increase (MonthlyCharges / TotalCharges up)
  * a younger customer base (lower tenure)
  * a shift toward fiber-optic internet

Writes data/raw/batch_drift.csv. Then:
    python monitor.py --current data/raw/batch_drift.csv

Usage:
    python scripts/simulate_drift.py
"""
import numpy as np
import pandas as pd

SRC = "data/raw/batch_2.csv"
OUT = "data/raw/batch_drift.csv"


def main():
    rng = np.random.default_rng(7)
    df = pd.read_csv(SRC)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Price increase.
    df["MonthlyCharges"] = (df["MonthlyCharges"] * 1.25).round(2)
    df["TotalCharges"] = (df["TotalCharges"] * 1.25).round(2)

    # Younger customer base: pull tenure down, clip at 0.
    df["tenure"] = (df["tenure"] - 20).clip(lower=0)

    # Shift internet mix toward fiber optic for ~40% of non-fiber rows.
    mask = (df["InternetService"] != "Fiber optic") & (
        rng.random(len(df)) < 0.40
    )
    df.loc[mask, "InternetService"] = "Fiber optic"

    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows) with simulated drift")


if __name__ == "__main__":
    main()
