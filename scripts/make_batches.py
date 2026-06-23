"""Split the Telco Churn CSV into 3 ordered batches to simulate incoming data.

batch_0 -> reference / training set
batch_1 -> first "future" batch (retraining + drift)
batch_2 -> second "future" batch

The Telco CSV has no real time column, so we just chunk it. That's the
deliberate simulation: pretend each batch arrived later in time.

Usage:
    python scripts/make_batches.py
"""
import os
import pandas as pd

RAW = os.path.join("data", "raw", "telco.csv")
OUT_DIR = os.path.join("data", "raw")


def main():
    if not os.path.exists(RAW):
        raise SystemExit(
            f"Could not find {RAW}. Download the Telco Customer Churn CSV from "
            "Kaggle and save it there as 'telco.csv' first."
        )

    df = pd.read_csv(RAW)
    # Shuffle once with a fixed seed so batches are comparable but not
    # accidentally sorted by some hidden order in the file.
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n = len(df)
    thirds = [df.iloc[: n // 3], df.iloc[n // 3 : 2 * n // 3], df.iloc[2 * n // 3 :]]

    for i, chunk in enumerate(thirds):
        path = os.path.join(OUT_DIR, f"batch_{i}.csv")
        chunk.to_csv(path, index=False)
        print(f"wrote {path}  ({len(chunk)} rows)")


if __name__ == "__main__":
    main()
