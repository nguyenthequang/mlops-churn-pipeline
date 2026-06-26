# Screenshots for the main README

Drop these four PNGs in this folder (the main README references them by these
exact filenames):

| File | What to capture |
|---|---|
| `ci-green.png` | A passing GitHub Actions run — the green ✅ with the `train` and `evaluation gate` steps. From the Actions tab. |
| `pr-blocked.png` | The `demo/worse-model` PR showing the red ❌ where the gate failed. |
| `mlflow.png` | The MLflow experiment view on DagsHub showing logged runs + metrics. |
| `drift.png` | The Evidently report (`drift_report.html`) opened in a browser, showing the drifted features. Regenerate it with `python scripts/simulate_drift.py && python monitor.py --current data/raw/batch_drift.csv`. |

Tip: crop to the relevant area and keep each image under ~500 KB.
