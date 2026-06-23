# MLOps Churn Pipeline

End-to-end MLOps pipeline: a code or data change triggers data versioning →
training → an evaluation gate that blocks regressing models → automatic deploy
→ drift monitoring. The model (churn classifier on tabular data) is deliberately
simple — the **automated, monitored plumbing is the deliverable.**

## Stack

| Concern | Tool |
|---|---|
| Data versioning | DVC + Google Drive remote |
| Training | scikit-learn |
| Experiment tracking + registry | MLflow |
| Evaluation gate | Plain Python in CI |
| CI/CD | GitHub Actions |
| Serving | FastAPI + Docker |
| Deploy target | Hugging Face Spaces (Docker SDK) |
| Drift monitoring | Evidently |

**Dataset:** Telco Customer Churn (~7,043 rows, 21 columns, binary `Churn` target).

## Status

- [ ] Week 1 — Data + DVC versioning
- [ ] Week 2 — Training + MLflow tracking
- [ ] Week 3 — Evaluation gate (champion/challenger)
- [ ] Week 4 — CI/CD with GitHub Actions
- [ ] Week 5 — FastAPI serving + Docker + HF Spaces deploy
- [ ] Week 6 — Evidently drift monitoring + README polish

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the Telco Customer Churn CSV from Kaggle, save it as
`data/raw/telco.csv`, then:

```powershell
python scripts\make_batches.py   # creates batch_0/1/2.csv
```

## Design decisions

- **Champion stored as a DVC-tracked artifact.** CI has no persistent state, so
  the current best model + its `metrics.json` live in DVC. The gate pulls the
  champion, compares the challenger, and promotes on a win. MLflow handles
  experiment logging (UI/screenshots) decoupled from the gate logic.
