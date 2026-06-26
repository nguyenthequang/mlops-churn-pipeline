# MLOps Churn Pipeline

[![ml-pipeline](https://github.com/nguyenthequang/mlops-churn-pipeline/actions/workflows/pipeline.yml/badge.svg)](https://github.com/nguyenthequang/mlops-churn-pipeline/actions/workflows/pipeline.yml)

**Live API:** https://QuangNguyen85-churn-api.hf.space ([/docs](https://QuangNguyen85-churn-api.hf.space/docs))

End-to-end MLOps pipeline: a code or data change triggers data versioning →
training → an evaluation gate that blocks regressing models → automatic deploy
→ drift monitoring. The model (churn classifier on tabular data) is deliberately
simple — the **automated, monitored plumbing is the deliverable.**

## Stack

| Concern | Tool |
|---|---|
| Data versioning | DVC + DagsHub remote |
| Training | scikit-learn |
| Experiment tracking + registry | MLflow |
| Evaluation gate | Plain Python in CI |
| CI/CD | GitHub Actions |
| Serving | FastAPI + Docker |
| Deploy target | Hugging Face Spaces (Docker SDK) |
| Drift monitoring | Evidently |

**Dataset:** Telco Customer Churn (~7,043 rows, 21 columns, binary `Churn` target).

## Status

- [x] Week 1 — Data + DVC versioning
- [x] Week 2 — Training + MLflow tracking
- [x] Week 3 — Evaluation gate (champion/challenger)
- [x] Week 4 — CI/CD with GitHub Actions
- [x] Week 5 — FastAPI serving + Docker + HF Spaces deploy
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
- **DagsHub as the DVC remote (not Google Drive).** Free personal Google
  accounts can't back a DVC remote in 2026: interactive OAuth is blocked by
  Google, and service accounts have no Drive storage quota (a Shared Drive needs
  paid Workspace). DagsHub gives token-based auth that works locally and in CI
  with no browser step, plus a hosted MLflow server.
