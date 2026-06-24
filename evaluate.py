"""Evaluation gate: block a challenger that regresses against the champion.

Compares metrics/candidate.json (the freshly trained challenger) against
metrics/champion.json (the current best). Exits non-zero -- which fails CI --
if the challenger's ROC-AUC is worse than the champion's beyond a small
tolerance. If there is no champion yet, the challenger passes by default.

Usage:
    python evaluate.py
"""
import json
import os
import sys

CANDIDATE = "metrics/candidate.json"
CHAMPION = "metrics/champion.json"
METRIC = "roc_auc"
TOL = 0.005  # allow tiny noise; only block real regressions


def main():
    with open(CANDIDATE) as f:
        chal = json.load(f)

    if not os.path.exists(CHAMPION):
        print(f"No champion yet -> challenger passes ({METRIC}={chal[METRIC]:.4f}).")
        return

    with open(CHAMPION) as f:
        champ = json.load(f)

    chal_score = chal[METRIC]
    champ_score = champ[METRIC]

    if chal_score + TOL < champ_score:
        print(
            f"REGRESSION: challenger {METRIC}={chal_score:.4f} < "
            f"champion {champ_score:.4f} (tol={TOL}). Blocking release."
        )
        sys.exit(1)

    print(
        f"PASS: challenger {METRIC}={chal_score:.4f} >= "
        f"champion {champ_score:.4f} (tol={TOL})."
    )


if __name__ == "__main__":
    main()
