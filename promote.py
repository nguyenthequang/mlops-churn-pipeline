"""Promote the current challenger to champion.

Overwrites the champion model + metrics with the candidate, then re-tracks the
champion model dir with DVC and pushes it to the remote. Run this only after
evaluate.py has passed (the CI workflow gates it). Also used once to seed the
very first champion.

The champion model lives in champion/ (DVC-tracked, since model files are large
and binary); the champion metrics live in metrics/champion.json (git-tracked, so
the gate can read it without a dvc pull of the model).

Usage:
    python promote.py
"""
import os
import shutil
import subprocess
import sys

CAND_MODEL = "models/candidate.joblib"
CAND_METRICS = "metrics/candidate.json"
CHAMP_DIR = "champion"
CHAMP_MODEL = os.path.join(CHAMP_DIR, "model.joblib")
CHAMP_METRICS = "metrics/champion.json"


def dvc(*args):
    """Run a dvc subcommand, resolving the dvc executable next to this Python."""
    exe = shutil.which("dvc")
    if exe is None:
        exe = os.path.join(os.path.dirname(sys.executable), "dvc")
    subprocess.run([exe, *args], check=True)


def main():
    os.makedirs(CHAMP_DIR, exist_ok=True)
    shutil.copyfile(CAND_MODEL, CHAMP_MODEL)
    shutil.copyfile(CAND_METRICS, CHAMP_METRICS)
    print(f"copied {CAND_MODEL} -> {CHAMP_MODEL}")
    print(f"copied {CAND_METRICS} -> {CHAMP_METRICS}")

    dvc("add", CHAMP_DIR)
    dvc("push")
    print("Champion promoted and pushed to DVC remote.")
    print("Commit champion.dvc + metrics/champion.json to git to finalize.")


if __name__ == "__main__":
    main()
