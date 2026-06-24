"""
Compute inter-annotator agreement between two label columns.

Usage:
  1. Fill annotator_b in inter_annotator_sample.csv (35 posts, independent labels)
  2. python compute_agreement.py

Requires both annotator_a and annotator_b columns to be filled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

SAMPLE_PATH = Path("inter_annotator_sample.csv")
OUTPUT_PATH = Path("inter_annotator_results.json")
VALID_LABELS = {"analysis", "hot_take", "reaction"}


def main() -> None:
    if not SAMPLE_PATH.exists():
        raise SystemExit(f"Missing {SAMPLE_PATH}")

    df = pd.read_csv(SAMPLE_PATH)
    for col in ("annotator_a", "annotator_b"):
        if col not in df.columns:
            raise SystemExit(f"CSV must have column: {col}")

    missing_b = df["annotator_b"].isna() | (df["annotator_b"].astype(str).str.strip() == "")
    if missing_b.any():
        print(f"annotator_b still blank on {missing_b.sum()} rows. Fill all labels first.")
        print("Share inter_annotator_sample.csv with a second person — do not show annotator_a.")
        return

    df = df[df["annotator_a"].isin(VALID_LABELS) & df["annotator_b"].isin(VALID_LABELS)]
    if len(df) < 30:
        raise SystemExit("Need at least 30 labeled pairs.")

    agree = (df["annotator_a"] == df["annotator_b"]).sum()
    pct = agree / len(df)
    kappa = cohen_kappa_score(df["annotator_a"], df["annotator_b"])

    disagreements = df[df["annotator_a"] != df["annotator_b"]][
        ["text", "annotator_a", "annotator_b"]
    ].to_dict(orient="records")

    results = {
        "n_examples": len(df),
        "percent_agreement": round(pct, 4),
        "cohens_kappa": round(float(kappa), 4),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "interpretation": (
            "Strong agreement" if kappa >= 0.8
            else "Moderate agreement" if kappa >= 0.6
            else "Fair agreement — label definitions may need tightening"
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
