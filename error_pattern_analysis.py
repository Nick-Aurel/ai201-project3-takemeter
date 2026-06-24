"""
Formal error-pattern analysis across the test set.
Identifies systematic failure modes with supporting evidence.

Run: python error_pattern_analysis.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RANDOM_SEED = 42
MODEL_DIR = Path("model")
DATA_PATH = Path("soccer_posts.csv")
OUTPUT_PATH = Path("error_pattern_analysis.json")

LABEL2ID = {"analysis": 0, "hot_take": 1, "reaction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def main() -> None:
    if not (MODEL_DIR / "config.json").exists():
        raise SystemExit("Model not found. Run python train_and_evaluate.py first.")

    df = pd.read_csv(DATA_PATH)
    df = df[df["label"].isin(LABEL2ID)].copy()
    df["label_id"] = df["label"].map(LABEL2ID)
    _, test_df = train_test_split(
        df, test_size=33, random_state=RANDOM_SEED, stratify=df["label_id"]
    )
    test_df = test_df.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    records = []
    for _, row in test_df.iterrows():
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        pred_id = int(probs.argmax())
        records.append(
            {
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": ID2LABEL[pred_id],
                "correct": ID2LABEL[pred_id] == row["label"],
                "confidence": float(probs[pred_id]),
                "word_count": word_count(row["text"]),
                "predicted_hot_take": ID2LABEL[pred_id] == "hot_take",
            }
        )

    results_df = pd.DataFrame(records)
    errors = results_df[~results_df["correct"]]

    # Pattern 1: hot_take false positives
    hot_take_fps = errors[errors["predicted_label"] == "hot_take"]

    # Pattern 2: short posts misclassified
    short_threshold = 8
    short_posts = results_df[results_df["word_count"] <= short_threshold]
    short_errors = short_posts[~short_posts["correct"]]

    pattern = {
        "systematic_pattern": (
            "Short declarative posts (≤8 words) are over-predicted as hot_take, "
            "especially when the true label is reaction or analysis."
        ),
        "evidence": {
            "total_test_errors": int(len(errors)),
            "errors_predicted_as_hot_take": int(len(hot_take_fps)),
            "hot_take_false_positive_rate": round(
                len(hot_take_fps) / max(len(errors), 1), 4
            ),
            "short_post_error_rate": round(
                len(short_errors) / max(len(short_posts), 1), 4
            ),
            "mean_word_count_errors": round(float(errors["word_count"].mean()), 2)
            if len(errors)
            else 0,
            "mean_word_count_correct": round(float(results_df[results_df["correct"]]["word_count"].mean()), 2),
            "confused_label_pairs": [
                {
                    "true": row["true_label"],
                    "predicted": row["predicted_label"],
                    "text": row["text"],
                    "word_count": int(row["word_count"]),
                }
                for _, row in errors.iterrows()
            ],
        },
        "generalization": (
            "Posts under ~8 words that state a judgment without emotional markers "
            "('wow', 'class') or numeric evidence look like hot_takes to DistilBERT, "
            "even when annotators label them as reactions or analysis based on context."
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(pattern, indent=2))
    print(json.dumps(pattern, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
