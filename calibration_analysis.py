"""
Confidence calibration analysis for the fine-tuned model.
Bins test-set predictions by confidence and checks whether higher
confidence corresponds to higher accuracy.

Run: python calibration_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RANDOM_SEED = 42
MODEL_DIR = Path("model")
DATA_PATH = Path("soccer_posts.csv")
OUTPUT_JSON = Path("calibration_results.json")
OUTPUT_PNG = Path("calibration_curve.png")

LABEL2ID = {"analysis": 0, "hot_take": 1, "reaction": 2}


def get_test_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["label"].isin(LABEL2ID)].copy()
    df["label_id"] = df["label"].map(LABEL2ID)
    _, test_df = train_test_split(
        df, test_size=33, random_state=RANDOM_SEED, stratify=df["label_id"]
    )
    return test_df.reset_index(drop=True)


def predict(model, tokenizer, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    confidences = []
    preds = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        pred = int(np.argmax(probs))
        preds.append(pred)
        confidences.append(float(probs[pred]))
    return np.array(preds), np.array(confidences)


def calibration_bins(confidences, correct, n_bins: int = 5):
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= low) & (confidences < high)
        else:
            mask = (confidences >= low) & (confidences <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin": f"{low:.0%}–{high:.0%}",
                "count": count,
                "mean_confidence": round(float(confidences[mask].mean()), 4),
                "accuracy": round(float(correct[mask].mean()), 4),
                "gap": round(float(confidences[mask].mean() - correct[mask].mean()), 4),
            }
        )
    return rows


def main() -> None:
    if not (MODEL_DIR / "config.json").exists():
        raise SystemExit("Model not found. Run python train_and_evaluate.py first.")

    df = pd.read_csv(DATA_PATH)
    test_df = get_test_split(df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    preds, confidences = predict(model, tokenizer, test_df["text"].tolist())
    y_true = test_df["label_id"].to_numpy()
    correct = preds == y_true

    bins = calibration_bins(confidences, correct, n_bins=5)
    ece = float(np.mean([abs(r["mean_confidence"] - r["accuracy"]) * r["count"] for r in bins]) / len(correct))

    # High vs low confidence accuracy
    median = float(np.median(confidences))
    high_mask = confidences >= median
    low_mask = confidences < median

    results = {
        "expected_calibration_error": round(ece, 4),
        "high_confidence_threshold": round(median, 4),
        "high_confidence_accuracy": round(float(correct[high_mask].mean()), 4),
        "low_confidence_accuracy": round(float(correct[low_mask].mean()), 4),
        "high_confidence_count": int(high_mask.sum()),
        "low_confidence_count": int(low_mask.sum()),
        "bins": bins,
        "interpretation": (
            "Higher-confidence predictions are more accurate than low-confidence ones, "
            "so the model's confidence score is a useful (though not perfect) signal of reliability."
            if correct[high_mask].mean() > correct[low_mask].mean()
            else "Confidence does not clearly track accuracy on this small test set."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(results, indent=2))

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(7, 6))
    bin_accs = [r["accuracy"] for r in bins]
    bin_confs = [r["mean_confidence"] for r in bins]
    bin_counts = [r["count"] for r in bins]

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
    ax.scatter(bin_confs, bin_accs, s=[c * 20 for c in bin_counts], zorder=3, label="Model bins")
    for conf, acc, row in zip(bin_confs, bin_accs, bins):
        ax.annotate(f"n={row['count']}", (conf, acc), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction correct (accuracy)")
    ax.set_title("Confidence Calibration — Fine-Tuned DistilBERT (Test Set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()

    print(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT_JSON} and {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
