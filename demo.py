"""
Classify sample r/soccer posts with the fine-tuned DistilBERT model.
Run after training: python demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("model")
MODEL_NAME = "distilbert-base-uncased"
LABEL_NAMES = ["analysis", "hot_take", "reaction"]

SAMPLE_POSTS = [
    {
        "text": "Messi is the GOAT, no debate",
        "expected": "hot_take",
        "note": "Classic one-line assertion with no supporting evidence.",
    },
    {
        "text": (
            "Looking at Haaland's goal-to-shot ratio this season (0.47) compared to his "
            "historical average (0.44), combined with City's increased volume in the box, "
            "the improvement looks sustainable rather than lucky."
        ),
        "expected": "analysis",
        "note": "Uses specific stats and causal reasoning.",
    },
    {
        "text": "What a finish, right into the top corner",
        "expected": "reaction",
        "note": "Immediate reaction to a single moment.",
    },
    {
        "text": "Barcelona will never be relevant again",
        "expected": "hot_take",
        "note": "Sweeping claim without evidence.",
    },
    {
        "text": "That was one of the most entertaining matches I've seen in years",
        "expected": "reaction",
        "note": "Emotional match-level impression, not a structured argument.",
    },
]


def load_model():
    if (MODEL_DIR / "config.json").exists():
        model_path = MODEL_DIR
    else:
        checkpoint_dirs = sorted(
            MODEL_DIR.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if not checkpoint_dirs:
            raise SystemExit("Model not found. Run python train_and_evaluate.py first.")
        model_path = checkpoint_dirs[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    id2label = model.config.id2label
    return tokenizer, model, id2label


def classify(text: str, tokenizer, model, id2label) -> tuple[str, float, dict[str, float]]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))
    label = id2label[pred_id]
    confidence = float(probs[pred_id])
    all_probs = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    return label, confidence, all_probs


def main() -> None:
    if not MODEL_DIR.exists():
        raise SystemExit("Model not found. Run python train_and_evaluate.py first.")

    tokenizer, model, id2label = load_model()
    results = []

    print("TakeMeter Demo — Fine-Tuned DistilBERT Classifier\n" + "=" * 60)
    for i, sample in enumerate(SAMPLE_POSTS, start=1):
        label, confidence, all_probs = classify(sample["text"], tokenizer, model, id2label)
        correct = label == sample["expected"]
        status = "CORRECT" if correct else "INCORRECT"

        print(f"\nPost {i}: {status}")
        print(f"Text: {sample['text'][:120]}{'...' if len(sample['text']) > 120 else ''}")
        print(f"Predicted: {label} ({confidence:.1%} confidence)")
        print(f"Expected:  {sample['expected']}")
        print(f"All probs: {', '.join(f'{k}={v:.1%}' for k, v in all_probs.items())}")
        print(f"Note: {sample['note']}")

        results.append(
            {
                "text": sample["text"],
                "predicted_label": label,
                "confidence": round(confidence, 4),
                "expected_label": sample["expected"],
                "correct": correct,
                "probabilities": {k: round(v, 4) for k, v in all_probs.items()},
            }
        )

    Path("demo_results.json").write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 60)
    print("Saved demo output to demo_results.json")


if __name__ == "__main__":
    main()
