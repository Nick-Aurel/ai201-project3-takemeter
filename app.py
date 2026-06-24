"""
Simple web interface for TakeMeter classification.
Run: python app.py
Then open http://127.0.0.1:7860
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("model")


def load_classifier():
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError("Model not found. Run: python train_and_evaluate.py")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


TOKENIZER, MODEL = load_classifier()


def classify_post(text: str) -> tuple[str, dict[str, float]]:
    if not text.strip():
        return "Enter a post to classify.", {}

    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = MODEL(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))
    label = MODEL.config.id2label[pred_id]
    confidence = float(probs[pred_id])
    prob_dict = {MODEL.config.id2label[i]: float(probs[i]) for i in range(len(probs))}

    summary = f"**{label}** ({confidence:.1%} confidence)"
    return summary, prob_dict


demo = gr.Interface(
    fn=classify_post,
    inputs=gr.Textbox(
        label="r/soccer post",
        placeholder="Paste a post here, e.g. Messi is the GOAT, no debate",
        lines=4,
    ),
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Label(label="Class probabilities", num_top_classes=3),
    ],
    title="TakeMeter — r/soccer Discourse Classifier",
    description=(
        "Fine-tuned DistilBERT classifier. Labels: **analysis**, **hot_take**, **reaction**."
    ),
    examples=[
        ["Messi is the GOAT, no debate"],
        ["What a finish, right into the top corner"],
        [
            "Looking at Haaland's goal-to-shot ratio this season (0.47) compared to his "
            "historical average (0.44), the improvement looks sustainable rather than lucky."
        ],
    ],
)

if __name__ == "__main__":
    demo.launch()
