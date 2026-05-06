import os
import logging
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

LABELS = ["toxic", "severe_toxic", "obscene",
          "threat", "insult", "identity_hate"]
THRESHOLDS = {l: 0.5 for l in LABELS}
MODEL_REPO = "thisisadi/hate-speech-roberta"

logger.info("Loading tokenizer from roberta-base...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

logger.info(f"Loading model weights from {MODEL_REPO}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_REPO,
)
model.eval()
logger.info("Model ready.")


def predict(comment: str) -> dict:
    inputs = tokenizer(
        comment, return_tensors="pt",
        truncation=True, max_length=128, padding=True
    )
    with torch.no_grad():
        probs = torch.sigmoid(
            model(**inputs).logits.float()
        ).numpy()[0]

    scores = {}
    flagged = []
    for i, label in enumerate(LABELS):
        prob = float(probs[i])
        is_flagged = bool(prob >= THRESHOLDS[label])
        scores[label] = {
            "probability": round(prob * 100, 2),
            "threshold":   THRESHOLDS[label] * 100,
            "flagged":     is_flagged,
        }
        if is_flagged:
            flagged.append(label)

    return {
        "comment":            comment,
        "is_toxic":           bool(len(flagged) > 0),
        "flagged_categories": flagged,
        "scores":             scores,
    }


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "model":   MODEL_REPO,
        "version": "3.0.0",
        "backend": "RoBERTa fine-tuned on 2M comments (float16)",
    })


@app.route("/predict", methods=["POST"])
def predict_post():
    data = request.get_json()
    if not data or "comment" not in data:
        return jsonify({"error": "Missing comment"}), 400
    comment = data["comment"].strip()
    if not comment:
        return jsonify({"error": "Empty comment"}), 400
    if len(comment) > 5000:
        return jsonify({"error": "Comment exceeds 5000 character limit"}), 400
    return jsonify(predict(comment))


@app.route("/predict", methods=["GET"])
def predict_get():
    comment = request.args.get("comment", "").strip()
    if not comment:
        return jsonify({"error": "Missing comment"}), 400
    return jsonify(predict(comment))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
