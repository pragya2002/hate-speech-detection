import os
import re
import json
import struct
import logging
import numpy as np
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
THRESHOLDS = {
    "toxic":         0.50,
    "severe_toxic":  0.80,
    "obscene":       0.50,
    "threat":        0.70,
    "insult":        0.50,
    "identity_hate": 0.65,
}
NUM_FEATURES = 10000
BASE_DIR = os.path.dirname(__file__)

# Common profanity that L1 regularization zeroed out but should always be flagged
PROFANITY = {
    'fuck', 'fucking', 'fucked', 'fucker', 'fucks',
    'shit', 'shitting', 'bullshit',
    'bitch', 'bitches', 'bitching',
    'cunt', 'cunts',
    'asshole', 'assholes', 'ass',
    'bastard', 'bastards',
    'whore', 'whores',
    'slut', 'sluts',
    'dick', 'dicks', 'cock', 'cocks',
    'pussy', 'pussies',
    'faggot', 'faggots', 'fag',
    'retard', 'retarded',
    'nigger', 'nigga',
    'motherfucker', 'motherfucking',
    'cocksucker', 'dipshit', 'dumbass',
    'jackass', 'moron', 'imbecile',
    'prick', 'twat', 'wanker',
}

logger.info("Loading model weights...")
with open(os.path.join(BASE_DIR, "model_weights.json")) as f:
    raw = json.load(f)

WEIGHTS = {
    label: {
        "coef":      np.array(w["coef"],      dtype=np.float64),
        "idf":       np.array(w["idf"],       dtype=np.float64),
        "intercept": float(w["intercept"]),
    }
    for label, w in raw.items()
}

with open(os.path.join(BASE_DIR, "stopwords.json")) as f:
    STOP_WORDS = set(json.load(f))

logger.info(f"Loaded {len(WEIGHTS)} models. API ready.")


def murmur3_32(data: str, seed: int = 42) -> int:
    """Spark-compatible MurmurHash3 (Guava impl, seed=42)."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    length = len(data)
    h = seed
    nblocks = length // 4
    for block in range(nblocks):
        k = struct.unpack_from('<i', data, block * 4)[0]
        k = (k * 0xcc9e2d51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1b873593) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF
    tail = data[nblocks * 4:]
    k = 0
    tail_size = length & 3
    if tail_size >= 3: k ^= tail[2] << 16
    if tail_size >= 2: k ^= tail[1] << 8
    if tail_size >= 1:
        k ^= tail[0]
        k = (k * 0xcc9e2d51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1b873593) & 0xFFFFFFFF
        h ^= k
    h ^= length
    h ^= (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= (h >> 16)
    if h >= 0x80000000:
        h -= 0x100000000
    return h


def hash_token(token: str) -> int:
    return abs(murmur3_32(token)) % NUM_FEATURES


def hashing_tf(tokens: list) -> np.ndarray:
    freq = np.zeros(NUM_FEATURES, dtype=np.float64)
    for t in tokens:
        freq[hash_token(t)] += 1.0
    return freq


def preprocess(text: str) -> list:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.split(r"\W+", text)
    return [t for t in tokens if t and t not in STOP_WORDS]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def check_profanity(text: str) -> list:
    """Returns list of profanity-triggered categories."""
    words = set(re.split(r'\W+', text.lower()))
    hits = words & PROFANITY
    if hits:
        return ["toxic", "obscene", "insult"]
    return []


def predict(comment: str) -> dict:
    tokens = preprocess(comment)
    tf = hashing_tf(tokens)

    scores = {}
    flagged = set()

    # ML model predictions
    for label in LABELS:
        w = WEIGHTS[label]
        tfidf = tf * w["idf"]
        raw_score = float(np.dot(w["coef"], tfidf)) + w["intercept"]
        prob = sigmoid(raw_score)
        threshold = THRESHOLDS[label]
        is_flagged = bool(prob >= threshold)
        scores[label] = {
            "probability": round(prob * 100, 2),
            "threshold":   threshold * 100,
            "flagged":     is_flagged,
        }
        if is_flagged:
            flagged.add(label)

    # Profanity override — catches words L1 regularization zeroed out
    profanity_flags = check_profanity(comment)
    for label in profanity_flags:
        flagged.add(label)
        if not scores[label]["flagged"]:
            scores[label]["flagged"] = True
            scores[label]["flagged_by"] = "profanity_filter"

    flagged_list = [l for l in LABELS if l in flagged]

    return {
        "comment":            comment,
        "cleaned":            " ".join(tokens),
        "is_toxic":           bool(len(flagged_list) > 0),
        "flagged_categories": flagged_list,
        "scores":             scores,
    }


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "service":       "Hate Speech Detection API",
        "models_loaded": LABELS,
        "version":       "2.2.0",
        "backend":       "numpy + Spark MurmurHash3 + profanity filter",
    })


@app.route("/predict", methods=["POST"])
def predict_post():
    data = request.get_json()
    if not data or "comment" not in data:
        return jsonify({"error": "Missing 'comment' field"}), 400
    comment = data["comment"].strip()
    if not comment:
        return jsonify({"error": "Comment cannot be empty"}), 400
    if len(comment) > 5000:
        return jsonify({"error": "Comment exceeds 5000 character limit"}), 400
    return jsonify(predict(comment))


@app.route("/predict", methods=["GET"])
def predict_get():
    comment = request.args.get("comment", "").strip()
    if not comment:
        return jsonify({"error": "Missing 'comment' query parameter"}), 400
    return jsonify(predict(comment))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
