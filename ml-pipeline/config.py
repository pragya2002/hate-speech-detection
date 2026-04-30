"""Shared configuration for the hate-speech detection pipeline.

Source of truth for paths, label order, flag thresholds, and high-risk
user rules. Scripts should import from here rather than redefining.
"""

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"
LOCAL_BASE = "/home/aj4955_nyu_edu/hatespeech_data"

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# severe_toxic / threat raised: rare-class probabilities cluster near ~55%.
# identity_hate raised: reduces bias against identity mentions.
THRESHOLDS = {
    "toxic":         0.50,
    "severe_toxic":  0.80,
    "obscene":       0.50,
    "threat":        0.70,
    "insult":        0.50,
    "identity_hate": 0.65,
}

HIGH_RISK_TOXICITY_RATIO = 0.15
HIGH_RISK_MIN_COMMENTS = 50
