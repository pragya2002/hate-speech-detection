# Automated Hate Speech Detection and Moderation Reporting System

**CS-GY 6513 Big Data | Spring 2026 | Section D | Prof. Amit Patel**
**New York University Tandon School of Engineering**

| Name | NetID | Role |
|------|-------|------|
| Aditya Jha | aj4955 | ML Infrastructure and Pipeline Orchestration |
| Pragya Awasthi | pa2755 | NLP Preprocessing and Analytics |
| Tharun Murugesan | tm4717 | Reporting, Performance Evaluation and RoBERTa Fine-Tuning |

---

## Overview

A fully automated, distributed hate speech detection and moderation system built on Apache Spark and Hadoop. The system ingests 2,009,376 social media comments across 4 datasets, trains six multi-label toxicity classifiers via Spark MLlib on a 5-node GCP YARN cluster, and generates automated HTML moderation reports — all orchestrated by Apache Airflow. A fine-tuned RoBERTa model serves as the production inference backend, integrated with a live Telegram moderation bot.

### Live Deployments

| Component | URL |
|-----------|-----|
| RoBERTa API (HuggingFace Spaces) | https://thisisadi-hate-speech-detection.hf.space |
| TF-IDF API (Render) | https://hate-speech-api-lqrg.onrender.com |
| RoBERTa Model Weights | https://huggingface.co/thisisadi/hate-speech-roberta |

---

## Repository Structure

```
hate-speech-detection/
│
├── api/
│   ├── bot.py                          # Telegram moderation bot (works with both APIs)
│   ├── roberta/
│   │   ├── app.py                      # Flask API serving RoBERTa from HuggingFace Hub
│   │   ├── Dockerfile                  # Docker config for HuggingFace Spaces deployment
│   │   └── requirements.txt
│   └── tf-idf/
│       ├── app.py                      # Flask API serving TF-IDF weights via pure NumPy
│       ├── Dockerfile                  # Docker config for Render deployment
│       ├── model_weights.json          # Extracted TF-IDF + LogReg weights (1.4MB)
│       ├── stopwords.json              # Stop word list used during training
│       ├── render.yaml                 # Render deployment config
│       └── requirements.txt
│
├── ml-pipeline/                        # Core Spark training pipeline
│   ├── ingestion/                      # Dataset download and HDFS upload scripts
│   ├── models/                         # TF-IDF + LogReg training, cross-validation
│   ├── orchestration/                  # Airflow DAG definition
│   ├── reporting/                      # HTML report and chart generation
│   └── benchmarking/                   # Scalability benchmark at 25/50/100% scale
│
├── scripts/                            # Standalone PySpark scripts
│   ├── 05-user-aggregation.py          # Spark SQL per-user toxicity aggregation
│   ├── 06-generate-moderation-report.py
│   └── 07-predict.py                   # Batch predictions on full 2M dataset
│
├── docs/                               # Benchmark charts and experiment outputs
├── tests/                              # Unit and integration tests
├── model_metadata.json                 # Single-dataset model training metadata
├── multilabel_metadata.json            # Multi-label model metadata
├── multilabel_combined_metadata.json   # Combined 4-dataset model metadata
└── .gitignore
```

---

## Prerequisites

- NYU Dataproc cluster (Hadoop 3.3.6, Spark 3.5.3, YARN)
- Python 3.11

```bash
pip install pyspark matplotlib jinja2 apache-airflow==2.8.1 requests python-telegram-bot
```

- Kaggle API credentials (`~/.kaggle/kaggle.json`)

---

## Data Setup

Download all four datasets and upload to HDFS:

```bash
# Jigsaw Toxic Comment Classification
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge

# Jigsaw Unintended Bias in Toxicity Classification (Civil Comments)
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification

# Twitter Hate Speech
kaggle datasets download -d mrmorj/hate-speech-and-offensive-language-dataset

# HateXplain
git clone https://github.com/hate-alert/HateXplain

# Upload all to HDFS
hadoop fs -mkdir -p /user/$USER/hatespeech/data
hadoop fs -put *.csv /user/$USER/hatespeech/data/
```

| Dataset | Rows | Source |
|---------|------|--------|
| Jigsaw Toxic Comment Classification | 159,571 | Kaggle |
| Jigsaw Unintended Bias / Civil Comments | 1,804,874 | Kaggle |
| Twitter Hate Speech (mrmorj) | 24,783 | Kaggle |
| HateXplain | 20,148 | GitHub |
| **Combined** | **2,009,376** | — |

---

## Running the Pipeline

### Manual (step by step)

```bash
# 1. Harmonize all 4 datasets into unified 6-label schema
spark-submit ml-pipeline/ingestion/harmonize-datasets.py

# 2. Train all 6 multi-label classifiers on 2M rows
spark-submit --num-executors 4 --executor-memory 4g --executor-cores 2 \
  ml-pipeline/models/multilabel-train-combined.py

# 3. Run batch predictions on the full dataset
spark-submit scripts/07-predict.py

# 4. Aggregate per-user toxicity statistics
spark-submit scripts/05-user-aggregation.py

# 5. Generate HTML moderation report and PNG charts
spark-submit scripts/06-generate-moderation-report.py

# 6. Run scalability benchmark (25 / 50 / 100% of data)
spark-submit ml-pipeline/benchmarking/run-benchmark.py

# 7. Classify a single comment interactively
spark-submit ml-pipeline/predict-comment.py "your comment here"
```

### Via Airflow (automated daily at midnight)

```bash
airflow db init
cp ml-pipeline/orchestration/hatespeech-pipeline.py ~/airflow/dags/
airflow scheduler &
airflow webserver --port 8080
```

The DAG `hatespeech-detection-pipeline` runs 5 tasks in sequence:

```
ingest-data → run-predictions → aggregate-user-behavior → generate-report → notify-completion
```

---

## ML Pipeline: How It Works

### 1. Data Harmonization
Reads all 4 source datasets and maps their heterogeneous label schemas into a unified binary 6-column schema: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. Outputs Parquet partitioned by `toxic` label to HDFS.

### 2. Model Training
For each of the 6 toxicity labels, trains an independent Spark MLlib Pipeline:

`RegexTokenizer` → `StopWordsRemover` → `HashingTF (10,000 features)` → `IDF` → `LogisticRegression`

- Inverse class frequency weighting handles label imbalance (severe_toxic: 0.08% of data)
- Hyperparameters selected by 3-fold cross-validation: `regParam=0.01`, `elasticNetParam=0.5`
- Models saved as PipelineModel artifacts to HDFS (~50MB each)

### 3. Batch Prediction
Loads all 6 models and applies them sequentially to the full 2M-row dataset. Custom thresholds applied per label to control false positives.

| Label | AUC-ROC | Threshold |
|-------|---------|-----------|
| toxic | 0.857 | 50% |
| severe_toxic | 0.928 | 80% |
| obscene | 0.916 | 50% |
| threat | 0.919 | 70% |
| insult | 0.884 | 50% |
| identity_hate | 0.929 | 65% |

### 4. Report Generation
Generates a standalone HTML moderation report with embedded PNG charts using Matplotlib and Jinja2: category distribution bar chart, toxic/clean donut chart, high-risk user heatmap, and volume trend.

---

## Serving: Two-Model Architecture

The system uses two serving backends depending on the use case:

### TF-IDF API (`api/tf-idf/`) — deployed on Render
Weights extracted from the trained Spark models into `model_weights.json` (1.4MB). Served via pure NumPy — no JVM, no Spark. Runs on 128MB RAM with <10ms latency.

```bash
cd api/tf-idf
pip install -r requirements.txt
python app.py
```

### RoBERTa API (`api/roberta/`) — deployed on HuggingFace Spaces
RoBERTa-base fine-tuned on the same 2,009,376-row dataset for 2 epochs on an A100 GPU. Mean AUC-ROC: 0.9855 vs 0.906 for TF-IDF. Runs as a Docker container on HuggingFace Spaces.

```bash
cd api/roberta
pip install -r requirements.txt
python app.py
```

### API Usage (same interface for both)

```bash
# Health check
curl https://thisisadi-hate-speech-detection.hf.space/

# Predict
curl -X POST https://thisisadi-hate-speech-detection.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"comment": "your text here"}'
```

Response:
```json
{
  "comment": "your text here",
  "is_toxic": true,
  "flagged_categories": ["toxic", "insult"],
  "scores": {
    "toxic":         {"probability": 94.3, "flagged": true,  "threshold": 50.0},
    "severe_toxic":  {"probability": 12.1, "flagged": false, "threshold": 80.0},
    "obscene":       {"probability": 31.4, "flagged": false, "threshold": 50.0},
    "threat":        {"probability":  2.0, "flagged": false, "threshold": 70.0},
    "insult":        {"probability": 87.6, "flagged": true,  "threshold": 50.0},
    "identity_hate": {"probability":  8.3, "flagged": false, "threshold": 65.0}
  }
}
```

---

## Telegram Moderation Bot (`api/bot.py`)

The bot works with either API — set `HATE_SPEECH_API_URL` to switch between them.

```bash
# Using RoBERTa on HuggingFace (default)
TELEGRAM_BOT_TOKEN="your_token" \
HATE_SPEECH_API_URL="https://thisisadi-hate-speech-detection.hf.space" \
python api/bot.py

# Using TF-IDF on Render
TELEGRAM_BOT_TOKEN="your_token" \
HATE_SPEECH_API_URL="https://hate-speech-api-lqrg.onrender.com" \
python api/bot.py
```

**Behavior:**
- Monitors all group messages silently
- Deletes toxic messages and issues a warning (1/3, 2/3, 3/3)
- Mutes the user for 30 minutes after 3 warnings
- Re-adding a removed user clears their warning count and mute state
- `/warnings` — check your current warning count
- `/about` — model and project info
- **Known limitation:** Group admins and owners cannot be muted (Telegram API restriction)

---

## Scalability Results

| Scale | Rows | Total Time | Throughput |
|-------|------|------------|------------|
| 25% | 503,835 | 138.83s | 3,629 rows/s |
| 50% | 1,005,093 | 152.20s | 6,603 rows/s |
| 100% | 2,009,376 | 291.33s | 6,897 rows/s |

**Sub-linear scaling:** 4x data → only 2.1x time increase.
**Parquet vs CSV:** 3.6x faster reads; filtered reads (toxic=1 only) are 11x faster.

---

## Dataset Links

- Jigsaw Toxic Comment: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- Jigsaw Unintended Bias: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
- Twitter Hate Speech: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
- HateXplain: https://github.com/hate-alert/HateXplain