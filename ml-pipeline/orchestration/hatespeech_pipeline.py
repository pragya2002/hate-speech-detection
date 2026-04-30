from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner": "aj4955",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

SPARK_SUBMIT = """
spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-memory 4g \
    --executor-cores 2 \
    --num-executors 4 \
    {script}
"""

SCRIPTS_BASE = "/home/aj4955_nyu_edu/hatespeech-bigdata"

# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="hatespeech_detection_pipeline",
    description="Automated hate speech detection and moderation reporting pipeline",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval="0 0 * * *",  # daily at midnight
    catchup=False,
    tags=["bigdata", "nlp", "hate-speech"],
) as dag:

    # ── Task 1: Ingest and partition data on HDFS ─────────────────────────────
    ingest = BashOperator(
        task_id="ingest_data",
        bash_command=SPARK_SUBMIT.format(
            script=f"{SCRIPTS_BASE}/ml-pipeline/ingestion/ingest.py"
        ),
        execution_timeout=timedelta(minutes=15),
    )

    # ── Task 2: Apply NLP preprocessing + run predictions ────────────────────
    predict = BashOperator(
        task_id="run_predictions",
        bash_command=SPARK_SUBMIT.format(
            script=f"{SCRIPTS_BASE}/scripts/07_predict.py"
        ),
        execution_timeout=timedelta(minutes=15),
    )

    # ── Task 3: Aggregate user behavior via Spark SQL ─────────────────────────
    aggregate = BashOperator(
        task_id="aggregate_user_behavior",
        bash_command=SPARK_SUBMIT.format(
            script=f"{SCRIPTS_BASE}/scripts/05_user_aggregation.py"
        ),
        execution_timeout=timedelta(minutes=15),
    )

    # ── Task 4: Generate moderation report ────────────────────────────────────
    # Reads HDFS prediction outputs, applies per-label thresholds, writes
    # summary / category_breakdown / flagged_comments / high_risk_users
    # under outputs/reports/<run_date>/. Replaces the old local-mode
    # scripts/06_generate_moderation_report.py (kept in repo for reference).
    report = BashOperator(
        task_id="generate_report",
        bash_command=SPARK_SUBMIT.format(
            script=f"{SCRIPTS_BASE}/ml-pipeline/reporting/generate_report.py"
        ),
        execution_timeout=timedelta(minutes=15),
    )

    # ── Task 5: Log pipeline completion ───────────────────────────────────────
    notify = BashOperator(
        task_id="notify_completion",
        bash_command="""
            echo "Pipeline completed at $(date)" >> \
            /home/aj4955_nyu_edu/hatespeech_data/logs/pipeline_runs.log
        """,
    )

    # ── Dependencies: ingest → predict → aggregate → report → notify ──────────
    ingest >> predict >> aggregate >> report >> notify
