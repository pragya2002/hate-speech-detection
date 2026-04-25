from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

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

SCRIPTS_BASE = "/home/aj4955_nyu_edu/hatespeech_data"

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

    # ── Task 1: Ingest data from HDFS ─────────────────────────────────────────
    ingest = BashOperator(
        task_id="ingest_data",
        bash_command=SPARK_SUBMIT.format(
            script=f"{SCRIPTS_BASE}/ingest.py"
        ),
        execution_timeout=timedelta(minutes=15),
    )

    # ── Task 2: Run TF-IDF + prediction using saved final model ───────────────
    predict = BashOperator(
        task_id="run_predictions",
        bash_command=SPARK_SUBMIT.format(
            script=f"{SCRIPTS_BASE}/predict.py"
        ),
        execution_timeout=timedelta(minutes=15),
    )

    # ── Task 3: Aggregate user behavior via Spark SQL ─────────────────────────
    aggregate = BashOperator(
        task_id="aggregate_user_behavior",
        bash_command=SPARK_SUBMIT.format(
            script=f"{SCRIPTS_BASE}/aggregate.py"
        ),
        execution_timeout=timedelta(minutes=15),
    )

    # ── Task 4: Generate moderation report ────────────────────────────────────
    report = BashOperator(
        task_id="generate_report",
        bash_command=SPARK_SUBMIT.format(
            script=f"{SCRIPTS_BASE}/report.py"
        ),
        execution_timeout=timedelta(minutes=15),
    )

    # ── Task 5: Notify pipeline completion ────────────────────────────────────
    notify = BashOperator(
        task_id="notify_completion",
        bash_command="""
            echo "Pipeline completed at $(date)" >> \
            /home/aj4955_nyu_edu/hatespeech_data/logs/pipeline_runs.log
        """,
    )

    # ── Dependencies: ingest → predict → aggregate → report → notify ──────────
    ingest >> predict >> aggregate >> report >> notify
