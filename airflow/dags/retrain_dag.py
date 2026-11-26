import json
from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator

PROJECT = "/workspaces/reco-engine"
PY = f"{PROJECT}/.venv/bin/python"          # the project's own environment (TensorFlow, Spark, FAISS)

def decide():
    m = json.load(open(f"{PROJECT}/eval/metrics.json"))
    return "promote" if m["new_ndcg"] > m["old_ndcg"] * 1.01 else "skip"    # needs a 1% improvement

def sh(task_id, script):
    return BashOperator(task_id=task_id, bash_command=f"cd {PROJECT} && {PY} {script}")

with DAG(dag_id="reco_retrain", schedule="@weekly", start_date=datetime(2026, 1, 1),
         catchup=False, tags=["recommender"]) as dag:
    interactions = sh("spark_interactions", "spark_jobs/build_interactions.py")
    features     = sh("spark_features",     "spark_jobs/build_features.py")
    retrieval    = sh("train_retrieval",    "retrieval/two_tower.py")
    faiss_       = sh("build_faiss",        "retrieval/build_faiss.py")
    ranker_data  = sh("ranker_data",        "ranking/build_ranker_data.py")
    ranker       = sh("train_ranker",       "ranking/train_ranker.py")
    evaluate     = sh("evaluate",           "eval/evaluate.py")
    gate         = BranchPythonOperator(task_id="gate", python_callable=decide)
    promote      = sh("promote",            "serving/promote.py")
    skip         = EmptyOperator(task_id="skip")

    (interactions >> features >> retrieval >> faiss_ >> ranker_data
     >> ranker >> evaluate >> gate >> [promote, skip])