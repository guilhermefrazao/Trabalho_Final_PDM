from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import logging 
import os
from datetime import datetime

from pinhas_model.train_mdeberta import run_training_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, 'pinhas_model', 'models', 'modelo_treinado_v3')


MLFLOW_URI = "http://mlflow-service.default.svc.cluster.local"

def treinar_modelo():
    import logging
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    from sklearn.linear_model import LinearRegression
    import numpy as np

    logger = logging.getLogger("airflow.task")
    
    logger.info("Iniciando a função de treinamento do modelo.")

    model, tokenizer, acc, f1_int, f1_ner = run_training_pipeline(epochs=1)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("train_model")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Iniciando MLflow Run ID: {run_id}")

        mlflow.log_artifacts(local_dir=OUTPUT_DIR, artifact_path="model")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_intent", f1_int)
        mlflow.log_metric("f1_entity", f1_ner)

        logger.info(f"Métricas logadas: Acc={acc}, F1_Int={f1_int}")

        logger.info("Modelo logado no MLflow.")

        model_uri = f"runs:/{run.info.run_id}/model"

        mlflow.register_model(model_uri, "modelo_movies_bot")

        logger.info(f"Registrando modelo com URI: {model_uri}")



default_args = {
    "owner": "airflow",                 # dono do DAG
    "depends_on_past": False,           # não depende de execuções anteriores
    "email": ["guilhermefrazao@discente.ufg.br"],   # para notificações
    "email_on_failure": False,          # desativa alertas de falha
    "email_on_retry": False,
    "retries": 1,                       # número de tentativas em caso de erro
    "retry_delay": timedelta(minutes=0.1) # intervalo entre tentativas
}


with DAG(
    dag_id="executar_treinamento_k8s",    
    description="Dag de treinamento do modelo que está no cluster kubernetes",
    schedule=None,
    default_args=default_args,
    start_date=datetime(2023, 11, 7),        # primeira data de execução
    catchup=False,                           # não roda execuções antigas
    tags=["exemplo", "tutorial"],            # tags para filtro no UI
) as dag:


    tarefa_inicial = BashOperator(
        task_id="inicio",
        bash_command="echo 'Iniciando treinamento do modelo...'"
    )  

    treinamento = PythonOperator(
        task_id="execute_training_with_mlflow",
        python_callable = treinar_modelo    ,
        execution_timeout=datetime.timedelta(minutes=5) 
    )

    tarefa_final = BashOperator(
        task_id="fim",
        bash_command="echo 'Finalizado treinamento do modelo com sucesso!'"
    )


    tarefa_inicial >> treinamento >> tarefa_final
