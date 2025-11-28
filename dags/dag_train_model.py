from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import logging 
import os
from datetime import datetime

from kubernetes.client import models as k8s
from pinhas_model.train_mdeberta import run_training_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = "/tmp/modelo_treinado_v3"


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

    try:
        model, tokenizer, acc, f1_int, f1_ner = run_training_pipeline(epochs=1)

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("train_model")

    except Exception as e:
        logger.error(f"Erro durante o treinamento do modelo: {e}")
        raise

    try:
        logger.info("Iniciando o log do modelo no MLflow.")
    
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

    except Exception as e:
        logger.error(f"Erro durante o log do modelo no MLflow: {e}")
        raise



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
        python_callable = treinar_modelo,
        executor_config={
        "pod_override": k8s.V1Pod(
            spec=k8s.V1PodSpec(
                containers=[
                    k8s.V1Container(
                        name="base",
                        resources=k8s.V1ResourceRequirements(
                            requests={"memory": "4Gi", "cpu": "2000m"},
                            limits={"memory": "8Gi", "cpu": "4000m"} 
                        )
                    )
                ]
            )
        )
    }
    )

    tarefa_final = BashOperator(
        task_id="fim",
        bash_command="echo 'Finalizado treinamento do modelo com sucesso!'"
    )


    tarefa_inicial >> treinamento >> tarefa_final
