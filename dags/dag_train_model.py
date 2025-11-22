from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import logging 
from datetime import datetime


MLFLOW_URI = "http://mlflow-service.default.svc.cluster.local:5000"

def treinar_modelo():
    import logging
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LinearRegression
    import numpy as np

    logger = logging.getLogger("airflow.task")
    
    logger.info("Iniciando a função de treinamento do modelo.")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("exemplo-registro-modelo")
    
    logger.info("Preparando dados de treino...")
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    logger.info("Instanciando e treinando LinearRegression...")
    model = LinearRegression()
    model.fit(X, y)
    logger.info("Treinamento concluído.")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Iniciando MLflow Run ID: {run_id}")

        mlflow.sklearn.log_model(model, artifact_path="modelo")
        logger.info("Modelo logado no MLflow.")

        mlflow.log_metric("rmse", 0.0)
        logger.info("Métricas logadas.")
        
        model_uri = f"runs:/{run_id}/modelo"
        
        logger.info(f"Registrando modelo com URI: {model_uri}")
        mv = mlflow.register_model(model_uri=model_uri, name="modelo_linear_teste")
        
        client = mlflow.tracking.MlflowClient()
        logger.info(f"Transicionando versão {mv.version} para Production...")
        
        client.transition_model_version_stage(
            name="modelo_linear_teste",
            version=mv.version,
            stage="Production"
        )
        logger.info(f"SUCESSO: Modelo versão {mv.version} promovido para Production.")


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
        python_callable = treinar_modelo     
    )

    tarefa_final = BashOperator(
        task_id="fim",
        bash_command="echo 'Finalizado treinamento do modelo com sucesso!'"
    )


    tarefa_inicial >> treinamento >> tarefa_final
