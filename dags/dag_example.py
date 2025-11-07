# ===============================================
# Exemplo completo de um DAG no Apache Airflow
# ===============================================

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from frontend.app import run

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
    dag_id="exemplo_pipeline_completo",    
    description="Exemplo de DAG com Python e BashOperator",
    default_args=default_args,
    start_date=datetime(2025, 11, 7),        # primeira data de execução
    schedule="*/1 * * * *",        # executa a cada 10 minutos
    catchup=False,                           # não roda execuções antigas
    tags=["exemplo", "tutorial"],            # tags para filtro no UI
) as dag:

    
    def processar_dados():
        print("Processando dados...")

    # Tarefa 1: apenas imprime algo no console
    tarefa_inicial = BashOperator(
        task_id="inicio",
        bash_command="echo 'Iniciando pipeline...'"
    )  

    # Tarefa 2: executa uma função Python
    tarefa_python = PythonOperator(
        task_id="processar",
        python_callable=processar_dados
    )

    # Tarefa 3: simula finalização
    tarefa_final = BashOperator(
        task_id="fim",
        bash_command="echo 'Pipeline concluída com sucesso!'"
    )


    tarefa_inicial >> tarefa_python >> tarefa_final
