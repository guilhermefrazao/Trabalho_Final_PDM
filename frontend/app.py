from fastapi import FastAPI, Request, HTTPException
import requests
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import random
import pytz
import time
import mlflow.pyfunc
import os
import pandas as pd
from google.cloud import bigquery
from typing import Dict, Any, Optional
from pydantic import BaseModel

from bigquery_queries import (
    set_bigquery_client,
    get_filmes_por_ano,
    get_duracao_filme,
    get_genero_filme,
    get_nota_filme,
    get_votos_filme,
    get_media_genero,
    get_votos_genero,
    get_filmes_por_genero,
    get_filmes_por_duracao,
    get_filmes_por_nota,
    get_filmes_por_votos,
    get_atores_filme,
    get_filmes_ator
)


# --- Variáveis de Ambiente e Configuração ---
AIRFLOW_API = "http://airflow-webserver.airflow.svc.cluster.local:8080/api/v1"
DAG_ID = "executar_treinamento_k8s"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service.default.svc.cluster.local")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASS", "admin")

BIGQUERY_PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID", "ornate-shape-471913-t7")


# --- Variáveis Globais de Modelos ---
ml_models: Dict[str, Any] = {}

def detect_drift() -> bool:
    """
    Função simples para detectar drift de dados.
    Retorna True se drift for detectado, False caso contrário.
    """
    drift_detected = False
    if drift_detected:
        print("Drift de dados detectado!")
    else:
        print("Nenhum drift de dados detectado.")
    return drift_detected

# --- Funções Auxiliares (Airflow) ---

async def wait_for_dag_result(dag_id, dag_run_id):
    while True:
        resp = requests.get(
            f"{AIRFLOW_API}/dags/{dag_id}/dagRuns/{dag_run_id}",
            auth=(AIRFLOW_USER, AIRFLOW_PASS)
        )
        data = resp.json()
        state = data.get("state")
        print("Estado atual:", state)

        if state in ("success", "failed"):
            return data
        time.sleep(5)


# --- Função Lifespan (Inicialização da API e Clientes) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Esta função roda APENAS UMA VEZ quando o servidor sobe.
    Carrega o modelo MLflow e inicializa o cliente BigQuery.
    """
    
    print("Inicializando API e carregando dependências...")
    
    # 1. Inicialização do BigQuery Client
    try:
        bq_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
        set_bigquery_client(bq_client) 
        print(f"Cliente BigQuery inicializado. Projeto: {bq_client.project}")
    except Exception as e:
        print(f"Erro CRÍTICO ao inicializar o BigQuery: {e}")
        
    # 2. Carregamento do Modelo MLflow 
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model("models:/model_movies_intention/Production")
        ml_models["model"] = model
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro CRÍTICO ao carregar modelo: {e}")
        
    yield
    
    ml_models.clear()
    set_bigquery_client(None)
    print("API desligando.")


app = FastAPI(lifespan=lifespan)


# --- Definição do Payload ---
class FrazoPayload(BaseModel):
    query: str = ""
    intent: dict  # {"name": "..."}
    entities: list  # [{"entity": "GENERO", "value": "Ação"}]


# --- Função Auxiliar para Extrair Entidades ---
def extrair_entidade(entities: list, nome_entidade: str):
    """Extrai o valor de uma entidade específica da lista de entidades."""
    return next((e['value'] for e in entities if e['entity'] == nome_entidade), None)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def home():
    return {"status": "FastAPI testando o novo CI com o push no github"}


@app.post("/retrain_dag")
async def retrain_dag():
    """
    Recebe a pergunta, dispara um DAG no Airflow e retorna a resposta.
    """
    dag_run_url = f"{AIRFLOW_API}/dags/{DAG_ID}/dagRuns"

    random_id = random.randint(0, 100)

    local_tz = pytz.timezone("America/Sao_Paulo")
    now_local = local_tz.localize(datetime.now())
    two_hours_before = now_local - timedelta(hours=3)
    logical_date = two_hours_before.astimezone(pytz.UTC).isoformat()

    data_dag = {
        "dag_run_id": f"testando_funcionamento_dag_via_requests_{str(random_id)}",
        "logical_date": logical_date,
        "conf": {"question": "teste"},
        "note": "Disparo via FastAPI"
    }

    resp = requests.post(
        url=dag_run_url,
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
        json=data_dag
    )

    if resp.status_code == 200:
        data = resp.json()
        print("resp obtido:\n", data)
    else:
        print("Falha ao obter resp:\n", resp.status_code, resp.text)

    data_response = await wait_for_dag_result(DAG_ID, data_dag["dag_run_id"])

    return {
        "mensagem": f"DAG '{DAG_ID}' executando...",
        "dados": {
            "response": data_response
        }
    }


@app.post("/chat")
def predict_question(question: str):
    model = ml_models.get("linear_model")
    
    if not model:
        return {"error": "Modelo de predição não carregado."}
        
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    print("Usando modelo carregado do MLflow...")
    
    detect_drift()

    if detect_drift() == True:
        dag_run_url = f"{AIRFLOW_API}/dags/{DAG_ID}/dagRuns"

        random_id = random.randint(0, 100)

        local_tz = pytz.timezone("America/Sao_Paulo")
        now_local = local_tz.localize(datetime.now())
        two_hours_before = now_local - timedelta(hours=3)
        logical_date = two_hours_before.astimezone(pytz.UTC).isoformat()

        data_dag = {
        "dag_run_id": f"testando_funcionamento_dag_via_requests_{str(random_id)}",
        "logical_date": logical_date,
        "conf": {"question": "teste"},
        "note": "Disparo via FastAPI"
        }

        resp = requests.post(
            url=dag_run_url,
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            json=data_dag
        )

        return {"pergunta": question, "resposta": "Foi detectado drift de dados. O modelo está sendo re-treinado."}

    else:
        resposta = "Usando modelo sem drift de dados."
        
        return {"pergunta": question, "resposta": resposta}