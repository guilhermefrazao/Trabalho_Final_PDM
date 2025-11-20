from fastapi import FastAPI
import requests
from datetime import datetime, timedelta
import random
import pytz
import time

app = FastAPI()

AIRFLOW_API = "http://airflow-webserver.airflow.svc.cluster.local:8080/api/v2"
DAG_ID = "exemplo_pipeline_completo"

async def wait_for_dag_result(dag_id, dag_run_id, headers):
    while True:
        resp = requests.get(
            f"{AIRFLOW_API}/dags/{dag_id}/dagRuns/{dag_run_id}",
            headers=headers
        )
        data = resp.json()
        state = data.get("state")
        print("Estado atual:", state)

        if state in ("success", "failed"):
            return data  # terminou

        time.sleep(5)


@app.get("/")
async def home():
    return {"status": "FastAPI testando o novo CI com o push no github"}

@app.post("/executar_dag")
async def chat(question: str):
    """
    Recebe a pergunta, dispara um DAG no Airflow e retorna a resposta.
    """

    url_token = "http://airflow-webserver.airflow.svc.cluster.local:8080/auth/token"

    data = {
    "username": "airflow",
    "password": "airflow"
    }


    token = requests.post(url=url_token, headers={"Content-Type": "application/json"}, json=data)

    if token.status_code == 201:
        data = token.json()
        token = data.get("access_token") or data.get("token") or data.get("jwt")  # o campo exato pode variar conforme versão
        print("Token obtido:\n", token)
    else:
        print("Falha ao obter token:", token.status_code, token.text, "\n")

    dag_run_url = f"{AIRFLOW_API}/dags/{DAG_ID}/dagRuns"

    random_id = random.randint(0, 100)

    local_tz = pytz.timezone("America/Sao_Paulo")
    now_local = local_tz.localize(datetime.now())
    two_hours_before = now_local - timedelta(hours=3)
    logical_date = two_hours_before.astimezone(pytz.UTC).isoformat()


    data_dag = {
    "dag_run_id": f"testando_funcionamento_dag_via_requests_{str(random_id)}",
    "logical_date": logical_date,  # obrigatório
    "conf": {"question": "teste"},
    "note": "Disparo via FastAPI"
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    resp = requests.post(
    url=dag_run_url,
    headers=headers,
    json=data_dag
    )

    if resp.status_code == 200:
        data = resp.json()
        print("resp obtido:\n", data)
    else:
        print("Falha ao obter resp:\n", resp.status_code, resp.text)

    data_response = await wait_for_dag_result(DAG_ID, data_dag["dag_run_id"], headers=headers)

    return {
        "mensagem": f"DAG '{DAG_ID}' executando...",    
        "dados": {
            "question": question,
            "response": data_response
        }
    }
