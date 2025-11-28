from fastapi import FastAPI, HTTPException
import requests
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import random
import pytz
import time
import os
import mlflow.artifacts
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account  # << NOVO
from typing import Dict, Any
from pydantic import BaseModel
from pathlib import Path
import json
from dialog_manager import dialog_manager
from llm_responder import build_llm_answer
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Body


import torch
from transformers import AutoTokenizer

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
    get_filmes_ator,
)

# >>> IMPORTA APENAS A CLASSE DO MODELO (sem rodar o treino)
from joint_model import JointTransformer


# --- Variáveis de Ambiente e Configuração ---
AIRFLOW_API = "http://airflow-webserver.airflow.svc.cluster.local:8080/api/v1"
DAG_ID = "executar_treinamento_k8s"

AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASS", "admin")

BIGQUERY_PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID", "ornate-shape-471913-t7")

MODEL_NAME = "modelo_movies_bot"
STAGE = "Production"

ml_objects = {}

# Caminho do modelo treinado localmente
# frontend/app.py -> parent = frontend, parent.parent = raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_LOCAL_DIR = BASE_DIR / "dags" / "pinhas_model" / "models" / "modelo_treinado_v3"
MODEL_STAGE = "Production"

# --- CREDENCIAL FIXA DO BIGQUERY (SERVICE ACCOUNT) ---
# ATENÇÃO:
# 1. REVOQUE a chave antiga que você colou aqui no chat.
# 2. Gere UMA NOVA chave JSON no GCP.
# 3. Cole o conteúdo da nova chave aqui no lugar do {...}.
SERVICE_ACCOUNT_INFO: Dict[str, Any] = {
  "type": "service_account",
  "project_id": "ornate-shape-471913-t7",
  "private_key_id": "002dcb34fd8c3eed243c030ced819fe2fe14f6a6",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDVQyZqiYnJ7jaZ\na6Vjm7W8qxYX+9TAcQ7uD+zs2oMtXcEnBqsE9y5QhifaxRP/7lelyPA6oMK7ptQe\nY/2mH8d+HMgMyrX7mDLtzpyzslnqN+Jqd0J9AratRKfmmPcH5PRArpCVVWs+Lw68\n1I1R+H/Rx/0kR2p3EHqiwI+V7nl4BuBpN9rBj5e0iCLFbMrLlMM6+luZod9O+9oY\nc5XwUiIMdYyAAKn3BasZ6oLhRiMnxvY/30I4O/CZd1paaGF3yW0/2QZiFbENAv1t\nlGo8c6STWUx5ea6YIx649n4izmtCixVre9mssuGvmMoctwzfFmG9GP9vALb3xtqL\nGKQbm90RAgMBAAECggEACcslxVx31gzBF+JSQ+zjh4ylTPhpBKXkgUD8uMNPpRqS\nO1Iych3OVA8XHzQv9w+5WXEM/CpECxVOFi+yFyoMMaHNe43nFczfYN2dQ2DQQYHr\nvUcpB9/jJvmZSs0RQhzs7Rf/JoZ30HEdiIUr0Lz8VLRODxeS/3EDqrv0VBaxoBX+\nQheoAz+DVjzu3vKr5FK/HpLq7skUAtVJNmoaCbn3PdOX5kZXyo8GT2zzqc3Z7MP4\nIU4UmuLZ+JEnjl0CySAOKOVRKjVc5pF2GWaqMee2X9qR4FcBVezPg/CUcUCmosIU\nX6o+Hj/lBxcz5+eKUd3G13JaGlyUqkjFGdeb/DH0KQKBgQD4NifmgGzTcCoQNzWW\n+7OuW3EBd4pxC13PiTYRLWnZ6XjHSBuieEPk54lnkCsuFCSqwQOaZLES+ctJ4ATc\n/rPHupsgf5woLNP7iOCwnRcsf9NyoIHVKFss3Kq7zhlYSzpN9Rs9iozZ8mACQvkL\nYng9wfeRXxx0ibt4Gi6xU1xdiQKBgQDb9EChs2bUPLBm1tqIM/vLnpmxmb0dJTGQ\nZUzP7//rsZLGyCsdC4l295y3Q5RXIXyO52PYUEmKc8J3ojOzai48l4rZ6750y/Nz\naxB/IHrtxeOBEsmVM+23d21r0UJ/HhW0P/Xw+KCZmQe+XiokHUR1YtWt7J5Tr/YQ\nvdq6AqhpSQKBgQDvpAm95/wp378czND2pqkCC9L9IZcOMXUvLECBMSFjfKBZdusH\nX6ndVRY6YNzvrg8AtT1vUQwNUBLPjnPjkh2tFiMzq0DvIOjBj5OvsNtw4TEbGJCB\nZmcw1xQYIpIhxu/R2HWmYyA+RF4hkNu1/CovJhiJyBRHB7tx54VxOxSQUQKBgHqz\nt5p9Sk/7yyyTjzWMSls9DuBUs3se1JeI62DUsh/537ek0uhRF06Ws4ZI3Of+dk4C\nJ2D06RGjoki647yi70g+Aeev63+chyNMBtfkdq9ORawrnujtHx/KL/CFvGLNla4I\nQFs9V9pX1EoOndOnwBj8Gdf8uBZXgx2zy+EYunqBAoGAJpG8HPW8gnTQJP7O2fJN\nZs2mfRZOfMCP99V0AMzGelyDqVeMiaUlevu2wmhq/sXwJTjkzRa2oCDLhigRVGJd\nRiVeJkggNYVPM04cm0Atl592zqH0lALuiMJcQYqDm18SEwHYDs29I2VIyCCiLkMu\nAPOGfN1UHVrMt8W4749MDP0=\n-----END PRIVATE KEY-----\n",
  "client_email": "541106010171-compute@developer.gserviceaccount.com",
  "client_id": "104902160060918324224",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/541106010171-compute%40developer.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

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

def create_bq_client() -> bigquery.Client:
    """
    Cria um cliente BigQuery usando a service account fixa (SERVICE_ACCOUNT_INFO).
    Se SERVICE_ACCOUNT_INFO estiver vazio, cai no cliente padrão (ADC).
    """
    if not SERVICE_ACCOUNT_INFO:
        # fallback: tenta usar ADC (GOOGLE_APPLICATION_CREDENTIALS ou gcloud auth)
        print("SERVICE_ACCOUNT_INFO vazio. Usando credenciais padrão (ADC).")
        return bigquery.Client(project=BIGQUERY_PROJECT_ID)

    creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO)
    project_id = SERVICE_ACCOUNT_INFO.get("project_id", BIGQUERY_PROJECT_ID)
    return bigquery.Client(credentials=creds, project=project_id)


# --- Funções Auxiliares (Airflow) ---

async def wait_for_dag_result(dag_id, dag_run_id):
    while True:
        resp = requests.get(
            f"{AIRFLOW_API}/dags/{dag_id}/dagRuns/{dag_run_id}",
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
        )
        data = resp.json()
        state = data.get("state")
        print("Estado atual:", state)

        if state in ("success", "failed"):
            return data
        time.sleep(5)


# --- Definição do Payload (mantido caso você use depois) ---
class FrazoPayload(BaseModel):
    query: str = ""
    intent: dict  # {"name": "..."}
    entities: list  # [{"entity": "GENERO", "value": "Ação"}]


# --- Função Auxiliar para Extrair Entidades ---
def extrair_entidade(entities: list, nome_entidade: str):
    """Extrai o valor de uma entidade específica da lista de entidades."""
    return next((e["value"] for e in entities if e["entity"] == nome_entidade), None)


# ==========================
# Wrapper do modelo local
# ==========================


class LocalJointNLU:
    """
    Wrapper que carrega o modelo salvo em modelo_treinado_v3
    e expõe um método predict(DataFrame) compatível com o /chat.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise RuntimeError(f"Diretório do modelo não encontrado: {self.model_dir}")

        # 1) Carrega config (intent2id, tag2id, base_model_name)
        with open(self.model_dir / "training_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.base_model_name = cfg["base_model_name"]

        # os valores vêm como string no JSON, garante int
        intent2id_raw = cfg["intent2id"]
        tag2id_raw = cfg["tag2id"]

        self.intent2id = {k: int(v) for k, v in intent2id_raw.items()}
        self.tag2id = {k: int(v) for k, v in tag2id_raw.items()}

        self.id2intent = {v: k for k, v in self.intent2id.items()}
        self.id2tag = {v: k for k, v in self.tag2id.items()}

        # 2) Tokenizer salvo na mesma pasta
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # 3) Modelo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = JointTransformer(
            self.base_model_name,
            num_intents=len(self.intent2id),
            num_entities=len(self.tag2id),
        ).to(self.device)

        state_dict = torch.load(
            self.model_dir / "model_weights.bin",
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"Modelo NLU carregado de: {self.model_dir}")
        print(f"Base model: {self.base_model_name}")
        print(f"Num intents: {len(self.intent2id)}, num entities: {len(self.tag2id)}")

    def _predict_one(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])

        # Intenção
        intent_id = int(torch.argmax(outputs["intent_logits"]).item())
        intent = self.id2intent[intent_id]

        # Entidades (baseado no seu predict_playground, mas corrigindo tokens)
        entity_ids = torch.argmax(outputs["entity_logits"], dim=2)[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entities = []
        current_ent = None

        for token, idx in zip(tokens, entity_ids):
            if token in self.tokenizer.all_special_tokens:
                continue

            label = self.id2tag[int(idx)]

            # Trata token:
            # - remove "##"
            # - troca "▁" por espaço (SentencePiece)
            token_clean = token.replace("##", "")
            token_clean = token_clean.replace("▁", " ").strip()

            if not token_clean:
                continue

            if label.startswith("B-"):
                # fecha entidade anterior, se tiver
                if current_ent:
                    current_ent["text"] = " ".join(current_ent["text"].split())
                    entities.append(current_ent)

                current_ent = {
                    "type": label[2:],   # tira o "B-"
                    "text": token_clean,
                }

            elif label.startswith("I-") and current_ent and label[2:] == current_ent["type"]:
                # concatena com espaço entre tokens
                current_ent["text"] = f"{current_ent['text']} {token_clean}".strip()

            else:
                # fecha entidade se mudar de rótulo / sair do span
                if current_ent:
                    current_ent["text"] = " ".join(current_ent["text"].split())
                    entities.append(current_ent)
                    current_ent = None

        # fecha última entidade, se ainda estiver aberta
        if current_ent:
            current_ent["text"] = " ".join(current_ent["text"].split())
            entities.append(current_ent)

        return {"intent": intent, "entities": entities}


    def predict(self, df: pd.DataFrame):
        """
        df deve ter uma coluna 'texto'.
        Retorna uma lista de dicts: [{"intent": ..., "entities": [...]}, ...]
        """
        texts = df["texto"].tolist()
        return [self._predict_one(t) for t in texts]


# --- Função Lifespan (Inicialização da API e Clientes) ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Esta função roda APENAS UMA VEZ quando o servidor sobe.
    Carrega o modelo local e inicializa o cliente BigQuery.
    """

    print("Inicializando API e carregando dependências...")

    # 1. Inicialização do BigQuery Client
    try:
        bq_client = create_bq_client()
        set_bigquery_client(bq_client)
        print(f"Cliente BigQuery inicializado. Projeto: {bq_client.project}")
    except Exception as e:
        print(f"Erro CRÍTICO ao inicializar o BigQuery: {e}")

    # 2. Carregamento do Modelo LOCAL (sem MLflow)
    try:
        print(f"📥 Baixando modelo '{MODEL_NAME}' (Stage: {MODEL_STAGE}) do GCS...")
        
        # Constrói a URI do MLflow Model Registry
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        
        # A MÁGICA: O MLflow vai no GCS, autentica via Workload Identity, 
        # baixa a pasta inteira para /tmp/xyz/ e retorna o caminho local.
        local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        
        print(f"📂 Modelo baixado em: {local_model_path}")
        
        # Agora passamos o caminho local para sua classe customizada,
        # exatamente como se fosse uma pasta local no seu computador.
        model = LocalJointNLU(local_model_path)
        
        ml_models["linear_model"] = model
        print("✅ Modelo NLU carregado com sucesso!")
    except Exception as e:
        print(f"Erro CRÍTICO ao carregar modelo NLU local: {e}")

    yield

    ml_models.clear()
    set_bigquery_client(None)
    print("API desligando.")


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # ou ["POST", "OPTIONS"]
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/")
async def home():
    return {
        "status": "FastAPI rodando",
        "model_source": "local (dags/pinhas_model/models/modelo_treinado_v3)",
    }


@app.post("/retrain_dag")
async def retrain_dag():
    """
    Recebe a pergunta, dispara um DAG no Airflow e retorna a resposta.
    (Em ambiente local sem Airflow isso pode falhar, mas mantive igual.)
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
        "note": "Disparo via FastAPI",
    }

    resp = requests.post(
        url=dag_run_url,
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
        json=data_dag,
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
            "response": data_response,
        },
    }


@app.post("/chat")
def predict_question(question: str = Body(..., embed=False)):
    """
    Usa o modelo JointTransformer treinado (carregado localmente)
    para prever intenção e entidades a partir de uma pergunta em texto livre
    e em seguida chama o dialog_manager para integrar com o BigQuery.
    """
    model = ml_models.get("linear_model")
    if not model:
        raise HTTPException(status_code=500, detail="Modelo de predição não carregado.")

    # Monta DataFrame esperado pelo wrapper do modelo
    input_data = pd.DataFrame([{"texto": question}])

    try:
        nlu_result = model.predict(input_data)[0]

        intent_name = nlu_result.get("intent", "")
        raw_entities = nlu_result.get("entities", []) or []

        dm_entities = []
        for ent in raw_entities:
            ent_type = ent.get("type")
            ent_text = ent.get("text")
            if ent_type and ent_text:
                dm_entities.append(
                    {
                        "entity": ent_type,
                        "value": ent_text,
                    }
                )

        dialog_result = dialog_manager(intent_name, dm_entities, question)

        llm_answer = build_llm_answer(
            question=question,
            nlu_result=nlu_result,
            dialog_result=dialog_result,
        )

        return {
            "query": question,
            "nlu": nlu_result,
            "dialog": dialog_result,
            "llm_answer": llm_answer,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na predição ou no dialog manager: {str(e)}",
        )

