import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

try:
    dados_val = load_data('dags\pinhas_model\data\dataset_v3_val.json')
    dados_train = load_data('dags\pinhas_model\data\dataset_v3_train.json')
except FileNotFoundError:
    print("Arquivos não encontrados. Certifique-se de que os .json estão na pasta.")
    dados_val = [] 
    dados_train = []

dataset_unificado = dados_train + dados_val

textos_referencia = [item['text'] for item in dataset_unificado]

print(f"Total de frases na base de conhecimento: {len(textos_referencia)}")


print("Carregando modelo de embedding...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Gerando embeddings de referência...")
embeddings_referencia = model.encode(textos_referencia)


def verificar_drift(novo_texto, threshold=0.5):
    """
    Recebe uma frase, gera o embedding e compara com a base.
    Se a similaridade máxima for menor que o threshold, é uma anomalia.
    """
    # 1. Gera embedding do input
    novo_embedding = model.encode([novo_texto])
    
    # 2. Calcula similaridade de cosseno contra TODAS as frases do treino
    # (Retorna uma matriz de scores entre 0 e 1)
    similaridades = cosine_similarity(novo_embedding, embeddings_referencia)
    
    # 3. Pega a maior similaridade encontrada (o vizinho mais próximo)
    max_score = np.max(similaridades)
    
    # 4. Lógica de decisão
    # Se max_score for baixo, significa que a frase não se parece com NADA que o modelo viu.
    houve_drift = max_score < threshold
    
    resultado = {
        "input": novo_texto,
        "drift_detectado": houve_drift,
        "score_similaridade": round(float(max_score), 4),
        "mensagem": " ALERTA: Frase fora do domínio!" if houve_drift else " Dentro do domínio."
    }
    
    return resultado

def testar_frases(lista):
    resultados = [verificar_drift(t) for t in lista]
    df = pd.DataFrame(resultados)
    print(df)

testar_frases([
    "Qual a nota do filme Batman?",
    "Gostaria de pedir uma pizza de calabresa",
    "receita de bolo de cenoura",
    "Mainha quer saber qual a nota do filme Batman",
    "Gostaria de pedir uma pipoca",
    "Quais filmes foram estrelados por Christian Bale",
    "Quero assistir um filme com Christian Bale no elenco",
    "Quais filmes o Brad Pitt participou recentemente?",
    "Me recomende filmes de ficção científica com nota alta",
    "Onde fica o posto de gasolina mais próximo?",
    "A vida é como um filme de drama sem roteiro"
])