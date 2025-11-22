# Use a imagem base oficial do Airflow (ajuste a versão conforme seu cluster)
FROM apache/airflow:2.7.1

# Mude para o usuário root para instalar dependências de sistema (se necessário)
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         gcc \
         g++ \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Volte para o usuário airflow para instalar pacotes Python
USER airflow

# Instale as bibliotecas que seu modelo precisa APENAS UMA VEZ aqui
RUN pip install --no-cache-dir \
    mlflow \
    scikit-learn \
    numpy \
    pandas