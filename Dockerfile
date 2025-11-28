# 1. Imagem base
FROM apache/airflow:2.10.3

# 2. Instalação de pacotes do SO (como root)
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Volta para o usuário airflow
USER airflow

# 4. Copia o requirements.txt para a pasta atual (geralmente /opt/airflow)
# É melhor copiar para "." do que para a raiz "/"
COPY requirements.txt requirements.txt

# 5. Instala as dependências Python
# O --no-cache-dir é ótimo para manter a imagem leve
RUN pip install --no-cache-dir -r requirements.txt