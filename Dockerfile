# Use a mesma versão que você definiu no Terraform (defaultAirflowTag)
FROM apache/airflow:2.10.3

# Troca para usuário root para instalar dependências do sistema se necessário (opcional)
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         build-essential \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Volta para o usuário airflow para instalar pacotes Python (Prática de segurança)
USER airflow

# Instala as bibliotecas que seu modelo precisa
# google-cloud-storage: para salvar o modelo no bucket
# mlflow: para registrar o modelo
# scikit-learn, numpy, pandas: para o treinamento
RUN pip install --no-cache-dir \
    mlflow \
    google-cloud-storage \
    scikit-learn \
    numpy \
    pandas

env:
  - name: "_PIP_ADDITIONAL_REQUIREMENTS"
    value: "mlflow google-cloud-storage scikit-learn numpy pandas"