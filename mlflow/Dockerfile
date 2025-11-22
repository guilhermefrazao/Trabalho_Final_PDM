# Use uma imagem Python leve
FROM python:3.10-slim

RUN pip install mlflow google-cloud-storage psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0"]