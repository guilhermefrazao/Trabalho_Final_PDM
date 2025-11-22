FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install fastapi requests uvicorn pytz mlflow pandas

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

