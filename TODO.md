#TODO: Ativar a api do GKE utilizando o terraform.
#TODO: criar o cluster manualmente
#TODO: criar o cluster automáticamente e adicionar o docker-compose dentro dele.

args = [
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--backend-store-uri", "sqlite:///mlflow.db", 
            "--default-artifact-root", "gs://${var.bucket_name}/mlruns" 
          ]

          env {
            name  = "GOOGLE_APPLICATION_CREDENTIALS"
            value = "/var/secrets/google/key.json" # Exemplo, ajuste se usar Workload Identity
          }