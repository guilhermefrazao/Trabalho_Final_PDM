#TODO: Ativar a api do GKE utilizando o terraform.
#TODO: criar o cluster manualmente
#TODO: criar o cluster automáticamente e adicionar o docker-compose dentro dele.

resource "kubernetes_deployement" "model_ml_flow" {
    metadata {
        name = "mlflow-app"
        labels = {
        app = "mlflow"
        }
    }

  spec {
    replicas = 2
    selector {
      match_labels = {
        app = "mlflow"
      }
    }
    template {
      metadata {
        labels = {
          app = "mlflow"
        }
      }
      spec {
        container {
          image = "us-central1-docker.pkg.dev/${var.project}/${data.terraform_remote_state.infra.outputs.repo_name}/mlflow-app:${var.image_tag_mlflow}"
          name  = "mlflow"
          port {
            container_port = 80
          }
          env {
            name  = "AIRFLOW_HOST"
            value = "airflow-webserver.airflow.svc.cluster.local:5000"
          }
        }
      }
    }
  }
}