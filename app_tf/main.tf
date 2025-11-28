data "terraform_remote_state" "infra" {
  backend = "local"

  config = {
    path = "../infra_tf/terraform.tfstate"
  }
}


data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${data.terraform_remote_state.infra.outputs.cluster_endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(data.terraform_remote_state.infra.outputs.cluster_ca_certificate)
}

provider "helm" {
  kubernetes = {
    host                   = "https://${data.terraform_remote_state.infra.outputs.cluster_endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(data.terraform_remote_state.infra.outputs.cluster_ca_certificate)
  }
}


resource "kubernetes_service_account" "ml_app_sa" {
  metadata {
    name      = "ml-app-sa"
    namespace = "default" 
    annotations = {
      "iam.gke.io/gcp-service-account" = data.terraform_remote_state.infra.outputs.service_account_email
    }
  }
}

resource "helm_release" "airflow" {
  name             = "airflow"
  repository       = "https://airflow.apache.org"
  chart            = "airflow"
  namespace        = "airflow"
  create_namespace = true
  timeout          = 600
  wait             = false
  version          = "1.11.0"

  values = [
    <<EOF

defaultAirflowRepository: "us-central1-docker.pkg.dev/${var.project}/${data.terraform_remote_state.infra.outputs.repo_name}/airflow"
defaultAirflowTag: "${var.image_tag_airflow}"

# Política de pull (Always garante que ele pegue a versão nova se a tag for a mesma)
images:
  airflow:
    pullPolicy: Always

executor: KubernetesExecutor

apiServer:
  enabled: false

postgresql:
  enabled: true
  image:
    repository: postgres
    tag: 13-alpine  
    pullPolicy: IfNotPresent
dags:
  gitSync:
    enabled: true
    repo: "https://github.com/guilhermefrazao/Trabalho_Final_PDM.git"
    branch: "feat/infra"
    subPath: "dags"

webserver:
    service:
        type: LoadBalancer
    startupProbe:
        failureThreshold: 60 
        periodSeconds: 10

    resources:
        requests:
          cpu: 2000m
          memory: 1024Mi

serviceAccount:
  create: true
  name: "airflow-sa" # Tem que bater com o nome usado no Binding acima
  annotations:
    iam.gke.io/gcp-service-account: "${data.terraform_remote_state.infra.outputs.service_account_email}"

env:
  - name: "AIRFLOW__API__AUTH_BACKENDS"
    value: "airflow.api.auth.backend.basic_auth"
EOF
  ]
}


resource "kubernetes_deployment" "fastapi" {
  metadata {
    name = "fastapi-app"
    labels = {
      app = "fastapi"
    }
  }

  spec {
    replicas = 2
    selector {
      match_labels = {
        app = "fastapi"
      }
    }
    template {
      metadata {
        labels = {
          app = "fastapi"
        }
      }
      spec {
        service_account_name = kubernetes_service_account.ml_app_sa.metadata[0].name

        container {
          image = "us-central1-docker.pkg.dev/${var.project}/${data.terraform_remote_state.infra.outputs.repo_name}/fastapi-app:${var.image_tag_fastapi}"
          name  = "fastapi"
          port {
            container_port = 80
          }
          env {
            name  = "AIRFLOW_HOST"
            value = "airflow-webserver.airflow.svc.cluster.local:8000"
          }
          env {
            name  = "MLFLOW_TRACKING_URI"
            value = "http://mlflow-service.default.svc.cluster.local"
          }
        }
      }
    }
  }
}


resource "kubernetes_service" "fastapi_service" {
  metadata {
    name = "fastapi-service"
  }
  spec {
    selector = {
      app = "fastapi"
    }
    port {
      port        = 80
      target_port = 8000
    }
    type = "LoadBalancer"
  }
}

resource "kubernetes_deployment" "model_ml_flow" {
  metadata {
    name = "mlflow-app"
    labels = {
      app = "mlflow"
    }
  }

  spec {
    replicas = 1
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
        service_account_name = kubernetes_service_account.ml_app_sa.metadata[0].name

        container {
          image = "us-central1-docker.pkg.dev/${var.project}/${data.terraform_remote_state.infra.outputs.repo_name}/ml-flow:${var.image_tag_mlflow}"
          name  = "mlflow"
          port {
            container_port = 5000
          }

        env {
          name  = "AIRFLOW_HOST"
          value = "airflow-webserver.airflow.svc.cluster.local:5000"
        }
        args = [
          "mlflow", "server",
          "--host", "0.0.0.0",
          "--port", "5000",
          "--backend-store-uri", "sqlite:///mlflow.db",
          "--default-artifact-root", "gs://${data.terraform_remote_state.infra.outputs.bucket_name}/mlruns",
          "--allowed-hosts", "*"
        ]
      }
      }
    }
  }
}

resource "kubernetes_service" "mlflow_service" {
  metadata {
    name = "mlflow-service"
  }
  spec {
    selector = {
      app = "mlflow"
    }
    port {
      port        = 80
      target_port = 5000
    }
    type = "LoadBalancer"
  }
}


