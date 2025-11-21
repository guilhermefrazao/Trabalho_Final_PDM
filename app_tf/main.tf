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

resource "helm_release" "airflow" {
  name             = "airflow"
  repository       = "https://airflow.apache.org"
  chart            = "airflow"
  namespace        = "airflow"
  create_namespace = true
  timeout          = 600
  wait = false
  version          = "1.11.0"

  values = [
    <<EOF
defaultAirflowTag: "2.10.3"

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

