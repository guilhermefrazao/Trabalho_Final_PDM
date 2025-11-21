data "terraform_remote_state" "infra" {
  backend = "local"

  config = {
    path = "../infra_tf/terraform.tfstate"
  }
}


data "google_client_config" "default" {}


data "google_container_cluster" "cluster_airflow" {
  name     = google_container_cluster.primary.name
  location = google_container_cluster.primary.location

  depends_on = [google_container_node_pool.primary_nodes]
}

provider "kubernetes" {
  host                   = "https://${data.google_container_cluster.cluster_airflow.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(data.google_container_cluster.cluster_airflow.master_auth[0].cluster_ca_certificate)
}

provider "helm" {
  kubernetes = {
    host                   = "https://${data.google_container_cluster.cluster_airflow.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(data.google_container_cluster.cluster_airflow.master_auth[0].cluster_ca_certificate)
  }
}

resource "helm_release" "airflow" {
  name             = "airflow"
  repository       = "https://airflow.apache.org"
  chart            = "airflow"
  namespace        = "airflow"
  create_namespace = true
  timeout          = 600

  values = [
    <<EOF
executor: KubernetesExecutor
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
EOF
  ]

  depends_on = [google_container_node_pool.primary_nodes]
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
          image = "us-central1-docker.pkg.dev/${var.project}/${google_artifact_registry_repository.repo.name}/fastapi-app:${var.image_tag}"
          name  = "fastapi"
          port {
            container_port = 80
          }
          env {
            name  = "AIRFLOW_HOST"
            value = "airflow-webserver.airflow.svc.cluster.local:8080"
          }
        }
      }
    }
  }
  depends_on = [google_container_node_pool.primary_nodes]
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
      target_port = 80
    }
    type = "LoadBalancer"
  }
}