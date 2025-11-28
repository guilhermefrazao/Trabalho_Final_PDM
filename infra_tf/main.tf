
terraform {
  required_providers {
    google     = { source = "hashicorp/google", version = ">= 4.0" }
    helm       = { source = "hashicorp/helm", version = ">= 2.0" }
    kubernetes = { source = "hashicorp/kubernetes", version = ">= 2.0" }
  }
}

provider "google" {
  project = var.project
  region  = var.region
}

resource "google_project_service" "gcp_services" {
  for_each = toset(var.gcp_service_list)

  project            = var.project
  service            = each.key
  disable_on_destroy = false
}

resource "google_service_account" "Pdm-2025-creditos" {
  account_id   = "pdm-2025-creditos"
  display_name = "Service Account para o Cloud Run FastAPI"
}

resource "time_sleep" "wait_seconds" {
  create_duration = "30s"

  depends_on = [google_project_service.gcp_services]
}

resource "google_artifact_registry_repository" "repo" {
  repository_id = "airflow-fastapi-repo"
  format        = "DOCKER"
  location      = "us-central1"

  depends_on = [time_sleep.wait_seconds]
}


resource "google_storage_bucket" "mlflow_bucket" {
  name          = "${var.project}-mlflow-artifacts"
  location      = var.region
  force_destroy = true

  storage_class = "STANDARD"

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }
}


resource "google_storage_bucket_iam_member" "gke_access" {
  bucket = google_storage_bucket.mlflow_bucket.name
  role   = "roles/storage.objectAdmin"

  member = "serviceAccount:${google_service_account.Pdm-2025-creditos.email}"
}

resource "google_project_iam_member" "sa_artifact_registry_access" {
  project = var.project
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.Pdm-2025-creditos.email}"
}

resource "google_project_iam_member" "cloudbuild_registry" {
  project = var.project
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.Pdm-2025-creditos.email}"
}

resource "google_project_iam_member" "cloudbuild_run_admin" {
  project = var.project
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.Pdm-2025-creditos.email}"
}

resource "google_project_iam_member" "cloudbuild_sa_user" {
  project = var.project
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.Pdm-2025-creditos.email}"
}

resource "google_project_iam_member" "cloudbuild_logs_writer" {
  project = var.project
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.Pdm-2025-creditos.email}"
}

resource "google_cloudbuild_trigger" "build_trigger" {
  name = "trigger-airflow-deploy"

  service_account = google_service_account.Pdm-2025-creditos.id

  github {
    owner = "guilhermefrazao"
    name  = "Trabalho_Final_PDM"
    push {
      branch = "^feat/infra$"
    }
  }
  filename = "cloudbuild.yaml"

  substitutions = {
    _REGION    = var.region
    _REPO_NAME = google_artifact_registry_repository.repo.name
  }

  depends_on = [
    google_project_iam_member.cloudbuild_run_admin,
    google_project_iam_member.cloudbuild_sa_user
  ]
}

resource "google_container_cluster" "primary" {
  name     = "fastapi-airflow-cluster"
  location = "us-central1-a"

  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {
    workload_pool = "${var.project}.svc.id.goog"
  }
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "airflow-node" 
  location   = google_container_cluster.primary.location
  cluster    = google_container_cluster.primary.name
  
  initial_node_count = 1

  autoscaling {
    min_node_count = 1  # Mantém 1 sempre ligado (pra não demorar a iniciar)
    max_node_count = 5  # Pode crescer até 5 máquinas se o deploy for pesado
  }

  node_config {
    spot = true 

    machine_type = "e2-standard-4"


    disk_type    = "pd-balanced" 
    disk_size_gb = 50 # 50GB SSD é mais rápido que 100GB HDD para boot

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = {
      "cloud.google.com/gke-spot" = "true"
      "role"                      = "worker"
    }
  }
}



resource "google_service_account_iam_member" "workload_identity_airflow" {
  service_account_id = google_service_account.Pdm-2025-creditos.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project}.svc.id.goog[airflow/airflow-sa]"


  depends_on = [
    google_container_cluster.primary,
    google_container_node_pool.primary_nodes
  ]
}

resource "google_service_account_iam_member" "workload_identity_mlflow" {
  service_account_id = google_service_account.Pdm-2025-creditos.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project}.svc.id.goog[default/ml-app-sa]"

  depends_on = [
    google_container_cluster.primary,
    google_container_node_pool.primary_nodes
  ]
}


resource "null_resource" "configure_kubectl" {

  triggers = {
    cluster_endpoint = google_container_cluster.primary.endpoint
  }

  provisioner "local-exec" {
    command = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --zone ${google_container_cluster.primary.location} --project ${var.project}"

  }

  depends_on = [google_container_cluster.primary]
}