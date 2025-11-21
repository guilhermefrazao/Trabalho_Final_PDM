
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
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "airflow-node"
  location   = google_container_cluster.primary.location
  cluster    = google_container_cluster.primary.name
  node_count = 2

  node_config {
    machine_type = "e2-standard-2"
    disk_size_gb = 50
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}


