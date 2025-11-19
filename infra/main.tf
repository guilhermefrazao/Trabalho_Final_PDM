import {
  to = google_service_account.Pdm-2025-creditos

  id = "projects/pdm-2025-creditos/serviceAccounts/pdm-2025-creditos@pdm-2025-creditos.iam.gserviceaccount.com"
}

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
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
  repository_id = "fastapi-repo-triggers"
  format        = "DOCKER"
  location      = "us-central1"

  depends_on = [time_sleep.wait_seconds]
}


resource "google_cloud_run_v2_service" "fastapi_service" {
  name       = "fastapi-service"
  location   = "us-central1"
  depends_on = [google_project_service.gcp_services]

  template {
    service_account = google_service_account.Pdm-2025-creditos.email

    containers {
      image = "us-central1-docker.pkg.dev/${var.project}/fastapi-repo/fastapi-app:v4"
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

resource "google_cloud_run_service_iam_member" "public_invoker" {
  location = google_cloud_run_v2_service.fastapi_service.location
  project  = var.project
  service  = google_cloud_run_v2_service.fastapi_service.name

  role   = "roles/run.invoker"
  member = "allUsers"
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

resource "google_cloudbuild_trigger" "build_trigger" {
  name = "trigger-fastapi-deploy"

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
    _REGION       = var.region
    _SERVICE_NAME = google_cloud_run_v2_service.fastapi_service.name
    _REPO_NAME    = google_artifact_registry_repository.repo.name
  }

  depends_on = [
    google_project_iam_member.cloudbuild_run_admin,
    google_project_iam_member.cloudbuild_sa_user
  ]
}

