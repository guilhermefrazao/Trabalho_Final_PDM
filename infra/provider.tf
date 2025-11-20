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

data "google_client_config" "default" {}

data "google_container_cluster" "cluster_airflow" {
  name     = google_container_cluster.primary.name
  location = google_container_cluster.primary.location
}


provider "helm" {
  kubernetes = {
    host                   = "https://${google_container_cluster.primary.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
  }
}


provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
}