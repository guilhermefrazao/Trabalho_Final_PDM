variable "project" {}

variable "region" {
  default = "us-central1"
}

variable "zone" {
  default = "us-central1-c"
}

variable "gcp_service_list" {
  description = "Lista de APIs necessárias para o projeto"
  type        = list(string)
  default = [
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
  ]
}

variable "image_tag_fastapi" {
  description = "A tag da imagem que foi feita o push recentemente"
  type        = string
}

variable "image_tag_mlflow" {
  description = "A tag da imagem que foi feita o push recentemente"
  type        = string
}

variable "image_tag_airflow" {
  description = "A tag da imagem que foi feita o push recentemente"
  type        = string
}