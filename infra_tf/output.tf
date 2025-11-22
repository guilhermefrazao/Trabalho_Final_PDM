output "cluster_name" {
  description = "Nome do cluster GKE"
  value       = google_container_cluster.primary.name
}


output "cluster_endpoint" {
  description = "Endpoint do cluster GKE"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "Certificado CA do cluster (base64)"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "location" {
  value = google_container_cluster.primary.location
}

output "repo_name" {
  description = "Repositóry name"
  value       = google_artifact_registry_repository.repo.name
}

output "bucket_name" {
  description = "Bucket to keep de model .pkl"
  value       = google_storage_bucket.mlflow_bucket.name
}