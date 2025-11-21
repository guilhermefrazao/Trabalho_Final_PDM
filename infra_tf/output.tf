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