output "agent_service_account_email" {
  description = "Email of the dedicated IAM Service Account running PokéGrade AI."
  value       = google_service_account.pokegrade_agent_sa.email
}

output "staging_bucket_name" {
  description = "Name of the Cloud Storage staging bucket for Vertex AI artifacts."
  value       = google_storage_bucket.reasoning_engine_staging.name
}

output "staging_bucket_url" {
  description = "gs:// URI of the Cloud Storage staging bucket."
  value       = "gs://${google_storage_bucket.reasoning_engine_staging.name}"
}

output "tcg_market_secret_id" {
  description = "Fully-qualified Secret Manager resource name for TCG_MARKET_API_KEY."
  value       = google_secret_manager_secret.tcg_market_api_key.name
}

output "vertex_search_datastore_id" {
  description = "Discovery Engine DataStore ID for Vertex AI Search session retrieval."
  value       = google_discovery_engine_data_store.pokegrade_session_search_datastore.data_store_id
}

output "vertex_vector_search_index_name" {
  description = "Vertex AI Vector Search Index resource name for card appraisal embeddings."
  value       = google_vertex_ai_index.pokegrade_card_vector_index.name
}

output "reasoning_engine_config" {
  description = "Vertex AI Reasoning Engine deployment configuration metadata."
  value       = terraform_data.vertex_ai_reasoning_engine_runtime.output
}

