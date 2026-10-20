terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_apis = toset([
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
    "firestore.googleapis.com",
    "dlp.googleapis.com",
    "discoveryengine.googleapis.com",
  ])

  common_labels = {
    application = "pokegrade-ai"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# 1. Enable Required Google Cloud Platform APIs
resource "google_project_service" "enabled_apis" {
  for_each           = local.required_apis
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# 2. Dedicated Least-Privilege Service Account for PokéGrade AI Agent Runtime
resource "google_service_account" "pokegrade_agent_sa" {
  account_id   = "pokegrade-agent-${var.environment}-sa"
  display_name = "PokéGrade AI Multi-Agent Runtime Service Account (${var.environment})"
  project      = var.project_id
  depends_on   = [google_project_service.enabled_apis]
}

# 3. IAM Role Bindings (Vertex AI, Secret Manager, Cloud DLP, Vertex AI Search, Cloud Trace, Logging)
resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.pokegrade_agent_sa.email}"
}

resource "google_project_iam_member" "secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.pokegrade_agent_sa.email}"
}

resource "google_project_iam_member" "dlp_user" {
  project = var.project_id
  role    = "roles/dlp.user"
  member  = "serviceAccount:${google_service_account.pokegrade_agent_sa.email}"
}

resource "google_project_iam_member" "discovery_engine_viewer" {
  project = var.project_id
  role    = "roles/discoveryengine.viewer"
  member  = "serviceAccount:${google_service_account.pokegrade_agent_sa.email}"
}

resource "google_project_iam_member" "trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.pokegrade_agent_sa.email}"
}

resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.pokegrade_agent_sa.email}"
}

# 4. Cloud Storage Staging Bucket for Vertex AI Reasoning Engine & Vector Index Artifacts
resource "google_storage_bucket" "reasoning_engine_staging" {
  name                        = "${var.project_id}-pokegrade-staging-${var.environment}"
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  labels                      = local.common_labels

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 5
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_storage_bucket_iam_member" "staging_bucket_object_viewer" {
  bucket = google_storage_bucket.reasoning_engine_staging.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.pokegrade_agent_sa.email}"
}

# 5. Google Cloud Secret Manager Secret Containers (Zero Hardcoded Keys)
resource "google_secret_manager_secret" "tcg_market_api_key" {
  secret_id = "TCG_MARKET_API_KEY"
  project   = var.project_id
  labels    = local.common_labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled_apis]
}

# 6. Vertex AI Search Data Store (Persistent Conversational & Card Portfolio Retrieval)
resource "google_discovery_engine_data_store" "pokegrade_session_search_datastore" {
  location                    = "global"
  data_store_id               = "pokegrade-session-history-${var.environment}"
  display_name                = "PokéGrade AI Conversational History & Card Portfolio DataStore"
  industry_vertical           = "GENERIC"
  content_config              = "NO_CONTENT"
  solution_types              = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false
  project                     = var.project_id

  depends_on = [google_project_service.enabled_apis]
}

# 7. Vertex AI Vector Search Index for Card Appraisal Semantic Memory
resource "google_vertex_ai_index" "pokegrade_card_vector_index" {
  region              = var.region
  project             = var.project_id
  display_name        = "pokegrade-card-memory-index-${var.environment}"
  description         = "Vector store index for PokéGrade AI conversational history and card embeddings"
  index_update_method = "STREAM_UPDATE"
  labels              = local.common_labels

  metadata {
    contents_delta_uri = "gs://${google_storage_bucket.reasoning_engine_staging.name}/vector_index"
    config {
      dimensions                  = 16
      approximate_neighbors_count = 10
      distance_measure_type       = "COSINE_DISTANCE"
      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 500
          leaf_nodes_to_search_percent = 10
        }
      }
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

# 8. Vertex AI Reasoning Engine / Agent Runtime Deployment Specification
resource "terraform_data" "vertex_ai_reasoning_engine_runtime" {
  input = {
    project_id                   = var.project_id
    region                       = var.region
    display_name                 = var.agent_display_name
    service_account              = google_service_account.pokegrade_agent_sa.email
    staging_bucket_uri           = "gs://${google_storage_bucket.reasoning_engine_staging.name}"
    inspector_model              = var.inspector_model
    strategist_model             = var.strategist_model
    tcg_api_secret_name          = google_secret_manager_secret.tcg_market_api_key.name
    vertex_search_datastore_id   = google_discovery_engine_data_store.pokegrade_session_search_datastore.data_store_id
    vertex_vector_search_index   = google_vertex_ai_index.pokegrade_card_vector_index.name
  }

  depends_on = [
    google_project_iam_member.vertex_ai_user,
    google_project_iam_member.secret_accessor,
    google_project_iam_member.dlp_user,
    google_project_iam_member.discovery_engine_viewer,
    google_storage_bucket_iam_member.staging_bucket_object_viewer,
  ]
}

