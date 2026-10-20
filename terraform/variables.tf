variable "project_id" {
  description = "Google Cloud Platform project ID hosting PokéGrade AI."
  type        = string
  default     = "pokegrade-gcp-prod"
}

variable "region" {
  description = "Primary GCP region for Vertex AI Agent Runtime and Cloud Storage."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment label (dev, staging, prod)."
  type        = string
  default     = "prod"
}

variable "agent_display_name" {
  description = "Display name for the PokéGrade AI Vertex AI Reasoning Engine."
  type        = string
  default     = "pokegrade-ai-concierge"
}

variable "inspector_model" {
  description = "Fast Gemini model used by the Card Inspector Subagent."
  type        = string
  default     = "gemini-3.8-flash"
}

variable "strategist_model" {
  description = "High-reasoning Gemini model used by the Market Strategy Subagent."
  type        = string
  default     = "gemini-3.1-pro-preview"
}

