terraform {
  required_version = ">= 1.6.0"
}

# Reference only. State, credentials and production endpoints remain outside Git.
variable "project_id" { type = string }
variable "artifact_bucket" { type = string }

resource "google_storage_bucket" "model_artifacts" {
  name          = var.artifact_bucket
  project       = var.project_id
  location      = "NORTHAMERICA-NORTHEAST1"
  uniform_bucket_level_access = true
  versioning { enabled = true }
}

