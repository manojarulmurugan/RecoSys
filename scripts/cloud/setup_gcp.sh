#!/usr/bin/env bash
# scripts/cloud/setup_gcp.sh
#
# One-shot GCP setup for the RecoSys 14-day plan.
# Idempotent: safe to re-run.  Each step gracefully no-ops if already done.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (`gcloud auth login`)
#   - Billing enabled on the project
#   - Sufficient free credits ($95-105 estimated for the full run)
#
# What this configures:
#   - Enables APIs:    Vertex AI, Cloud Run, Artifact Registry, Cloud Scheduler,
#                      Cloud Build, Cloud Storage
#   - Creates:         Artifact Registry repo for Docker images
#                      MLflow artifacts bucket
#                      drift/ prefix in the data bucket (logical, no creation needed)
#   - Configures:      docker auth for the Artifact Registry region
#
# What this does NOT do (manual one-time steps surfaced at the end):
#   - A100 quota request (web form, can't be automated)
#   - Billing alerts (recommended but optional)
#
# Usage:
#   bash scripts/cloud/setup_gcp.sh
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ID="recosys-489001"
REGION="us-central1"
DATA_BUCKET="recosys-data-bucket"
MLFLOW_BUCKET="recosys-mlflow-artifacts"
AR_REPO="recosys-images"

# ── Helpers ──────────────────────────────────────────────────────────────────
section() { echo; echo "=================================================================="; echo "  $1"; echo "=================================================================="; }
step()    { echo; echo "  >  $1"; }
note()    { echo "     $1"; }

# ── Project context ──────────────────────────────────────────────────────────
section "RecoSys GCP setup"
note "Project : $PROJECT_ID"
note "Region  : $REGION"

step "Setting active gcloud project"
gcloud config set project "$PROJECT_ID" >/dev/null
note "active: $(gcloud config get-value project 2>/dev/null)"

# ── Enable APIs ──────────────────────────────────────────────────────────────
step "Enabling required APIs (idempotent — already-enabled APIs no-op)"
gcloud services enable \
    aiplatform.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudscheduler.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    bigquery.googleapis.com \
    iam.googleapis.com \
    --quiet
note "APIs enabled."

# ── Artifact Registry ────────────────────────────────────────────────────────
step "Creating Artifact Registry repo for Docker images"
if gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" >/dev/null 2>&1; then
    note "Repo '$AR_REPO' already exists in $REGION."
else
    gcloud artifacts repositories create "$AR_REPO" \
        --repository-format=docker \
        --location="$REGION" \
        --description="RecoSys Docker images (training + serving + mlflow)" \
        --quiet
    note "Created '$AR_REPO' in $REGION."
fi

step "Configuring local docker auth for $REGION-docker.pkg.dev"
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
note "docker auth configured."

# ── Storage buckets ──────────────────────────────────────────────────────────
step "Verifying data bucket exists"
if gcloud storage buckets describe "gs://$DATA_BUCKET" >/dev/null 2>&1; then
    note "gs://$DATA_BUCKET (existing — used for samples / processed / drift / models)"
else
    note "WARNING: gs://$DATA_BUCKET does not exist. Create it before continuing."
    exit 1
fi

step "Creating MLflow artifacts bucket"
if gcloud storage buckets describe "gs://$MLFLOW_BUCKET" >/dev/null 2>&1; then
    note "gs://$MLFLOW_BUCKET already exists."
else
    gcloud storage buckets create "gs://$MLFLOW_BUCKET" \
        --location="$REGION" \
        --uniform-bucket-level-access \
        --quiet
    note "Created gs://$MLFLOW_BUCKET in $REGION."
fi

# Create logical prefixes (zero-byte placeholders) so subsequent scripts can
# read/write without surprises.  No-op if they already exist.
step "Initialising drift/ and mlflow/ prefixes"
echo "" | gcloud storage cp - "gs://$DATA_BUCKET/drift/.placeholder" --quiet 2>/dev/null || true
note "gs://$DATA_BUCKET/drift/   ready"

# ── Service account permissions for Vertex AI ────────────────────────────────
# The default Compute Engine SA needs a few roles for Vertex AI training jobs
# to read GCS data and write to Artifact Registry.  These are usually already
# granted at project level; this block confirms.
step "Verifying default Compute Engine SA has Vertex AI / GCS access"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
note "Compute SA: $COMPUTE_SA"

# Idempotent — re-granting is a no-op
for role in \
    "roles/aiplatform.user" \
    "roles/storage.objectAdmin" \
    "roles/artifactregistry.reader" \
    "roles/logging.logWriter"; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$COMPUTE_SA" \
        --role="$role" \
        --condition=None \
        --quiet >/dev/null 2>&1 || true
done
note "IAM bindings applied (or already present)."

# ── Done — manual steps ──────────────────────────────────────────────────────
section "GCP automated setup complete"
cat <<EOF

  Resources ready:
    Project              : $PROJECT_ID
    Region               : $REGION
    Artifact Registry    : ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/
    MLflow bucket        : gs://${MLFLOW_BUCKET}
    Data bucket          : gs://${DATA_BUCKET} (existing)

  ──────────────────────────────────────────────────────────────────
  ONE-TIME MANUAL STEPS (cannot be automated — please complete now):
  ──────────────────────────────────────────────────────────────────

  [1] A100 GPU quota request — DO THIS NOW (approval can take 1-2 days):
      Open: https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}
      Filter: "Custom model training Nvidia A100 GPUs per region"
      Region: ${REGION}
      Action: Click EDIT QUOTAS → request limit = 1 → submit
      Notes: Mention "ML portfolio project" in the justification.

  [2] (Recommended) Set a billing budget alert:
      Open: https://console.cloud.google.com/billing/budgets?project=${PROJECT_ID}
      Action: Create budget, threshold \$150 with email alert.

  Once quota is approved (you'll receive an email), Days 5+ of the plan
  are unblocked.  Days 1-4 do not need the A100, so continue with
  diagnostic + 500k local training (Colab) in the meantime.

EOF
