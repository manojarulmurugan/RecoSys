#!/usr/bin/env bash
# Submit GRU4Rec V9 training job to Vertex AI.
#
# Prerequisites (run once if not done):
#   gcloud auth configure-docker us-central1-docker.pkg.dev
#   docker build -f Dockerfile.gru4rec \
#       -t us-central1-docker.pkg.dev/recosys-489001/recosys-images/gru4rec:v9 .
#   docker push us-central1-docker.pkg.dev/recosys-489001/recosys-images/gru4rec:v9
#
# Usage:
#   bash scripts/cloud/submit_vertex_job.sh
set -euo pipefail

PROJECT_ID="recosys-489001"
REGION="us-central1"
IMAGE="us-central1-docker.pkg.dev/${PROJECT_ID}/recosys-images/gru4rec:v9"
CONFIG="configs/vertex_gru4rec_1m.yaml"
TIMESTAMP=$(date +%Y%m%d-%H%M)

echo "================================================================"
echo "  GRU4Rec V9 — Vertex AI Job Submission"
echo "================================================================"
echo "  Image    : ${IMAGE}"
echo "  Config   : ${CONFIG}"
echo "  Region   : ${REGION}"
echo "  Job name : gru4rec-v9-1m-${TIMESTAMP}"
echo ""

# Verify GCS artifacts exist before submitting
echo "  Checking GCS artifacts..."
gsutil ls gs://recosys-data-bucket/data/1M/sequences_v2/train_sessions.parquet > /dev/null 2>&1 \
    || { echo "  ERROR: GCS artifacts not found. Run build_1m_sample.py first."; exit 1; }
echo "  GCS artifacts: OK"
echo ""

# Submit
gcloud ai custom-jobs create \
    --region="${REGION}" \
    --display-name="gru4rec-v9-1m-${TIMESTAMP}" \
    --config="${CONFIG}" \
    --project="${PROJECT_ID}"

echo ""
echo "  Job submitted. To stream logs:"
echo "  gcloud ai custom-jobs stream-logs JOB_ID --region=${REGION}"
echo ""
echo "  Monitor at:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
