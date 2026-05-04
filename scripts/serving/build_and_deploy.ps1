# Build the serving Docker image, push to Artifact Registry, deploy to Cloud Run.
# Usage: .\scripts\serving\build_and_deploy.ps1 [-Tag v1]
param([string]$Tag = "v1")
$ErrorActionPreference = "Stop"

# gcloud/docker are native executables — they don't throw on failure in PS.
# Use this after every native call to stop on non-zero exit codes.
function Assert-Exit {
    param([string]$Cmd)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Cmd failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

$PROJECT    = "recosys-489001"
$REGION     = "us-central1"
$IMAGE      = "$REGION-docker.pkg.dev/$PROJECT/recosys-images/gru4rec-serving:$Tag"
$SERVICE    = "recosys-recommender"
$GCS_CKPT   = "gs://recosys-data-bucket/models/gru4rec_session_v9_1M"
$GCS_VOCABS = "gs://recosys-data-bucket/data/1M/vocabs.pkl"

$ProjNum   = gcloud projects describe $PROJECT --format="value(projectNumber)"
Assert-Exit "gcloud projects describe"
$ComputeSA = "${ProjNum}-compute@developer.gserviceaccount.com"

# Grant the Cloud Run SA read access to the model bucket (idempotent)
Write-Host "=== Granting GCS read access to compute SA ===" -ForegroundColor Cyan
gsutil iam ch "serviceAccount:${ComputeSA}:objectViewer" gs://recosys-data-bucket
Assert-Exit "gsutil iam ch"
Write-Host "  Done: $ComputeSA -> objectViewer on recosys-data-bucket"

Write-Host "`n=== Building $IMAGE ===" -ForegroundColor Cyan
docker build -f Dockerfile.serving -t $IMAGE .
Assert-Exit "docker build"

Write-Host "`n=== Pushing $IMAGE ===" -ForegroundColor Cyan
docker push $IMAGE
Assert-Exit "docker push"

Write-Host "`n=== Deploying Cloud Run: $SERVICE ===" -ForegroundColor Cyan
$EnvVars = "GCS_CHECKPOINT_DIR=${GCS_CKPT},GCS_VOCABS_PATH=${GCS_VOCABS}"

# --cpu-boost allocates extra CPU during container startup (faster model load).
# --allow-unauthenticated may warn if org policy blocks allUsers — the service
# is still deployed; call it with an identity token if you get 403.
gcloud run deploy $SERVICE `
  --image $IMAGE `
  --region $REGION `
  --platform managed `
  --memory 4Gi `
  --cpu 2 `
  --timeout 120 `
  --concurrency 80 `
  --min-instances 0 `
  --max-instances 3 `
  --cpu-boost `
  --set-env-vars $EnvVars `
  --service-account $ComputeSA `
  --allow-unauthenticated `
  --project $PROJECT
Assert-Exit "gcloud run deploy"

$ServiceUrl = gcloud run services describe $SERVICE `
  --region $REGION --project $PROJECT --format "value(status.url)"
Assert-Exit "gcloud run services describe"

Write-Host "`n=== Deployed! ===" -ForegroundColor Green
Write-Host "Service URL : $ServiceUrl"
Write-Host ""
Write-Host "The service loads GCS artifacts in the background on first start."
Write-Host "Wait ~60s then run: .\scripts\serving\test_endpoint.ps1"
Write-Host ""
Write-Host "If you get 403 on requests (org policy blocks allUsers), authenticate with:"
Write-Host "  `$TOKEN = gcloud auth print-identity-token"
Write-Host "  Invoke-RestMethod `"$ServiceUrl/health`" -Headers @{Authorization=`"Bearer `$TOKEN`"}"
