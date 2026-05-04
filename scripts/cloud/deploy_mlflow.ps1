# Build the MLflow tracking server image and deploy it to Cloud Run.
# --min-instances=1 keeps the SQLite DB alive between requests.
# Usage: .\scripts\cloud\deploy_mlflow.ps1 [-Tag v1]
param([string]$Tag = "v1")
$ErrorActionPreference = "Stop"

function Assert-Exit {
    param([string]$Cmd)
    if ($LASTEXITCODE -ne 0) { Write-Error "$Cmd failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}

$PROJECT  = "recosys-489001"
$REGION   = "us-central1"
$IMAGE    = "$REGION-docker.pkg.dev/$PROJECT/recosys-images/gru4rec-mlflow:$Tag"
$SERVICE  = "recosys-mlflow"
$BUCKET   = "gs://recosys-mlflow-artifacts/"

$ProjNum   = gcloud projects describe $PROJECT --format="value(projectNumber)"
Assert-Exit "gcloud projects describe"
$ComputeSA = "${ProjNum}-compute@developer.gserviceaccount.com"

# Grant MLflow SA write access to artifact bucket (idempotent)
Write-Host "=== Granting GCS write access to compute SA ===" -ForegroundColor Cyan
try { gsutil iam ch "serviceAccount:${ComputeSA}:objectAdmin" gs://recosys-mlflow-artifacts/ } catch {}

# Create GCS artifact bucket (ignore error if already exists)
Write-Host "=== Ensuring MLflow artifact bucket exists ===" -ForegroundColor Cyan
try {
    gsutil mb -p $PROJECT -l $REGION $BUCKET
    if ($LASTEXITCODE -ne 0) { Write-Host "  Bucket already exists, skipping." }
} catch { Write-Host "  Bucket already exists, skipping." }

Write-Host "`n=== Building $IMAGE ===" -ForegroundColor Cyan
docker build -f Dockerfile.mlflow -t $IMAGE .
Assert-Exit "docker build"

Write-Host "`n=== Pushing $IMAGE ===" -ForegroundColor Cyan
docker push $IMAGE
Assert-Exit "docker push"

Write-Host "`n=== Deploying Cloud Run: $SERVICE ===" -ForegroundColor Cyan
gcloud run deploy $SERVICE `
  --image $IMAGE `
  --region $REGION `
  --platform managed `
  --memory 1Gi `
  --cpu 1 `
  --timeout 60 `
  --concurrency 10 `
  --min-instances 1 `
  --max-instances 1 `
  --service-account $ComputeSA `
  --allow-unauthenticated `
  --project $PROJECT
Assert-Exit "gcloud run deploy"

$MlflowUrl = gcloud run services describe $SERVICE `
  --region $REGION --project $PROJECT --format "value(status.url)"
Assert-Exit "gcloud run services describe"

Write-Host "`n=== MLflow deployed! ===" -ForegroundColor Green
Write-Host "MLflow UI : $MlflowUrl"
Write-Host ""
Write-Host "Log experiments (run after deploy):"
Write-Host "  python scripts/serving/log_experiments_mlflow.py --mlflow-uri $MlflowUrl"
