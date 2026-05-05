# Grant allUsers invoker access so the Cloud Run service can be reached
# from a browser without a GCP identity token.
#
# Prerequisites: gcloud CLI authenticated with an account that has
# Cloud Run Admin + resourcemanager.projects.setIamPolicy permissions.
#
# Usage: .\scripts\cloud\make_public.ps1

$Project = "recosys-489001"
$Region  = "us-central1"
$Service = "recosys-recommender"

Write-Host "Making $Service public…" -ForegroundColor Cyan

gcloud run services add-iam-policy-binding $Service `
    --member="allUsers" `
    --role="roles/run.invoker" `
    --region=$Region `
    --project=$Project

Write-Host ""
Write-Host "Done. The service is now publicly accessible." -ForegroundColor Green
Write-Host "To re-enable auth at any time, run:" -ForegroundColor Yellow
Write-Host "  gcloud run services remove-iam-policy-binding $Service ``" -ForegroundColor Gray
Write-Host "      --member=allUsers --role=roles/run.invoker ``" -ForegroundColor Gray
Write-Host "      --region=$Region --project=$Project" -ForegroundColor Gray
