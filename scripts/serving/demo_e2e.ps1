# End-to-end demo: live recommendations + MLflow dashboard + drift report.
# Usage: .\scripts\serving\demo_e2e.ps1 [-ServiceUrl <url>] [-MlflowUrl <url>]
param(
    [string]$ServiceUrl = "",
    [string]$MlflowUrl  = ""
)
$ErrorActionPreference = "Stop"

function Sep { Write-Host ("=" * 60) }

if (-not $ServiceUrl) {
    $ServiceUrl = gcloud run services describe recosys-recommender `
        --region us-central1 --project recosys-489001 --format "value(status.url)"
}
if (-not $MlflowUrl) {
    try {
        $MlflowUrl = gcloud run services describe recosys-mlflow `
            --region us-central1 --project recosys-489001 --format "value(status.url)" 2>$null
    } catch { $MlflowUrl = "" }
}
$ServiceUrl = $ServiceUrl.TrimEnd("/")

# Identity token for authenticated Cloud Run (org policy may block allUsers)
$TOKEN   = gcloud auth print-identity-token 2>$null
$Headers = if ($TOKEN) { @{ Authorization = "Bearer $TOKEN" } } else { @{} }

Sep
Write-Host "  RecoSys End-to-End Demo" -ForegroundColor Cyan
Sep
Write-Host "  Serving : $ServiceUrl"
if ($MlflowUrl) { Write-Host "  MLflow  : $MlflowUrl" }
Write-Host ""

# --- 1. Health check ---
Write-Host "--- 1. Serving Health Check ---" -ForegroundColor Yellow
$health = Invoke-RestMethod "$ServiceUrl/health" -Headers $Headers
$health | ConvertTo-Json
Write-Host ""

# Real item IDs from REES46 1M vocabulary
$I1 = "1000544"; $I2 = "1000894"; $I3 = "1000978"; $I4 = "1001588"; $I5 = "1001605"

# --- 2a. Recommendations -- 1-item session ---
Write-Host "--- 2a. Recommendations -- 1-item session ---" -ForegroundColor Yellow
$body1 = @{
    session = @( @{ item_id = $I1; event_type = "view" } )
    top_k   = 20
} | ConvertTo-Json -Depth 3
$r1 = Invoke-RestMethod -Method Post "$ServiceUrl/recommend" -Headers $Headers `
    -ContentType "application/json" -Body $body1
$r1 | ConvertTo-Json
Write-Host ""

# --- 2b. Recommendations -- 5-item session ---
Write-Host "--- 2b. Recommendations -- 5-item session (view to cart) ---" -ForegroundColor Yellow
$body5 = @{
    session = @(
        @{ item_id = $I1; event_type = "view" }
        @{ item_id = $I2; event_type = "view" }
        @{ item_id = $I3; event_type = "view" }
        @{ item_id = $I4; event_type = "cart" }
        @{ item_id = $I5; event_type = "cart" }
    )
    top_k = 20
} | ConvertTo-Json -Depth 3
$r5 = Invoke-RestMethod -Method Post "$ServiceUrl/recommend" -Headers $Headers `
    -ContentType "application/json" -Body $body5
$r5 | ConvertTo-Json
Write-Host ""

# --- 2c. Recommendations -- 10-item session ---
Write-Host "--- 2c. Recommendations -- 10-item session (purchase signal) ---" -ForegroundColor Yellow
$body10 = @{
    session = @(
        @{ item_id = $I1; event_type = "view"     }
        @{ item_id = $I2; event_type = "view"     }
        @{ item_id = $I3; event_type = "view"     }
        @{ item_id = $I4; event_type = "view"     }
        @{ item_id = $I5; event_type = "view"     }
        @{ item_id = $I1; event_type = "cart"     }
        @{ item_id = $I2; event_type = "cart"     }
        @{ item_id = $I3; event_type = "cart"     }
        @{ item_id = $I1; event_type = "purchase" }
        @{ item_id = $I2; event_type = "view"     }
    )
    top_k = 20
} | ConvertTo-Json -Depth 3
$r10 = Invoke-RestMethod -Method Post "$ServiceUrl/recommend" -Headers $Headers `
    -ContentType "application/json" -Body $body10
$r10 | ConvertTo-Json
Write-Host ""

# --- 3. Drift report ---
Write-Host "--- 3. Distribution Drift (Jan 2020 vs Mar 2020 COVID period) ---" -ForegroundColor Yellow
$driftPath = "reports/drift_report.json"
if (Test-Path $driftPath) {
    $drift = Get-Content $driftPath | ConvertFrom-Json
    Write-Host "  Train window          : $($drift.train_window)"
    Write-Host "  Test window           : $($drift.test_window) (COVID period)"
    Write-Host "  JSD (normalized)      : $($drift.jsd_normalized)  (threshold=$($drift.alert_threshold))"
    Write-Host "  Top-50 item overlap   : $($drift.top50_item_overlap_pct)%"
    Write-Host "  Test item coverage    : $($drift.test_item_coverage_pct)%"
    $alertLabel = if ($drift.alert) { "YES" } else { "No" }
    Write-Host "  Alert                 : $alertLabel"
    Write-Host "  $($drift.narrative)"
} else {
    Write-Host "  No drift_report.json found."
    Write-Host "  Run: python scripts/monitoring/compute_drift.py"
}
Write-Host ""

# --- 4. Final summary ---
Sep
Write-Host "  Final Results Summary" -ForegroundColor Cyan
Sep
Write-Host "  Model         : GRU4Rec V9 session-based"
Write-Host "  Dataset       : REES46 1M-user (890,736 users, 222,864 items)"
Write-Host "  NDCG@20       : 0.2676  (T4Rec target >=0.22)"
Write-Host "  HR@20         : 0.4815  (T4Rec target >=0.44)"
Write-Host "  vs Popularity : 7.6x NDCG@20, 6.0x HR@20"
Write-Host "  vs T4Rec XLNet: +0.0130 NDCG@20 (+5.1% relative)"
Write-Host "  Items indexed : $($health.n_items_indexed)"
Write-Host "  Serving URL   : $ServiceUrl"
if ($MlflowUrl) { Write-Host "  MLflow UI     : $MlflowUrl" }
Sep
