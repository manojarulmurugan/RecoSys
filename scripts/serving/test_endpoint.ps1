# Smoke-test the deployed Cloud Run serving endpoint.
# Usage: .\scripts\serving\test_endpoint.ps1 [-ServiceUrl <url>]
#        ServiceUrl auto-detected from gcloud if omitted.
param([string]$ServiceUrl = "")
$ErrorActionPreference = "Stop"

if (-not $ServiceUrl) {
    $ServiceUrl = gcloud run services describe recosys-recommender `
        --region us-central1 --project recosys-489001 --format "value(status.url)"
}
$ServiceUrl = $ServiceUrl.TrimEnd("/")
Write-Host "Testing endpoint: $ServiceUrl`n"

# Identity token for authenticated Cloud Run (org policy may block allUsers)
$TOKEN   = gcloud auth print-identity-token 2>$null
$Headers = if ($TOKEN) { @{ Authorization = "Bearer $TOKEN" } } else { @{} }

# Real item IDs from the REES46 1M vocabulary
$ITEM1 = "1000544"
$ITEM2 = "1000894"
$ITEM3 = "1000978"
$ITEM4 = "1001588"
$ITEM5 = "1001605"

$Pass = 0
$Fail = 0
$W    = -45

# ── 1. GET /health ─────────────────────────────────────────────────────────────
Write-Host -NoNewline ("  {0,$W}" -f "GET /health")
try {
    $r = Invoke-RestMethod "$ServiceUrl/health" -Headers $Headers
    if ($r.status -ne "ok") { throw "status=$($r.status)" }
    Write-Host "PASS  (n_items=$($r.n_items_indexed), ndcg20=$($r.ndcg_20))" -ForegroundColor Green; $Pass++
} catch {
    Write-Host "FAIL -- $_" -ForegroundColor Red; $Fail++
}

# ── 2. GET /recommend/example ──────────────────────────────────────────────────
Write-Host -NoNewline ("  {0,$W}" -f "GET /recommend/example")
try {
    Invoke-RestMethod "$ServiceUrl/recommend/example" -Headers $Headers | Out-Null
    Write-Host "PASS" -ForegroundColor Green; $Pass++
} catch {
    Write-Host "FAIL -- $_" -ForegroundColor Red; $Fail++
}

# ── 3. POST /recommend — 1-item session ───────────────────────────────────────
Write-Host -NoNewline ("  {0,$W}" -f "POST /recommend (1-item session)")
try {
    $body = "{`"session`":[{`"item_id`":`"$ITEM1`",`"event_type`":`"view`"}],`"top_k`":20}"
    $r = Invoke-RestMethod -Method Post "$ServiceUrl/recommend" `
        -ContentType "application/json" -Body $body -Headers $Headers
    if ($r.recommendations.Count -lt 1) { throw "0 recs returned" }
    Write-Host "PASS ($($r.recommendations.Count) recs)" -ForegroundColor Green; $Pass++
} catch {
    Write-Host "FAIL -- $_" -ForegroundColor Red; $Fail++
}

# ── 4. POST /recommend — 5-item session ───────────────────────────────────────
Write-Host -NoNewline ("  {0,$W}" -f "POST /recommend (5-item session)")
try {
    $body = "{`"session`":[{`"item_id`":`"$ITEM1`",`"event_type`":`"view`"},{`"item_id`":`"$ITEM2`",`"event_type`":`"view`"},{`"item_id`":`"$ITEM3`",`"event_type`":`"cart`"},{`"item_id`":`"$ITEM4`",`"event_type`":`"view`"},{`"item_id`":`"$ITEM5`",`"event_type`":`"purchase`"}],`"top_k`":20}"
    $r = Invoke-RestMethod -Method Post "$ServiceUrl/recommend" `
        -ContentType "application/json" -Body $body -Headers $Headers
    if ($r.recommendations.Count -lt 1) { throw "0 recs returned" }
    Write-Host "PASS ($($r.recommendations.Count) recs)" -ForegroundColor Green; $Pass++
} catch {
    Write-Host "FAIL -- $_" -ForegroundColor Red; $Fail++
}

# ── 5. POST /recommend — all-OOV should return 422 ────────────────────────────
Write-Host -NoNewline ("  {0,$W}" -f "POST /recommend (all-OOV -> 422)")
try {
    $body = '{"session":[{"item_id":"DOES_NOT_EXIST_999999999","event_type":"view"}],"top_k":20}'
    Invoke-RestMethod -Method Post "$ServiceUrl/recommend" `
        -ContentType "application/json" -Body $body -Headers $Headers | Out-Null
    Write-Host "FAIL (expected 422 but got 200)" -ForegroundColor Red; $Fail++
} catch {
    # PS 5.1: check StatusCode from the response object in the exception
    $code = $_.Exception.Response.StatusCode.value__
    if ($code -eq 422) {
        Write-Host "PASS (422 as expected)" -ForegroundColor Green; $Pass++
    } else {
        Write-Host "FAIL (expected 422, got HTTP $code)" -ForegroundColor Red; $Fail++
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Results: $Pass passed, $Fail failed"
if ($Fail -gt 0) { exit 1 }

Write-Host "`nFull /health response:" -ForegroundColor Cyan
Invoke-RestMethod "$ServiceUrl/health" -Headers $Headers | ConvertTo-Json
