$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:8000/api/v1/papers"
$userId = "11111111-1111-1111-1111-111111111111"

Write-Host "--- POST /mark-read (arxiv:2401.99999) ---"
$body = @{
    user_id   = $userId
    arxiv_id  = "2401.99999"
    source    = "arxiv"
    title     = "Smoke Test Paper"
} | ConvertTo-Json -Compress
try {
    $r = Invoke-WebRequest -Uri "$base/mark-read" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 10
    Write-Host "STATUS: $($r.StatusCode)"
    Write-Host "BODY:   $($r.Content)"
} catch {
    Write-Host "ERROR: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "--- GET /mark-read (verify) ---"
try {
    $r = Invoke-WebRequest -Uri "$base/mark-read?user_id=$userId&limit=20" -UseBasicParsing -TimeoutSec 10
    Write-Host "STATUS: $($r.StatusCode)"
    Write-Host "BODY:   $($r.Content)"
} catch {
    Write-Host "ERROR: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "--- POST /mark-read 重复 (幂等性) ---"
try {
    $r = Invoke-WebRequest -Uri "$base/mark-read" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 10
    Write-Host "STATUS: $($r.StatusCode)"
    Write-Host "BODY:   $($r.Content)"
} catch {
    Write-Host "ERROR: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "--- POST /mark-read 缺 arxiv_id/ieee_id (应 400) ---"
$badBody = @{ user_id = $userId } | ConvertTo-Json -Compress
try {
    $r = Invoke-WebRequest -Uri "$base/mark-read" -Method POST -ContentType "application/json" -Body $badBody -UseBasicParsing -TimeoutSec 10
    Write-Host "STATUS: $($r.StatusCode)"
} catch {
    $resp = $_.Exception.Response
    if ($resp) {
        $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
        Write-Host "STATUS: $($resp.StatusCode.value__)"
        Write-Host "BODY:   $($reader.ReadToEnd())"
    } else {
        Write-Host "ERROR: $($_.Exception.Message)"
    }
}

Write-Host ""
Write-Host "--- 清理 ---"
$cleanupBody = @{
    user_id   = $userId
    arxiv_id  = "2401.99999"
    source    = "arxiv"
} | ConvertTo-Json -Compress
# 用 mark-read 再次提交是幂等的，不会删除；要真清理需走 DB，这里用 psql 或 python 脚本。
Write-Host "(跳过清理，需用 SQL: DELETE FROM user_paper_read_state WHERE user_id='$userId')"
