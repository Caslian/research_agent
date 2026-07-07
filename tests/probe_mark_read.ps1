$ErrorActionPreference = "Stop"
try {
    $url = "http://127.0.0.1:8000/api/v1/papers/mark-read?user_id=00000000-0000-0000-0000-000000000000&limit=5"
    $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 10
    Write-Host "STATUS: $($resp.StatusCode)"
    Write-Host "BODY: $($resp.Content)"
} catch {
    Write-Host "ERROR: $($_.Exception.Message)"
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        Write-Host "BODY: $($reader.ReadToEnd())"
    }
}
