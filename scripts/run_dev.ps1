# Starts the SIGN-X backend (uvicorn) and frontend (Vite) dev servers.
# Each runs in its own PowerShell window so their logs stay readable.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

$backendDir = Join-Path $repoRoot "backend"
$frontendDir = Join-Path $repoRoot "frontend"

# Use backend\.venv when setup_windows.ps1 created one; otherwise fall back
# to whichever `python` is on PATH (it must have requirements.txt installed).
$activate = ""
if (Test-Path (Join-Path $backendDir ".venv\Scripts\Activate.ps1")) {
    $activate = ".venv\Scripts\Activate.ps1; "
} else {
    Write-Host "backend\.venv not found - using the system 'python' on PATH." -ForegroundColor Yellow
}

Write-Host "Starting SIGN-X backend on http://localhost:8000 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd `"$backendDir`"; $activate python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
)

Write-Host "Starting SIGN-X frontend (Vite) ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd `"$frontendDir`"; npm run dev"
)

Write-Host "`nBoth dev servers are starting in separate windows." -ForegroundColor Green
Write-Host "Backend:  http://localhost:8000/docs"
Write-Host "Frontend: check the Vite window for its local URL (typically http://localhost:5173)"
