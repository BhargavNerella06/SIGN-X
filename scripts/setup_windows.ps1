# SIGN-X first-time setup for Windows 11 + PowerShell.
# Run from anywhere: powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== SIGN-X setup (Windows) ===" -ForegroundColor Cyan

# --- Backend: Python virtual environment ---
Push-Location (Join-Path $repoRoot "backend")

Write-Host "`n[1/4] Creating Python virtual environment (backend\.venv)..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
} else {
    Write-Host ".venv already exists — skipping."
}

Write-Host "`n[2/4] Installing backend dependencies..." -ForegroundColor Cyan
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

Pop-Location

# --- MediaPipe model download ---
Write-Host "`n[3/4] Downloading MediaPipe model assets..." -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "download_models.ps1")

# --- Frontend: npm install ---
Write-Host "`n[4/4] Installing frontend dependencies..." -ForegroundColor Cyan
Push-Location (Join-Path $repoRoot "frontend")
if (Get-Command npm -ErrorAction SilentlyContinue) {
    npm install
} else {
    Write-Warning "npm was not found on PATH. Install Node.js 18+ (https://nodejs.org/) then run 'npm install' inside frontend\ manually."
}
Pop-Location

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Next: run .\scripts\run_dev.ps1 to start both the backend and frontend."
