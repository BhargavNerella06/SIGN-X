#!/usr/bin/env bash
# SIGN-X first-time setup for Linux/macOS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== SIGN-X setup (Linux/macOS) ==="

echo
echo "[1/4] Creating Python virtual environment (backend/.venv)..."
cd "$REPO_ROOT/backend"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
else
    echo ".venv already exists — skipping."
fi

echo
echo "[2/4] Installing backend dependencies..."
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo
echo "[3/4] Downloading MediaPipe model assets..."
"$REPO_ROOT/scripts/download_models.sh"

echo
echo "[4/4] Installing frontend dependencies..."
cd "$REPO_ROOT/frontend"
if command -v npm >/dev/null 2>&1; then
    npm install
else
    echo "WARNING: npm not found. Install Node.js 18+ then run 'npm install' inside frontend/ manually."
fi

echo
echo "Setup complete. Next: start the backend (uvicorn app.main:app --reload) and frontend (npm run dev) separately."
