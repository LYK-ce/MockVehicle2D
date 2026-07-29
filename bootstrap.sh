#!/bin/bash
# bootstrap.sh — One-shot environment setup for MockVehicle2D
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
else
    echo "    .venv already exists, skipping."
fi

echo "==> Activating and installing..."
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e ".[dev]"

echo ""
echo "✅ Bootstrap complete!"
echo ""
echo "   Activate:  source .venv/bin/activate"
echo "   Test:      python -m pytest"
echo "   Server:    mockvehicle2d serve"
echo ""
