#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Deploy Mamba Audio AI frontend to Vercel.
#
# Usage:
#   ./deploy.sh            # deploy to production
#   ./deploy.sh --preview  # deploy a preview build
#
# Requirements:
#   - Vercel CLI (`npm i -g vercel`)
#   - First run: you will be prompted to log in and link the project.
#     After that, every subsequent run is a one-command redeploy.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# ── Colors ───────────────────────────────────────────────────────────────────
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${BLUE}→${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn()    { echo -e "${YELLOW}⚠${NC} $1"; }
error()   { echo -e "${RED}✗${NC} $1" >&2; exit 1; }

# ── Pre-flight checks ────────────────────────────────────────────────────────
info "Running pre-flight checks..."

if ! command -v vercel &> /dev/null; then
    warn "Vercel CLI not found. Installing globally..."
    npm install -g vercel || error "Failed to install Vercel CLI"
fi

if [ ! -d "$FRONTEND_DIR" ]; then
    error "Frontend directory not found: $FRONTEND_DIR"
fi

cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
    info "Installing dependencies..."
    npm install || error "npm install failed"
fi

success "Pre-flight OK"

# ── Build ────────────────────────────────────────────────────────────────────
info "Building frontend..."
npm run build || error "Build failed"
success "Build complete"

# ── Deploy ───────────────────────────────────────────────────────────────────
DEPLOY_MODE="production"
VERCEL_ARGS="--prod --yes"

if [ "${1:-}" = "--preview" ]; then
    DEPLOY_MODE="preview"
    VERCEL_ARGS="--yes"
fi

info "Deploying to Vercel ($DEPLOY_MODE)..."
echo

vercel $VERCEL_ARGS

echo
success "Deployment complete!"
echo
info "Next steps:"
echo "  1. Set VITE_API_URL in Vercel dashboard → Settings → Environment Variables"
echo "     (point to your backend: e.g. https://your-backend.example.com)"
echo "  2. Re-run this script to redeploy with the new env var"
echo
