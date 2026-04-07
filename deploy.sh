#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Deploy Mamba Audio AI frontend to Vercel + push to GitHub.
#
# Usage:
#   ./deploy.sh                  # build, push to GitHub, deploy to production
#   ./deploy.sh --preview        # deploy a preview build (still pushes to git)
#   ./deploy.sh --skip-git       # skip git push, only deploy to Vercel
#   ./deploy.sh "commit message" # use custom commit message
#
# Requirements:
#   - Vercel CLI (`npm i -g vercel`)
#   - gh CLI authenticated (`gh auth login`)
#   - First run: you will be prompted to link the project.
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

# ── Parse args ───────────────────────────────────────────────────────────────
DEPLOY_MODE="production"
VERCEL_ARGS="--prod --yes"
SKIP_GIT=0
COMMIT_MSG=""

for arg in "$@"; do
    case "$arg" in
        --preview)
            DEPLOY_MODE="preview"
            VERCEL_ARGS="--yes"
            ;;
        --skip-git)
            SKIP_GIT=1
            ;;
        *)
            COMMIT_MSG="$arg"
            ;;
    esac
done

# ── Build ────────────────────────────────────────────────────────────────────
info "Building frontend..."
npm run build || error "Build failed"
success "Build complete"

# ── Git push ─────────────────────────────────────────────────────────────────
if [ "$SKIP_GIT" -eq 0 ]; then
    cd "$SCRIPT_DIR"
    if [ -d ".git" ]; then
        if [ -n "$(git status --porcelain)" ]; then
            info "Committing changes..."
            git add -A
            MSG="${COMMIT_MSG:-chore: deploy $(date '+%Y-%m-%d %H:%M')}"
            git commit -m "$MSG" || warn "Nothing to commit"
        else
            info "No local changes to commit"
        fi
        info "Pushing to GitHub..."
        git push origin HEAD || warn "git push failed — continuing with deploy"
        success "Pushed to GitHub"
    else
        warn "Not a git repository — skipping git push"
    fi
    cd "$FRONTEND_DIR"
else
    warn "Skipping git push (--skip-git)"
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
