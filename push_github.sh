#!/usr/bin/env bash
# push_github.sh — Initialize git repo and push to GitHub
# Usage: bash push_github.sh [GITHUB_REPO_URL]
# Example: bash push_github.sh https://github.com/yourname/email-triage-env
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[git]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

REPO_URL="${1:-}"

# ── Prerequisites ─────────────────────────────────────────────────────────────
command -v git &>/dev/null || error "git not installed. Get it from https://git-scm.com"

# ── Git identity ──────────────────────────────────────────────────────────────
if [ -z "$(git config --global user.email 2>/dev/null)" ]; then
    warn "Git user not configured. Setting defaults..."
    echo -n "  Enter your GitHub email: "; read GIT_EMAIL
    echo -n "  Enter your GitHub name:  "; read GIT_NAME
    git config --global user.email "$GIT_EMAIL"
    git config --global user.name  "$GIT_NAME"
fi
info "  Git user: $(git config --global user.name) <$(git config --global user.email)>"

# ── Init repo ─────────────────────────────────────────────────────────────────
if [ ! -d ".git" ]; then
    info "Initializing git repository..."
    git init
    git branch -M main
fi

# ── .gitignore ────────────────────────────────────────────────────────────────
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'GITIGNORE'
.venv/
__pycache__/
*.pyc
*.pyo
.env
.DS_Store
baseline_scores.json
*.egg-info/
dist/
build/
.pytest_cache/
GITIGNORE
    info "  Created .gitignore"
fi

# ── Stage and commit ──────────────────────────────────────────────────────────
info "Staging files..."
git add -A
if git diff --cached --quiet; then
    info "  Nothing new to commit."
else
    git commit -m "feat: Email Triage OpenEnv v1.0

Real-world email triage environment for AI agents.
- 15 realistic emails with ground-truth labels  
- 3 tasks: easy / medium / hard
- Dense reward function with partial credit
- FastAPI server (step/reset/state endpoints)
- Baseline inference script (OpenAI client)
- Docker + HF Spaces ready"
    info "  Committed ✓"
fi

# ── Remote ────────────────────────────────────────────────────────────────────
if [ -z "$REPO_URL" ]; then
    warn "No repo URL provided."
    echo ""
    echo "  Create a repo at https://github.com/new then run:"
    echo "  bash push_github.sh https://github.com/YOUR_USERNAME/email-triage-env"
    echo ""
    echo "  Or set the remote manually:"
    echo "  git remote add origin https://github.com/YOUR/REPO.git"
    echo "  git push -u origin main"
    exit 0
fi

# Add remote if not already there
if ! git remote get-url origin &>/dev/null; then
    git remote add origin "$REPO_URL"
    info "  Remote added: $REPO_URL"
else
    CURRENT=$(git remote get-url origin)
    if [ "$CURRENT" != "$REPO_URL" ]; then
        git remote set-url origin "$REPO_URL"
        info "  Remote updated: $REPO_URL"
    fi
fi

# ── Push ──────────────────────────────────────────────────────────────────────
info "Pushing to GitHub..."
git push -u origin main
info "  Pushed ✓"

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║   GitHub push complete!                   ║"
echo "╠════════════════════════════════════════════╣"
echo "║  Repo: $REPO_URL"
echo "╚════════════════════════════════════════════╝"
