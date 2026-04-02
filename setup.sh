#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-command local dev environment setup for Email Triage OpenEnv
# Works on: macOS, Ubuntu/Debian, WSL2
# Usage:  bash setup.sh
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Email Triage OpenEnv — Environment Setup  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Python version check ──────────────────────────────────────────────────
info "Checking Python version..."
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON="$cmd"
            info "  Found: $cmd ($VER) ✓"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python 3.10+ not found. Install from https://python.org or run:\n  Ubuntu: sudo apt install python3.12\n  macOS:  brew install python@3.12"
fi

# ── 2. Virtual environment ───────────────────────────────────────────────────
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in .venv/ ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    info "Virtual environment .venv/ already exists — skipping creation."
fi

# Activate
source "$VENV_DIR/bin/activate"
info "Activated: $(which python) ($(python --version))"

# ── 3. Install dependencies ──────────────────────────────────────────────────
info "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "  Dependencies installed ✓"

# ── 4. Git check ─────────────────────────────────────────────────────────────
info "Checking git..."
if ! command -v git &>/dev/null; then
    warn "git not found. Install from https://git-scm.com"
else
    info "  git $(git --version | cut -d' ' -f3) ✓"
fi

# ── 5. Docker check ──────────────────────────────────────────────────────────
info "Checking Docker..."
if ! command -v docker &>/dev/null; then
    warn "Docker not found. Install from https://docs.docker.com/get-docker/"
    warn "  You can still run the env locally without Docker."
else
    info "  Docker $(docker --version | cut -d' ' -f3 | tr -d ',') ✓"
fi

# ── 6. Hugging Face CLI ───────────────────────────────────────────────────────
info "Checking Hugging Face CLI..."
if ! command -v huggingface-cli &>/dev/null; then
    info "  Installing huggingface_hub CLI..."
    pip install huggingface_hub[cli] --quiet
fi
info "  huggingface-cli ✓"

# ── 7. OpenEnv install ────────────────────────────────────────────────────────
info "Checking OpenEnv..."
if ! python -c "import openenv" &>/dev/null 2>&1; then
    info "  Installing OpenEnv..."
    pip install openenv --quiet 2>/dev/null || warn "  OpenEnv not yet on PyPI — validation via /validate endpoint instead."
else
    info "  OpenEnv ✓"
fi

# ── 8. Environment variables ──────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    info "Creating .env template..."
    cat > .env << 'ENV'
# Email Triage OpenEnv — environment variables
# Copy this file and fill in your values

# LLM API endpoint (OpenAI-compatible)
API_BASE_URL=https://api.openai.com/v1

# Model to use for inference
MODEL_NAME=gpt-4o-mini

# Your API key (HuggingFace token or OpenAI key)
HF_TOKEN=your_key_here
OPENAI_API_KEY=your_key_here
ENV
    warn "  Created .env — fill in your API keys before running inference!"
else
    info "  .env already exists ✓"
fi

# ── 9. Quick smoke test ───────────────────────────────────────────────────────
info "Running smoke test..."
python - << 'SMOKETEST'
import sys
sys.path.insert(0, '.')
from env.environment import EmailTriageEnv, Action
from env.graders import GRADERS

env = EmailTriageEnv()
obs = env.reset('easy_triage')
assert obs.pending_count == 5, f"Expected 5 emails, got {obs.pending_count}"

result = env.step(Action(
    action_type='classify',
    email_id='e001',
    urgency='critical',
    category='technical_support',
))
assert result.reward.step_reward > 0

for task_id, grader in GRADERS.items():
    env2 = EmailTriageEnv()
    env2.reset(task_id)
    score = grader(env2)['score']
    assert 0.0 <= score <= 1.0

print("  All checks passed ✓")
SMOKETEST

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Setup complete! Next steps:               ║"
echo "╠══════════════════════════════════════════════╣"
echo "║                                              ║"
echo "║  1. Activate env:                            ║"
echo "║     source .venv/bin/activate                ║"
echo "║                                              ║"
echo "║  2. Fill in API keys:                        ║"
echo "║     nano .env                                ║"
echo "║                                              ║"
echo "║  3. Start the server:                        ║"
echo "║     uvicorn server:app --reload --port 7860  ║"
echo "║                                              ║"
echo "║  4. Run baseline inference:                  ║"
echo "║     source .env && python inference.py       ║"
echo "║                                              ║"
echo "║  5. Run tests:                               ║"
echo "║     pytest tests/ -v                         ║"
echo "║                                              ║"
echo "║  6. Docker build & run:                      ║"
echo "║     bash docker_run.sh                       ║"
echo "║                                              ║"
echo "║  7. Deploy to HF Spaces:                     ║"
echo "║     bash deploy_hf.sh                        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
