#!/usr/bin/env bash
# deploy_hf.sh — Deploy Email Triage OpenEnv to Hugging Face Spaces
# Usage: bash deploy_hf.sh [--space YOUR_HF_USERNAME/email-triage-env]
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[hf]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── Args ──────────────────────────────────────────────────────────────────────
SPACE_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --space) SPACE_ID="$2"; shift 2 ;;
        *) echo "Usage: bash deploy_hf.sh [--space USERNAME/space-name]"; exit 1 ;;
    esac
done

# Load .env
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

# ── Prerequisites ─────────────────────────────────────────────────────────────
info "Checking prerequisites..."

if ! command -v huggingface-cli &>/dev/null; then
    info "  Installing huggingface_hub..."
    pip install huggingface_hub[cli] --quiet
fi

if ! command -v git &>/dev/null; then
    error "git is required. Install from https://git-scm.com"
fi

# ── Login ─────────────────────────────────────────────────────────────────────
info "Logging into Hugging Face..."
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
    info "  Logged in with HF_TOKEN ✓"
else
    warn "  HF_TOKEN not set — attempting interactive login..."
    huggingface-cli login --add-to-git-credential
fi

# Get username
HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1 || echo "")
if [ -z "$HF_USER" ]; then
    error "Could not determine HF username. Run: huggingface-cli login"
fi
info "  Logged in as: $HF_USER"

# ── Space name ────────────────────────────────────────────────────────────────
if [ -z "$SPACE_ID" ]; then
    SPACE_ID="$HF_USER/email-triage-env"
fi
info "  Target Space: $SPACE_ID"

# ── Create Space if needed ────────────────────────────────────────────────────
info "Creating/checking HF Space: $SPACE_ID ..."
python3 - << PYEOF
from huggingface_hub import HfApi, create_repo
import sys

api = HfApi()
space_id = "$SPACE_ID"

try:
    api.repo_info(repo_id=space_id, repo_type="space")
    print("  Space already exists — will push updates.")
except Exception:
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
        print(f"  Created Space: {space_id}")
    except Exception as e:
        print(f"  Error creating space: {e}", file=sys.stderr)
        sys.exit(1)
PYEOF

# ── Push files ────────────────────────────────────────────────────────────────
info "Pushing files to HF Space..."
python3 - << PYEOF
from huggingface_hub import HfApi
import os

api = HfApi()
space_id = "$SPACE_ID"

# Files to upload (relative paths)
files = [
    "Dockerfile",
    "requirements.txt",
    "server.py",
    "inference.py",
    "openenv.yaml",
    "README.md",
    "env/__init__.py",
    "env/data.py",
    "env/environment.py",
    "env/graders.py",
]

for path in files:
    if not os.path.exists(path):
        print(f"  Skipping (not found): {path}")
        continue
    try:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path,
            repo_id=space_id,
            repo_type="space",
            commit_message=f"Upload {path}",
        )
        print(f"  Uploaded: {path}")
    except Exception as e:
        print(f"  Error uploading {path}: {e}")

print("  Upload complete.")
PYEOF

# ── Set Space secrets ─────────────────────────────────────────────────────────
info "Setting Space secrets..."
python3 - << PYEOF
from huggingface_hub import HfApi
import os

api = HfApi()
space_id = "$SPACE_ID"

secrets = {
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", os.environ.get("HF_TOKEN", "")),
    "API_BASE_URL": os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    "MODEL_NAME": os.environ.get("MODEL_NAME", "gpt-4o-mini"),
}

for key, val in secrets.items():
    if val:
        try:
            api.add_space_secret(repo_id=space_id, key=key, value=val)
            print(f"  Set secret: {key}")
        except Exception as e:
            print(f"  Could not set {key}: {e}")
    else:
        print(f"  Skipped (empty): {key}")
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────
SPACE_URL="https://huggingface.co/spaces/$SPACE_ID"
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║   Deployment complete!                            ║"
echo "╠════════════════════════════════════════════════════╣"
echo "║                                                    ║"
echo "║  Space URL:                                        ║"
echo "║    $SPACE_URL"
echo "║                                                    ║"
echo "║  API (once running):                               ║"
echo "║    $(echo $SPACE_URL | sed 's/spaces\///')-7860"
echo "║                                                    ║"
echo "║  Validate:                                         ║"
echo "║    curl $SPACE_URL/validate"
echo "║                                                    ║"
echo "║  Next: Set HF_TOKEN secret in Space settings if   ║"
echo "║  inference fails.                                  ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
