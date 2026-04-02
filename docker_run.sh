#!/usr/bin/env bash
# docker_run.sh — Build and run the Email Triage OpenEnv container
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[docker]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }

IMAGE_NAME="email-triage-env"
PORT=7860

# Load .env if present
if [ -f ".env" ]; then
    set -a; source .env; set +a
    info "Loaded .env"
fi

# Require HF_TOKEN
if [ -z "${HF_TOKEN:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    warn "HF_TOKEN / OPENAI_API_KEY not set — inference will fail."
    warn "Set it in .env or: export HF_TOKEN=sk-..."
fi

TOKEN="${HF_TOKEN:-${OPENAI_API_KEY:-dummy}}"
BASE_URL="${API_BASE_URL:-https://api.openai.com/v1}"
MODEL="${MODEL_NAME:-gpt-4o-mini}"

# ── Build ─────────────────────────────────────────────────────────────────────
info "Building Docker image: $IMAGE_NAME ..."
docker build -t "$IMAGE_NAME" .
info "  Build complete ✓"

# ── Stop any existing container ───────────────────────────────────────────────
if docker ps -q --filter "name=$IMAGE_NAME" | grep -q .; then
    info "Stopping existing container..."
    docker stop "$IMAGE_NAME" && docker rm "$IMAGE_NAME"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
info "Starting container on http://localhost:$PORT ..."
docker run -d \
    --name "$IMAGE_NAME" \
    -p "$PORT:$PORT" \
    -e "API_BASE_URL=$BASE_URL" \
    -e "MODEL_NAME=$MODEL" \
    -e "HF_TOKEN=$TOKEN" \
    -e "OPENAI_API_KEY=$TOKEN" \
    "$IMAGE_NAME"

info "  Container started: $IMAGE_NAME"
info "  Waiting for server to be ready..."
sleep 3

# ── Health check ──────────────────────────────────────────────────────────────
MAX_RETRIES=10
for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf "http://localhost:$PORT/" > /dev/null 2>&1; then
        info "  Server healthy ✓"
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "Server did not start in time. Logs:"
        docker logs "$IMAGE_NAME" --tail 30
        exit 1
    fi
    sleep 2
done

# ── Validate OpenEnv spec ─────────────────────────────────────────────────────
info "Running OpenEnv validation..."
VALIDATE=$(curl -sf "http://localhost:$PORT/validate" || echo '{"status":"error"}')
STATUS=$(echo "$VALIDATE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "?")

if [ "$STATUS" = "pass" ]; then
    info "  OpenEnv validation: PASS ✓"
else
    warn "  OpenEnv validation: $STATUS"
    echo "$VALIDATE" | python3 -m json.tool 2>/dev/null || echo "$VALIDATE"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Container running at localhost:$PORT   ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Endpoints:                              ║"
echo "║    GET  /           — info               ║"
echo "║    GET  /tasks      — list tasks         ║"
echo "║    POST /reset      — start episode      ║"
echo "║    POST /step       — agent action       ║"
echo "║    GET  /state      — env state          ║"
echo "║    GET  /validate   — spec check         ║"
echo "║    GET  /grade      — grader score       ║"
echo "║                                          ║"
echo "║  Logs:  docker logs $IMAGE_NAME -f      ║"
echo "║  Stop:  docker stop $IMAGE_NAME         ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Quick API demo ────────────────────────────────────────────────────────────
info "Quick API demo:"
echo ""
echo "  # Reset:"
echo "  curl -s -X POST http://localhost:$PORT/reset \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"task_id\": \"easy_triage\"}' | python3 -m json.tool | head -20"
echo ""
echo "  # Classify an email:"
echo "  curl -s -X POST http://localhost:$PORT/step \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"action_type\": \"classify\", \"email_id\": \"e001\", \"urgency\": \"critical\", \"category\": \"technical_support\"}'"
echo ""
