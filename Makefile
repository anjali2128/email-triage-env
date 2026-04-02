# Makefile — Email Triage OpenEnv
# Usage: make <target>

.PHONY: help install test server inference docker deploy-hf validate clean

# Default target
help:
	@echo ""
	@echo "Email Triage OpenEnv — Available commands:"
	@echo ""
	@echo "  make install     Install dependencies into .venv"
	@echo "  make test        Run pytest test suite"
	@echo "  make server      Start FastAPI server on :7860"
	@echo "  make inference   Run baseline inference script"
	@echo "  make validate    Run OpenEnv spec validation"
	@echo "  make docker      Build + run Docker container"
	@echo "  make deploy-hf   Deploy to HuggingFace Spaces"
	@echo "  make clean       Remove build artifacts"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip -q
	.venv/bin/pip install -r requirements.txt -q
	@echo "✓ Installed. Activate with: source .venv/bin/activate"

# ── Development ───────────────────────────────────────────────────────────────
test:
	@source .venv/bin/activate 2>/dev/null || true; pytest tests/ -v --tb=short

server:
	@source .venv/bin/activate 2>/dev/null || true; \
	uvicorn server:app --reload --host 0.0.0.0 --port 7860

inference:
	@if [ -f .env ]; then set -a && source .env && set +a; fi; \
	source .venv/bin/activate 2>/dev/null || true; \
	python inference.py

validate:
	@curl -sf http://localhost:7860/validate | python3 -m json.tool || \
	echo "Server not running. Start with: make server"

# ── Docker ────────────────────────────────────────────────────────────────────
docker:
	bash docker_run.sh

docker-build:
	docker build -t email-triage-env .

docker-stop:
	docker stop email-triage-env 2>/dev/null || true

docker-logs:
	docker logs email-triage-env -f

# ── Deploy ────────────────────────────────────────────────────────────────────
deploy-hf:
	bash deploy_hf.sh

push-github:
	bash push_github.sh

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache baseline_scores.json
	@echo "✓ Cleaned"

clean-all: clean
	rm -rf .venv
