FROM python:3.11-slim

# Metadata for HF Spaces
LABEL maintainer="openenv-submission"
LABEL org.opencontainers.image.title="Email Triage OpenEnv"
LABEL org.opencontainers.image.description="Real-world email triage environment for AI agents"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY env/ ./env/
COPY server.py .
COPY inference.py .
COPY openenv.yaml .

# Create empty __init__ files
RUN touch env/__init__.py

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/validate')"

# Environment variables with defaults
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
