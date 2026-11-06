# Production Dockerfile for ANPR-India API.
# Usage:
#   docker build -t anpr-india:latest .
#   docker run -p 8000:8000 --gpus all anpr-india:latest
#
# For CPU-only deployment, change base image to python:3.11-slim and remove
# the CUDA dependencies.

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-venv \
    libgl1 libglib2.0-0 curl git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Install deps first (cached layer)
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY benchmarks/ ./benchmarks/

# Install the package in editable mode
RUN pip install -e . --no-deps

# Create weights directory; mount at runtime
RUN mkdir -p /app/weights /app/data
VOLUME ["/app/weights", "/app/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default config
ENV ANPR_CONFIG=/app/configs/serve.yaml

# Expose FastAPI
EXPOSE 8000

# Run with uvicorn (single worker — ONNX models aren't fork-safe)
CMD ["uvicorn", "anpr.api:get_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
