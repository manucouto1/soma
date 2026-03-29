# ═══════════════════════════════════════════════════
# Soma Worker — Docker image for remote execution
# ═══════════════════════════════════════════════════
#
# Usage:
#   docker build -t soma-worker -f soma-worker/Dockerfile .
#   docker run -d \
#     -e NOUS_API_KEY=nous_xxx \
#     -e NOUS_URL=wss://your-server/nous/ws/worker \
#     soma-worker
#
# For GPU support:
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04 \
#     -t soma-worker:gpu -f soma-worker/Dockerfile .
#   docker run -d --gpus all \
#     -e NOUS_API_KEY=nous_xxx \
#     -e NOUS_URL=wss://your-server/nous/ws/worker \
#     soma-worker:gpu

ARG BASE_IMAGE=python:3.11-slim

# ── Stage 1: Build Rust binary ──
FROM rust:1.82-slim AS builder

WORKDIR /src
COPY . .

RUN apt-get update && apt-get install -y pkg-config libssl-dev && \
    cargo build --release -p soma-worker && \
    cp target/release/soma-worker /soma-worker || true

# If no binary target, create a placeholder
RUN test -f /soma-worker || echo "#!/bin/sh\necho 'worker binary not built'" > /soma-worker

# ── Stage 2: Runtime ──
FROM ${BASE_IMAGE}

# Install Python if base doesn't have it (for CUDA images)
RUN if ! command -v python3 &> /dev/null; then \
        apt-get update && \
        apt-get install -y python3 python3-pip python3-venv && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Common ML dependencies (cached in image)
RUN pip install --no-cache-dir \
    numpy>=1.26 \
    scipy>=1.12 \
    scikit-learn>=1.4 \
    pandas>=2.1 \
    matplotlib>=3.8 \
    soma>=0.1 \
    2>/dev/null || pip3 install --no-cache-dir --break-system-packages \
    numpy scipy scikit-learn pandas matplotlib

# Worker binary
COPY --from=builder /soma-worker /usr/local/bin/soma-worker

# Environment storage
RUN mkdir -p /envs /work
ENV SOMA_ENV_DIR=/envs
ENV SOMA_WORK_DIR=/work

# Default command
ENV NOUS_URL=ws://host.docker.internal:5050/nous/ws/worker
ENV NOUS_API_KEY=""

ENTRYPOINT ["soma-worker"]
CMD ["--coordinator", "${NOUS_URL}", "--api-key", "${NOUS_API_KEY}", "--env-dir", "/envs"]
