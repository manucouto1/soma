# The generic worker, in a container that has **nothing of your project**.
#
# That is the whole point of building one: the tests that run against it prove
# what a subprocess on this machine cannot: that a worker with no clone, no
# source tree and no `PYTHONPATH` of yours executes your nodes because they
# **travelled**, and refuses to guess when they did not.
#
#   docker build -f python/tests/cluster/docker/worker.Dockerfile --target worker .
#   docker build -f python/tests/cluster/docker/worker.Dockerfile --target worker-gpu .
#
# Two stages beyond the build so that the CPU workers stay at ~150 MB: torch is
# 2.5 GB and only the one with a GPU needs it.

# ── Building the wheel ────────────────────────────────────────────────────────
FROM python:3.13-slim AS wheel

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain 1.90.0
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir maturin

WORKDIR /src
# The manifests first, so that changing a line of Rust does not re-download the
# whole index.
COPY Cargo.toml Cargo.lock ./
COPY core/Cargo.toml core/
COPY store/Cargo.toml store/
COPY transport/Cargo.toml transport/
COPY python/Cargo.toml python/pyproject.toml python/
COPY core core
COPY store store
COPY transport transport
COPY python python
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/src/target \
    maturin build --release --manifest-path python/Cargo.toml --out /wheels

# ── A worker with no gradients in it ─────────────────────────────────────────
FROM python:3.13-slim AS worker

COPY --from=wheel /wheels/*.whl /wheels/
RUN pip install --no-cache-dir /wheels/*.whl cloudpickle && rm -rf /wheels

# Where a `project` worker looks for the code it is expected to have. Empty
# unless somebody mounts something on it, which is what the version tests do.
ENV PYTHONPATH=/clone
RUN mkdir -p /clone /store

# `--listen 0.0.0.0` because the client is outside this container. Whoever
# reaches this port runs code here: it belongs on a network you trust, and a
# compose network is one.
EXPOSE 7000
CMD ["python", "-m", "soma_next.worker", "--listen", "0.0.0.0:7000"]

# ── And one that has a GPU ───────────────────────────────────────────────────
FROM worker AS worker-gpu

# The CUDA runtime comes in the wheel; what the container needs from outside is
# the driver, and that is what `devices: [nvidia.com/gpu=all]` hands it.
#
# **Pinned**, and to the version the client has. Two reasons, and the second one
# was found the hard way: this suite compares the numbers a training run gets
# here against the ones it gets over there, and two torch versions are two
# arithmetics — and `2.11.0` kills this worker outright, with
# `_PyThreadState_Attach: non-NULL old thread state`, the moment a node runs
# `backward()` and another `forward` follows it. Whatever that is, it is not
# something to discover in a rebuild six months from now.
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
