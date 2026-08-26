# The generic worker, in a container that has **nothing of your project**.
#
# That is the whole point of building one: the tests that run against it prove
# what a subprocess on this machine cannot: that a worker with no clone, no
# source tree and no `PYTHONPATH` of yours executes your nodes because they
# **travelled**, and refuses to guess when they did not.
#
#   docker build -f soma-python/docker/worker.Dockerfile --target worker .
#   docker build -f soma-python/docker/worker.Dockerfile --target worker-gpu .
#
# It sits here and not under `tests/` because it is published: the cluster
# suite builds it through `tests/cluster/docker/compose.yaml`, and so does the
# image on ghcr that somebody standing up workers pulls instead of building.
#
# Two stages beyond the build so that the CPU workers stay at ~150 MB: torch is
# 2.5 GB and only the one with a GPU needs it.
#
# Installing is `uv` and not `pip` for one reason, and it is the third stage:
# `worker-gpu` **is** `worker` plus torch, so any change to the wheel invalidates
# its layer. With `pip --no-cache-dir` that is 2.5 GB off the network on every
# `SOMA_CLUSTER=build`; with a cache mount it is a copy from disk. The cache is a
# **mount and not a layer**, so nothing of it ends up in the image — which is why
# `--no-cache-dir` was there in the first place.

# ── Building the wheel ────────────────────────────────────────────────────────
FROM python:3.13-slim AS wheel

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && rm -rf /var/lib/apt/lists/*
# No toolchain here on purpose: `rust-toolchain.toml` at the root of the
# workspace names it, and rustup installs what that file says the first time
# cargo runs inside `/src/soma`. A number written here as well would be a
# second copy of the pin, and the two stopped agreeing once already.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain none
ENV PATH="/root/.cargo/bin:${PATH}"

# Pinned like the toolchain and like torch: an installer that changes under you
# between two rebuilds is the same class of surprise as a compiler that does.
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/
# The cache and `site-packages` are different filesystems, so a hardlink cannot
# be made; saying so is what stops uv from warning about it on every layer.
ENV UV_LINK_MODE=copy UV_SYSTEM_PYTHON=1
RUN --mount=type=cache,target=/root/.cache/uv uv pip install maturin

# One repository now. The wire and the broker came back from `soma-fabric`,
# so there is no second build context and no pair of directories to stand
# side by side: a path dependency that never leaves the workspace resolves
# wherever the workspace is copied.

WORKDIR /src/soma
# The manifests first, so that changing a line of Rust does not re-download the
# whole index.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
# The readme is metadata and not documentation here: `[workspace.package]` says
# `readme = "README.md"` and `pyproject.toml` says `../README.md`, so it is the
# wheel's long description. Without it maturin dies with a bare
# `No such file or directory` naming nothing, and the image stops building the
# day the package learns how to publish itself.
COPY README.md ./
COPY soma-core/Cargo.toml soma-core/
COPY soma-data/Cargo.toml soma-data/
COPY soma-health/Cargo.toml soma-health/
COPY soma-store/Cargo.toml soma-store/
COPY soma-study/Cargo.toml soma-study/
COPY soma-tree/Cargo.toml soma-tree/
COPY soma-fabric/wire/Cargo.toml soma-fabric/wire/
COPY soma-fabric/broker/Cargo.toml soma-fabric/broker/
COPY soma-python/Cargo.toml soma-python/pyproject.toml soma-python/
# Every member of the workspace, and not only what `python` names: cargo will
# not load a workspace one of whose members is not there.
COPY soma-core soma-core
COPY soma-data soma-data
COPY soma-health soma-health
COPY soma-store soma-store
COPY soma-study soma-study
COPY soma-tree soma-tree
COPY soma-fabric soma-fabric
COPY soma-python soma-python
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/src/target \
    maturin build --release --manifest-path soma-python/Cargo.toml --out /wheels

# ── A worker with no gradients in it ─────────────────────────────────────────
FROM python:3.13-slim AS worker

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_SYSTEM_PYTHON=1

COPY --from=wheel /wheels/*.whl /wheels/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install /wheels/*.whl cloudpickle && rm -rf /wheels

# Where a `project` worker looks for the code it is expected to have. Empty
# unless somebody mounts something on it, which is what the version tests do.
ENV PYTHONPATH=/clone
RUN mkdir -p /clone /store

# `--listen 0.0.0.0` because the client is outside this container. Whoever
# reaches this port runs code here: it belongs on a network you trust, and a
# compose network is one.
EXPOSE 7000
CMD ["python", "-m", "somatize.worker", "--listen", "0.0.0.0:7000"]

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
#
# `--index-url` and not `--extra-index-url`: the pytorch index carries torch's
# dependencies as well, and asking two indexes for the same name is how you get
# the CPU build of it by accident.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
