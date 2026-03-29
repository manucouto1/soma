---
title: Environment Manager
description: Isolated Python environments per pipeline with incremental dependency updates
---

# Environment Manager

The EnvManager creates and maintains isolated Python environments (venv or conda) for each pipeline, with intelligent incremental dependency updates.

## How it works

1. **First execution**: creates venv, installs all dependencies
2. **Same requirements**: reuses environment (instant, hash match)
3. **Added package**: `pip install` only the new one
4. **Changed version**: `pip install --upgrade` only that package
5. **Removed package**: `pip uninstall` the removed one
6. **Stale environments**: cleaned up after configurable timeout

## Requirements Hash

Dependencies are tracked by a normalized hash:
- Lines sorted alphabetically
- Comments and blank lines ignored
- Same deps in different order → same hash

```
# These two produce the same hash:
numpy==1.26           scikit-learn==1.4
scikit-learn==1.4      numpy==1.26
```

## Lockfile

Each environment stores a `lockfile.json`:
```json
{
  "packages": {"numpy": "1.26", "scikit-learn": "1.4"},
  "requirements_hash": "a3f8b2c1...",
  "env_type": "venv",
  "python_version": "3.11"
}
```

## Configuration

```rust
let env_mgr = EnvManager::new("/envs", EnvType::Venv);
// or
let env_mgr = EnvManager::new("/envs", EnvType::Conda);
```

## Docker Worker

The Soma worker Docker image uses EnvManager internally:
```bash
docker run -d \
  -e NOUS_API_KEY=nous_xxx \
  -e NOUS_URL=wss://server/nous/ws/worker \
  ghcr.io/manucouto1/soma-worker:latest
```

Pre-installed: numpy, scipy, scikit-learn, pandas, matplotlib, soma.
Additional dependencies installed per-pipeline on first execution.
