---
title: Gitflow & Workflow
description: Branch strategy, commit conventions, and development workflow.
---

## Branch Strategy

Soma follows a **Gitflow** branching model adapted for a multi-crate Rust workspace:

```
main ─────────────────────────────────────────────► (releases)
  │
  └── develop ────────────────────────────────────► (integration)
        │
        ├── feature/core-filter-trait ────────────► (merged to develop)
        ├── feature/compiler-cache-resolution ────► (merged to develop)
        ├── feature/runtime-event-bus ────────────► (merged to develop)
        │
        ├── release/0.1.0 ───────────────────────► (merged to main + develop)
        │
        └── hotfix/cache-key-collision ──────────► (merged to main + develop)
```

### Branch Types

| Branch | Purpose | Base | Merges to |
|---|---|---|---|
| `main` | Production releases. Tagged with versions. | - | - |
| `develop` | Integration branch. Always buildable. | `main` | - |
| `feature/<name>` | New functionality | `develop` | `develop` |
| `release/<version>` | Release preparation | `develop` | `main` + `develop` |
| `hotfix/<name>` | Critical fixes | `main` | `main` + `develop` |

### Feature Branch Naming

Feature branches follow the pattern: `feature/<crate>-<description>`

Examples:
- `feature/core-filter-trait`
- `feature/core-search-dimensions`
- `feature/compiler-topological-sort`
- `feature/compiler-cache-resolver`
- `feature/runtime-executor`
- `feature/runtime-bayesian-sampler`
- `feature/python-filter-bindings`

## Commit Conventions

Commits follow **Conventional Commits** with crate scope:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Usage |
|---|---|
| `feat` | New functionality |
| `fix` | Bug fix |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or modifying tests |
| `docs` | Documentation changes |
| `perf` | Performance improvement |
| `ci` | CI/CD changes |
| `chore` | Maintenance (deps, tooling) |

### Scopes

Scope matches the crate name:

| Scope | Crate |
|---|---|
| `core` | soma-core |
| `compiler` | soma-compiler |
| `runtime` | soma-runtime |
| `worker` | soma-worker |
| `memory` | soma-memory |
| `agent` | soma-agent |
| `python` | soma-python |
| `docs` | documentation |

### Examples

```
feat(core): add Filter trait with fit/forward lifecycle
feat(core): add SearchDimension enum with Float, Int, Categorical
feat(compiler): implement topological sort with Kahn's algorithm
fix(compiler): handle single-node graphs in parallelism detection
test(runtime): add integration tests for tiered cache promotion
refactor(core): rename Process to Filter for clarity
docs(design): add gradient propagation documentation
perf(runtime): use arena allocator for context store
```

## Pull Request Workflow

### Opening a PR

1. Create a feature branch from `develop`
2. Implement with TDD (see [TDD Strategy](/development/tdd/))
3. Ensure all tests pass: `cargo test --workspace`
4. Ensure clippy is clean: `cargo clippy --workspace`
5. Ensure formatting: `cargo fmt --check`
6. Open PR against `develop`

### PR Requirements

- Title follows conventional commit format
- Description includes: what, why, and how to test
- All CI checks pass (tests, clippy, fmt, docs)
- At least one approving review
- No merge conflicts with `develop`

### PR Size

Prefer small, focused PRs:

- **One trait/type per PR** in early development
- **One module per PR** for larger features
- If a PR touches more than 3 crates, consider splitting

## Release Process

1. Create `release/<version>` from `develop`
2. Update `Cargo.toml` versions across workspace
3. Update CHANGELOG.md
4. Run full test suite + manual testing
5. Merge to `main` with version tag
6. Merge back to `develop`
7. Publish to crates.io (Rust) and PyPI (Python)

### Versioning

Soma follows **Semantic Versioning** (SemVer):

- **0.x.y**: Pre-1.0 development. Breaking changes allowed in minor versions.
- **1.x.y**: Stable. Breaking changes only in major versions.

All crates in the workspace share the same version number.

## CI/CD Pipeline

```yaml
on: [push, pull_request]

jobs:
  check:
    - cargo fmt --check
    - cargo clippy --workspace -- -D warnings
    - cargo test --workspace
    - cargo doc --workspace --no-deps

  coverage:
    - cargo tarpaulin --workspace --out xml

  python:
    - maturin build
    - pytest tests/
```

## Development Environment

### Prerequisites

```bash
# Rust toolchain
rustup install stable
rustup component add clippy rustfmt

# Python (for soma-python development)
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest

# Documentation
cd docs && npm install
```

### Common Commands

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p soma-core

# Run clippy
cargo clippy --workspace -- -D warnings

# Format code
cargo fmt --workspace

# Build documentation
cargo doc --workspace --open

# Build Python package
cd soma-python && maturin develop

# Run docs site
cd docs && npm run dev
```
