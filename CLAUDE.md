# Soma - Development Guide

## Project

Soma is a computational graph runtime for research pipelines, agent orchestration, and data virtualization. Written in Rust with Python bindings (PyO3).

## Quick Commands

```bash
# Full check (what CI runs)
cargo fmt --all -- --check && cargo clippy --workspace -- -D warnings && cargo test --workspace

# With ChronosVector feature
cargo test -p soma-memory --features chronos

# Python
cd soma-python && maturin develop && pytest tests/ -v

# Docs
cd docs && npm run dev     # dev server
cd docs && npm run build   # production build
cargo doc --workspace --open  # Rust API docs
```

## Workspace Structure

```
soma-macros     → proc macro (#[derive(SomaFilter)]), no deps on soma-*
soma-core       → types, traits, enums (Filter, Value, Graph, Event, Schema, etc.)
soma-compiler   → Graph → ExecutionPlan (cache resolution, schema validation, distribution)
soma-runtime    → executor, Pipeline, samplers, pruners, cache, event bus, stream
soma-memory     → KnowledgeBase trait + MemoryKnowledgeBase + ChronosKnowledgeBase (feature-gated)
soma-worker     → protocol, Worker, Axum HTTP/WebSocket server
soma-agent      → Agent, Action, ResearchPlan trait
soma-python     → PyO3 bindings (Pipeline, Study, Filter, Lab)
```

## Conventions

- **Commits**: Conventional Commits with crate scope: `feat(core): add Schema type`
- **Branches**: Gitflow — `main`, `develop`, `feature/<crate>-<desc>`
- **Tests**: TDD. All public APIs must have tests. Edge cases in `tests/edge_cases.rs`.
- **Clippy**: `cargo clippy --workspace -- -D warnings` must pass (zero warnings)
- **Enums**: Public enums use `#[non_exhaustive]` for future extensibility
- **Errors**: Use `SomaError` (will be split per-crate before 1.0)

## Key Design Decisions

- **Filter trait**: `fit()` learns state, `forward()` transforms. Both independently cacheable.
- **CacheKey**: SHA-256 content-addressable. `hash(config + input_hash)` with cascade invalidation.
- **ExecutionPlan**: Compiled from Graph. Variants: Sequence, Parallel, Execute, Cached, Remote.
- **VirtualValue**: Lazy references (Materialized | Cached | Deferred | Stream).
- **Schema**: dtype + shape for compile-time type checking between connected filters.
- **TrialOutcome**: Separates control flow (Completed | Pruned) from errors.
- **StreamMode**: FixedState | Evolving (checkpoints) | Barrier (materializes).
- **Distribution**: Local | Remote(WorkerId | Tag) — compiler wraps in ExecutionPlan::Remote.

## Feature Flags

- `soma-memory/chronos`: Enables ChronosVector-backed KnowledgeBase

## Release

```bash
# Bump version in workspace Cargo.toml, then:
cargo release patch  # or minor/major
# This publishes to crates.io and creates a git tag.
# GitHub Actions publishes Python wheels to PyPI on tag push.
```
