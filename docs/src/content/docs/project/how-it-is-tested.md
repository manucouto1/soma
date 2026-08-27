---
title: How it is tested
description: The tests are another crate, so they only see the public API — five gates, and the four ways they go green without being green.
---

## The tests live outside `src/`

They are **another crate**. They see only the public API, so they cannot pass
by leaning on anything private, and a test that needs something private is a
signal that the something should be public or that the test is wrong.

One test binary per **type**, not one per file: `tests/unit/main.rs` with one
`mod` per module. The tests mirror the shape of the source, and a family of
types gets a folder with a `mod.rs` on both sides.

```
soma-core/tests/unit/main.rs
soma-store/tests/unit/main.rs
soma-tree/tests/{unit,end_to_end}/main.rs
soma-fabric/wire/tests/{unit,worker}/main.rs
```

## The five gates

```bash
uv run cargo test --workspace
uv run cargo clippy --workspace --all-targets -- -D warnings
uv run cargo fmt --all -- --check

conda activate mos && cd soma-python && maturin develop && python -m pytest tests/ -q

SOMA_CLUSTER=1 python -m pytest tests/cluster -q          # opt-in
```

Measured on 27 August 2026:

| gate | result | time |
|---|---|---|
| `cargo test --workspace` | **803 passed**, 4 ignored | ~10 s warm |
| clippy · fmt · `cargo doc` | clean, **0 warnings** | ~5 s |
| `pytest tests/` | **883 passed**, 41 skipped, 1 xfailed | 2:10–3:15 |
| `pytest tests/cluster` | 26 passed | **13–25 min** |

`uv run` and not bare `cargo`: PyO3 refuses the system interpreter, which is
ahead of what it supports, and `.python-version` names one it accepts. There is
no environment to activate on the Rust side and none to keep.

`--all-targets` matters more than it looks. The tests are separate crates
outside `src/`, so without it clippy never reads a line of them.

**`maturin develop` is not optional before `pytest`.** The Python suite runs
against the *installed* extension, so a change in `soma-python/src/` that was
not rebuilt means a suite that is green about code that is not the code.

Three of the five can run on every commit. The hook is in the repository and
installing it is one line, because a hook that installs itself is a hook that
runs code you did not read:

```bash
ln -s ../../.github/hooks/pre-commit .git/hooks/pre-commit
```

The Python suite and the cluster are **not** in it on purpose: they need an
interpreter with torch, and docker. **A hook that takes four minutes is a hook
people pass `--no-verify` to.**

## The four ways they go green without being green

Each of these happened.

**1. A test that skips looks green, and the signal is the clock.** The
end-to-end tests of `soma-tree` skip on purpose when there is no interpreter
that can import the library. Checking it properly means counting:

```bash
uv run cargo test -p somatize-tree --test end_to_end -- --nocapture
```

Zero skipped is that they ran. Five seconds running, 0.1 seconds skipping.

**1-bis. And if the image will not build, the cluster gate skips all 26 in
green.** `soma-python/pyproject.toml` began declaring `readme = "../README.md"`
for publication metadata and the Dockerfile did not copy `README.md`; maturin
died with a `No such file or directory` that **did not name the file**, the
fixture read *the cluster would not come up*, and everything skipped. The signal
was the clock again: 26 seconds where the suite takes 11 minutes.

> **Anything added to the publication metadata — a readme, a license file, an
> included file — has to be copied in the Dockerfile.**

**2. The cluster gate builds its image from the working tree.** Do not touch
the checkout while it runs. If the tree moved, the run proved something else and
the gate has to be repeated.

**3. Five of the cluster's tests want a GPU**, and their image is 11 GB. The
fixture skips if `worker-gpu` is not up, so on a machine without one they go
green having said very little.

**4. Anything that restores files with their old mtime leaves cargo asleep** —
`tar -x`, `cp -p`, `rsync -t`. The test count is the tell: add one, and if the
number does not go up, nothing recompiled.

## The store's two halves

A `Store` has one contract and two implementors, and the second one needs
something to run against:

```bash
docker compose -f soma-store/tests/docker/compose.yaml up -d
SOMA_S3=http://127.0.0.1:9000 uv run cargo test -p somatize-store --features s3
```

Opt-in the same way and with the same handshake on both sides: `SOMA_S3` being
set means there is one.

## The notebooks are a gate too, informally

`examples/` ships **executed, with its outputs**, so re-running one after an API
change is a test that reads its own results. Two real bugs were found writing
them, and one regression was caught by a notebook that no test could have
caught — the pre-pass named the root and the walk named it again, so the batch
was hashed twice and asking early cost exactly what it saves.

## One workspace, and it used to be two

`soma-fabric/wire` and `soma-fabric/broker` are ordinary members of the
workspace, so `cargo test --workspace` covers them like everything else. They
were a separate workspace next door once, and the first gate did not reach them
then; the copy that still sits beside this repository is abandoned and does not
compile.

They are two crates rather than one with modules, and the boundary earns its
keep: the dependency between them runs **one way only**, so *the cable knows
nothing about the rendezvous* is checked by the compiler. See
[the crates](/soma/reference/crates/).
