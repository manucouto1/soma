# Releasing

What has to happen for a version to publish, in order, and what the first one
needed that the ones after it do not.

## Where things stand

| | Published | At | Next |
|---|---|---|---|
| PyPI `somatize` | yes | **1.0.0** | a bump of it |
| crates.io — the eight this workspace publishes | yes | **1.0.0** | the same number, shared |
| crates.io `somatize`, `-runtime`, and eight more | yes, from the old implementation | 0.5.1 | **nothing, ever** |

**1.0.0 and not 0.6.0.** This is a re-derivation, not a version more: the names
carry over, the code does not. crates.io versions are immutable, so publishing
1.0.0 with another body breaks nobody who pinned 0.5.1 — and the ten old names
with no counterpart here (`somatize`, `-runtime`, `-agent`, `-llm`, `-mcp`,
`-coordinator`, `-compiler`, `-macros`, `-memory`, `-worker`) simply stop
publishing a version. There is nothing to do with them.

`somatize-python` is not published to crates.io at all: it is a `cdylib`, so
nothing can depend on it from Rust, and it carries `publish = false`. Its
artifact is the wheel.

**The number is shared.** `release.toml` sets `shared-version`, so all eight go
out at the tag's number whether or not anything in them moved — a patch that
only touches the Python half still publishes eight crates. That is the point:
one number describes one tree, and a reader who has `somatize-core` 1.0.1 knows
which `somatize-store` it was tested against.

## The first release needed a token. Only the first did, and it is done.

`release.yml` publishes with Trusted Publishing (OIDC) and stores no tokens.
That works for a crate that **already exists**: crates.io keys a
trusted-publisher config to a crate id, and unlike PyPI it has no "pending
publisher" for a name that has never been published. There is nowhere to
configure a publisher for a crate that does not exist yet.

`somatize-core` and `somatize-store` carried theirs over from 0.5.x — same
repository, same workflow file name, which is why **this file is still called
`release.yml`**. The other six went out once from a machine with a crates.io
token, in dependency order, and all eight are at 1.0.0 now. So none of this is
needed again — unless a **new crate** joins the list, in which case it is
exactly this, once, before the tag:

```bash
for c in somatize-core somatize-health somatize-study somatize-store \
         somatize-data somatize-tree somatize-fabric-wire \
         somatize-fabric-broker; do
  cargo publish -p "$c" || break
done
```

`cargo publish` waits for each crate to appear in the index before returning,
so the next one resolves. **Stop at the first failure** rather than pressing on
— a crate published out of order cannot be unpublished, only yanked. That is
what `|| break` is for, and it is the lesson the 0.5.0 release paid for: the
publish loop used to end every line in `|| true`, so three crates the facade
needed had never gone out and nobody knew.

## Trusted Publishing, configured once per crate

**crates.io** — for each of the eight crates `release.yml` publishes: Settings →
Trusted Publishing → GitHub, with

- repository owner `manucouto1`, repository `soma`
- workflow filename `release.yml`
- environment: leave unrestricted (the workflow declares no environment)

Check it exists for **every** crate in the list, not just the first.
Configuring one and assuming the rest is what stopped the 0.5.0 run at its
third crate with `403 Forbidden: The provided access token is not valid for
crate somatize-store`.

**PyPI** — project `somatize` → Publishing → a GitHub publisher with the same
repository and the same workflow filename.

## Then tag

```bash
cargo release patch     # bumps the shared version, commits, tags and pushes
```

By hand it is four things and not one, and the third is the one that is easy to
miss:

```bash
# 1. the shared version, AND the `version =` beside every internal `path`
#    dependency — a registry ignores `path`, so that number is what resolves.
# 2. cargo update --workspace          # the lock names the members too
# 3. python docs/scripts/python_surface.py
# 4. git commit && git tag v1.0.1 && git push && git push --tags
```

Step 3 is not documentation housekeeping: `docs/python-surface.json` is
committed, it records `__version__` of the package it was dumped from, and
`ci.yml` re-derives it with `--check` against the extension it just built. A
bump without it turns `main` red on the commit that releases.

`release.yml` runs on `v*` and does four jobs:

- `publish-crates` — eight crates in dependency order. A version already on
  crates.io is skipped; **anything else fails the job**.
- `publish-pypi` — one manylinux wheel plus an sdist, built in a single job so
  nothing passes through artifact storage. **One** wheel and not four:
  `somatize-python` builds against pyo3's limited API (`abi3-py310`), so a
  single `cp310-abi3` wheel answers for 3.10 and everything after it, including
  interpreters that did not exist when it was compiled.
- `github-release` — notes generated from the commits, and only if both of the
  above got there.
- `worker-image` — **called** by this workflow and not triggered by the release,
  so no image is cut for a version that failed to go out. It used to listen for
  `release: [published]` and never fired: GitHub raises no workflow from an
  event the `GITHUB_TOKEN` caused, and v1.0.0 went out with no image and nothing
  said so.

## What can be verified before you tag, and what cannot

The Python half can be checked completely without publishing anything:

```bash
maturin sdist -m soma-python/Cargo.toml --out dist
```

It vendors the whole workspace and only resolves because every path dependency
carries a `version` beside its `path`. Without that, cargo has nothing to write
in the manifest that goes to the registry, where `path` means nothing.

The Rust half cannot be dry-run end to end before the fact. `cargo package`
verifies against the registry, so every crate after the first reports

```
failed to select a version for the requirement `somatize-core = "^1.0.1"`
```

until `somatize-core` is actually published at the version being released.
**That is expected, not a defect.** What can be checked in advance is the metadata, and that is what
`cargo package --no-verify -p somatize-core` (or `-health`, or `-study` — the
three with no internal dependencies) confirms: every crate inherits `version`,
`license = "Elastic-2.0"`, `repository` and `readme` from `[workspace.package]`,
and carries a `description` of its own. crates.io accepts that licence; the
crates published at 0.5.1 already carry it.

The `LICENSE` text is **not** inside the packaged crate, deliberately: a crate
would have to drop the SPDX `license` field to carry `license-file`, and that
identifier is what crates.io indexes. The repository link in the metadata is
what carries the terms.

## Before any of it: the gates

A tag publishes; nothing between a tag and crates.io runs a test. `ci.yml`
covers the light half on every push to `main`, but the two opt-in gates are
yours and they are the ones that exercise the Dockerfile, the wheel built
inside the image, and a broker reaching workers over a network:

```bash
SOMA_CLUSTER=build python -m pytest tests/cluster -q      # 26 tests, 13–25 min
docker compose -f soma-store/tests/docker/compose.yaml up -d
SOMA_S3=http://127.0.0.1:9000 uv run cargo test -p somatize-store --features s3
```

The cluster suite needs the GPU workers up, or **14 of its 26 tests skip in
green** — the five device tests and all nine of the search. `cluster.yml` runs
it nightly on the self-hosted runner and fails on any skip, but a release is
worth running it by hand first.
