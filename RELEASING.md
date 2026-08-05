# Releasing

What actually has to happen for a version to publish, in order, and why
the first one was not like the ones after it.

## Where things stand

| | Published | At |
|---|---|---|
| crates.io — all eleven | yes | 0.4.0 |
| PyPI `somatize` | yes | 0.3.1 |
| crates.io `somatize-mcp` | never, deliberately | — |

The eleven crates went out on 2026-08-05 in one token-authenticated pass,
and a crate outside this workspace depending on `somatize = "0.4.0"` from
crates.io compiles and runs. That is the Rust half of "can a stranger
install this", and it is yes rather than unknown.

## Why the next tag is v0.5.0, not v0.4.1

The tempting move is to tag `v0.4.0` again: the workflow skips a crate
that is already on crates.io, so it would publish PyPI alone and cost
nothing. It is the wrong move.

Eight commits landed after that crates.io pass, seven of them breaking —
workers gained a DataStore, the four state and gradient messages, real
federated and data-parallel training, model parallelism, and an error
path that no longer hangs its caller. Tagging `v0.4.0` would put that
code on PyPI as `somatize` 0.4.0 while crates.io kept the *old* 0.4.0
under the same number. Two artifacts, one version, different code — the
exact desynchronization the whole trusted-publishing setup exists to
avoid.

So: bump the workspace version, tag it, and let one pass publish both
registries from one tree. Breaking changes below 1.0 bump the minor,
which is why 0.4.0 → **0.5.0**.

**What is left: PyPI.** Configure the trusted publisher (below), then tag.
The release workflow will publish the eleven crates and do the
wheels.

### How it stood before, and why

Eight crates were published at 0.3.1 — exactly the eight `release.yml` used
to list. `somatize-store`, `somatize-llm` and `somatize-coordinator` had
never been published, exactly the three it was missing, and at 0.3.1 the
facade did not depend on them, which is why nobody noticed.

At 0.4.0 `somatize` depends on all three, so a release with the old list
would have failed at the facade, and the `|| true` on every line would have
reported success while leaving `somatize` at 0.3.1 for good.

## Why it could not be done piecemeal

Publishing only the three missing crates does not work, and the error is
the same one you get for any of them:

```
error: failed to prepare local package for uploading
  failed to select a version for the requirement `somatize-core = "^0.4.0"`
  candidate versions found which didn't match: 0.3.1, 0.3.0, 0.2.46, ...
```

Every workspace crate requires its siblings at `^0.4.0`, and the registry
only has 0.3.1. So the whole chain has to go out at 0.4.0, in order — the
eleven crates `release.yml` lists, which is the correct order (validated
against the dependency graph).

### The cycle that had to be broken first

`somatize-macros` had `somatize-core` as a dev-dependency declared
`{ workspace = true }`, and the workspace entry carries
`version = "0.4.0"`. A **versioned** dev-dependency survives into the
published manifest, so:

```
somatize-macros  needs  somatize-core   ^0.4.0    (dev-dependency)
somatize-core    needs  somatize-macros ^0.4.0    (normal dependency)
```

Neither can go first, and nothing published breaks the tie. The published
`somatize-macros` 0.3.1 manifest lists three dependencies and no dev ones,
which is the clue: cargo strips a dev-dependency that has **no version**.
It is now declared path-only (`{ path = "../soma-core" }`), it packages
(13 files, 41.9 kB), and `cargo test -p somatize-macros` is unaffected —
the path is all it ever needed.

## The first release needs a token. Only the first.

`release.yml` publishes with Trusted Publishing (OIDC) and stores no
tokens. That works for a crate that **already exists**: crates.io keys a
trusted-publisher config to a crate id
(`trustpub_configs_github.crate_id`), and unlike PyPI it has no "pending
publisher" for names that have never been published. There is nowhere to
configure a publisher for a crate that does not exist yet.

The first 0.4.0 release was therefore one token-authenticated pass over the
whole chain, from a machine with a crates.io token — **already done**, kept
here because the next major version starts from the same place if a new
crate is ever added:

```bash
for c in somatize-macros somatize-core somatize-store somatize-compiler \
         somatize-runtime somatize-memory somatize-llm somatize-worker \
         somatize-agent somatize-coordinator somatize; do
  cargo publish -p "$c" || break
done
```

`cargo publish` waits for each crate to appear in the index before
returning, so the next one resolves. Stop at the first failure rather than
pressing on — a crate published out of order cannot be unpublished, only
yanked.

After that pass, the eight that already existed and the three new ones are
all on 0.4.0, and every one of them can be given a trusted publisher.

## Then configure Trusted Publishing

**crates.io** — for each of the eleven crates `release.yml` publishes:
Settings → Trusted Publishing → GitHub, with

- repository owner `manucouto1`, repository `soma`
- workflow filename `release.yml`
- environment: leave unrestricted (the workflow declares no environment)

**PyPI** — project `somatize` → Publishing → add a GitHub publisher with
the same repository and workflow filename.

## Then tag

```bash
cargo release patch    # or minor/major — bumps the workspace version
```

`release.yml` runs on `v*` and does three jobs:

- `publish-crates` — eleven crates in dependency order. A version already
  on crates.io is skipped; **anything else fails the job**. That is the
  change that makes a release mean something: every line used to end in
  `|| true`.
- `publish-pypi` — manylinux wheels for 3.12 and 3.13 plus an sdist, built
  in one job so nothing passes through artifact storage.
- `github-release` — notes generated from the commits.

## Verifying before you tag

The Python half can be checked completely without publishing anything:

```bash
maturin sdist -m soma-python/Cargo.toml --out dist   # 237 files, ~790 kB
```

The Rust half cannot be dry-run end to end before the fact:
`cargo package` verifies against the registry, so every crate after the
first reports `failed to select a version for the requirement
somatize-core = "^0.4.0"` until `somatize-core` 0.4.0 is actually
published. That is expected, not a defect — the metadata is what can be
checked in advance, and it is complete (every crate inherits `description`,
`license = "Elastic-2.0"` and `repository` from `[workspace.package]`;
crates.io accepts that licence, the eight published crates carry it).

## What "released" will mean

Until a `v*` tag runs green with the crates and the wheels actually
landing, "Soma works" is a claim about a local checkout. The
[soma-examples](https://github.com/manucouto1/soma-examples) repository
installs Soma the way a stranger would, so once the release lands, pointing
its `requirements.txt` at `somatize` from PyPI makes its CI the standing
proof that the published package works.
