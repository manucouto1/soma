# Releasing

What actually has to happen for `v0.4.0` to publish, in order, and why the
first one is not like the ones after it.

## Where things stand

Verified against crates.io and PyPI on 2026-08-05:

| | Published | At |
|---|---|---|
| PyPI `somatize` | yes | 0.3.1 |
| crates.io `somatize-macros`, `-core`, `-compiler`, `-runtime`, `-memory`, `-worker`, `-agent`, `somatize` | yes | 0.3.1 |
| crates.io `somatize-store`, `somatize-llm`, `somatize-coordinator` | **never** | — |
| crates.io `somatize-mcp` | never, deliberately | — |

The eight published crates are exactly the ones `release.yml` used to list.
The three that were never published are exactly the three it was missing —
and at 0.3.1 the facade did not depend on them, which is why nobody
noticed. At 0.4.0 `somatize` depends on all three, so a release with the
old list would fail at the facade, and the `|| true` on every line would
have reported success while leaving `somatize` at 0.3.1 for good.

## The first release needs a token. Only the first.

`release.yml` publishes with Trusted Publishing (OIDC) and stores no
tokens. That works for a crate that **already exists**: crates.io keys a
trusted-publisher config to a crate id
(`trustpub_configs_github.crate_id`), and unlike PyPI it has no "pending
publisher" for names that have never been published. There is nowhere to
configure a publisher for a crate that does not exist yet.

So, once, from a machine with a crates.io token:

```bash
cargo publish -p somatize-store
cargo publish -p somatize-llm
cargo publish -p somatize-coordinator
```

They must go in dependency order relative to what they need
(`somatize-store` and `somatize-llm` need `somatize-core` and
`somatize-runtime`, both already published; `somatize-coordinator` needs
`somatize-worker`, also published), so the order above works as written.

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
