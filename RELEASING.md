# Releasing

What actually has to happen for a version to publish, in order, and why
the first one was not like the ones after it.

## Where things stand

| | Published | At | Next |
|---|---|---|---|
| PyPI `somatize` | yes | 0.5.1 | **0.6.0** |
| crates.io — all eleven | yes | 0.5.1 | **0.6.0** |
| crates.io `somatize-mcp` | never, deliberately | — | — |

Verified against both registries on 2026-08-06: all eleven crates and the
wheel are at 0.5.1, with no crate lagging behind the others this time.

### What 0.6.0 carries

Breaking, so the minor digit is the one that moves. `Runner::fit` and
`GraphSession::fit` return `Fitted` instead of `(Value, HashMap<NodeId,
Value>)` — a tuple whose two halves nothing distinguished. Rust callers
have to destructure a struct; the Python surface is untouched.

Three corrections ride along, and none of them was being looked for —
each surfaced while giving a type a name:

- **`fit` stored one node's output as another node's state.** Persisted
  state from 0.5.x on the differentiable path came from the wrong place;
  re-fit.
- **The zarr chunk cache was written and never read.** Cost and latency
  only, no result changes.
- **A `Remote` node inside a `Loop` ran locally, in silence.** The
  compiler did not descend into loop bodies, so distribution was dropped
  with no error and no warning.

The rest is structural: nine domain folders across `soma-core`,
`soma-runtime` and `soma-python`, and `graph/mod.rs` down from 2 371 to
798 LOC.

### What 0.5.1 carried

A patch: one behaviour fix, one panic, and the READMEs.

- **A missing `safetensors` silently erased every channel snapshot.**
  `gradient_audit(channels=True)` recorded nothing and said nothing; the
  symptom appeared much later as `plot_channels` reporting "no channel
  snapshots (available: [])", a message about the filter rather than the
  missing package. Found by running an example against the *published*
  0.5.0 in a clean environment — the package is now declared in the `viz`
  extra, and the ImportError warns.
- **`num_replicas: 0` reached a panic.** It passed straight through
  `unwrap_or(1)`, made both the data-parallel loops run zero times, and
  handed an empty slice to an aggregator that indexed `[0]`. Guarded at
  the Python boundary, in the aggregator, and in the shared helper.
- **Every crate has a README.** All eleven rendered an empty one on
  crates.io and always had: cargo uses `README.md` from the package root,
  and only `soma-python` had one.

Nothing breaking, so the patch digit is the right one to move.

**Both halves are verified, not assumed.**

- PyPI: clean venv on 3.13, `pip install somatize==0.5.0`, `import soma`,
  a graph fitted and forwarded, a `soma.Pbt` constructed.
- crates.io: a brand-new crate outside this workspace, `cargo add
  somatize@0.5.0`, compiles and runs.

Wheels cover **3.12 and 3.13 only** (`--interpreter python3.12
python3.13` in `release.yml`). A 3.14 user falls back to the sdist and
therefore needs a Rust toolchain — worth adding an interpreter to that
line before it becomes the common case.

## What the v0.5.0 release taught, for the next one

The first pass **stopped at the third crate**:

```
403 Forbidden: The provided access token is not valid for crate `somatize-store`
```

Trusted publishing had been configured for `somatize-macros` and
`somatize-core` and not for the other nine. Two things made that
recoverable rather than a disaster:

1. **The fail-fast.** Every line in the publish loop used to end in
   `|| true`. It now swallows *only* "already exists", so the job stopped
   in dependency order having published exactly two crates — a consistent
   prefix — instead of pressing on and leaving holes. A crate published
   out of order cannot be unpublished, only yanked.
2. **"Already exists" is skipped.** After adding the missing publishers,
   `gh run rerun <id> --failed` walked the same list, skipped the two
   that were done, and continued from `somatize-store`.

So: before tagging, check the trusted publisher exists for **every**
crate in the list, not just the first. Configuring one crate and assuming
the rest looks identical until the run is halfway through.

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
