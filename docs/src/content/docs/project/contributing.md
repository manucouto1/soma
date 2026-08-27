---
title: Contributing
description: One branch per slice, a commit subject that says why rather than what, and five gates — with the one rule about writing anything at all.
---

## The rule before any of the others

**Nothing is written without a real consumer today.**

Not a crate, not a trait, not a variant, not a page. Building something before
anything uses it is how the version before this one ended up with fourteen
traits with a single implementor and two with none.

If two real implementors of a trait cannot be named **today**, it is a struct.
If a hole has no tenant, it is deleted — that has happened once already, to
`Driver`, after eighteen use cases. See
[the rules it is written under](/soma/model/philosophy/).

## The unit of work is a slice, not a layer

A change reaches all the way to Python or it is not finished. `soma-core`
alone is not a deliverable; a use case is.

Every one of them is written down at the moment it closes, in
[the use cases](/soma/use-cases/) — what was decided, what it was decided
*over*, and what would have to change for the answer to be different. That is
the design record, and it is the only one: a second hand-written decision page
is two records that slowly stop agreeing.

## Branches and commits

One branch per slice, named after what it does rather than what it touches:

```
feat/a-kept-value-says-where-it-came-from
refactor/codec-out-of-transport
```

`main` is the default branch and merges are fast-forward, so the individual
commit messages survive. `legacy-0.5` is the published history of the version
this replaces; it stays where it is and is never merged.

Commit subjects are `type: what changed, and why` — conventional commits, but
the half after the colon is a sentence and not a filename:

```
fix: the worker image is called by the release, because an event it caused raises nothing
feat: a move is reached by a name somebody chose, not by the number the store handed out
refactor: soma is a framework, and a framework does not mount a server
```

The types in use are `feat`, `fix`, `refactor`, `docs`, `chore` and `test`,
optionally scoped (`feat(python)`). A subject that only says *what* is a subject
whose reason has to be reconstructed from the diff a year later.

## Before pushing

```bash
uv run cargo test --workspace
uv run cargo clippy --workspace --all-targets -- -D warnings
uv run cargo fmt --all -- --check
conda activate mos && cd soma-python && maturin develop && python -m pytest tests/ -q
```

The first three can be a hook:

```bash
ln -s ../../.github/hooks/pre-commit .git/hooks/pre-commit
```

And read [how it is tested](/soma/project/how-it-is-tested/) before trusting a
green run — there are four documented ways these go green without being green,
and each of them has happened.

## Where a thing goes

**One file per type.** The type, its inherent `impl`s and the errors its
operations produce, together. An inherent `impl` is **never** split across
files: if it feels like it should be, the operation was probably not a method of
that type.

A family of types gets a folder with a `mod.rs` — never a `family.rs` sitting
next to a `family/`. The folder is created when the family already has members,
never in anticipation. Nested folders per concept are how
`soma-runtime/src/optimizer/sampler/mod.rs` happened in the old version.

**The core knows nothing about Python.** `#[pyclass]` does not go in
`soma-core`. The moment a core type carries one it can no longer be used
without an interpreter loaded. `soma-python/` translates; a domain rule written
there is in the wrong place.

**Comments earn their place.** They say *why*, not *what*, and the header of a
module is pruned like everything else.

## Adding a dependency

The bar is high and it is measured, not argued. Three crates in this workspace
have **no dependencies at all** — the core, `study` and `health` — and for
`health` that is what makes an invariant testable rather than aspirational.

`data/` takes Arrow and leaves the expression engine to whoever wants one.
There is no async runtime anywhere, and that is what keeps SQL out. See
[the crates](/soma/reference/crates/) for what each one costs today.

## The documentation

The site is this directory, built with Starlight:

```bash
npm --prefix docs install
npm --prefix docs run dev     # http://localhost:4321/soma/
npm --prefix docs run check   # guards + production build
```

Each guard exists because something failed silently once: every page must be in
the sidebar; every internal link must carry the `/soma` base, land on a page
that exists **and**, if it has a `#`, reach a heading; every `file:line` anchor
must resolve; and every mermaid fence must parse with no backtick in its
caption — a fence's info string may not contain one, so a caption with a
backtick is not a fence at all and the diagram ships as literal text.

Three groups are **generated and not committed** — the thirteen tutorials from
`examples/`, the use cases from `docs/use-cases.md`, and the Python reference
from the package's own docstrings. The source is the truth, and a committed
copy would eventually disagree with it. Adding a notebook to `examples/` is
deliberately **not** enough to publish it: it has to be given a place in the
sidebar, and the guard fails until it is.

The **figures** on the hand-written pages are committed for the same reason:
drawing one needs torch, plotly and kaleido. `python docs/scripts/figures.py`
redraws all of them from real runs, with fixed seeds. One group also needs
`cargo build --release -p somatize-tree`, because the reasoning DAG's moves are
written by the CLI; it says so and stops rather than skipping. Astro fails the build if a page points at a figure that is not there, so
they can go stale but not missing.

The reference has one step the others do not, because reading a docstring needs
the extension built and the site builds with a bare `python3`:

```bash
python docs/scripts/python_surface.py     # after editing any public docstring
```

That rewrites `docs/python-surface.json`, which **is** committed, and CI runs
the same script with `--check` in the job that has just installed the package.
`npm run check` cannot catch a stale dump — it never imports somatize.

## Adding a notebook

They ship **executed, with their outputs**, so opening one shows what it does.
Re-running the ones that report a timing needs a release build — a debug
extension writes numbers about ten times worse into the record. The details are
in [`examples/README.md`](https://github.com/manucouto1/soma/blob/main/examples/README.md).
