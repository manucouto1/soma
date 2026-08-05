# somatize-macros

Procedural macros for Soma: `#[derive(SomaFilter)]` and `#[derive(SomaStep)]`.

Two derives, and the second is load-bearing: `#[derive(SomaStep)]` is what
gives every step its journal key, so an effect performed once is replayed
rather than re-run on resume.

`#[soma(cache_version)]` on a field participates in a filter's identity —
the canonical CBOR of its fields is the cache key, so changing a field
changes what is cached.

This crate has no runtime and no dependency on the rest of the workspace
at publish time: its `somatize-core` dev-dependency is path-only on
purpose, because a versioned one would make a publish cycle neither crate
could break.

---

Part of [**Soma**](https://github.com/manucouto1/soma), a computational
graph runtime for research pipelines, agent orchestration and data
virtualization. Most users want the [`somatize`](https://crates.io/crates/somatize)
facade or the [Python package](https://pypi.org/project/somatize/) rather
than this crate on its own.

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize-macros>

Licensed under the Elastic License 2.0.
