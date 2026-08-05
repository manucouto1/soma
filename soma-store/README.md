# somatize-store

Remote `DataStore` backends (S3, Zarr), feature-gated and off by default.

Split out of `somatize-core` for one reason: each backend owns a tokio
runtime, and core's promise is that it drags none in. A caller who needs
S3 pays for it explicitly.

Both backends are behind features and off by default, so depending on this
crate without enabling one costs almost nothing.

---

Part of [**Soma**](https://github.com/manucouto1/soma), a computational
graph runtime for research pipelines, agent orchestration and data
virtualization. Most users want the [`somatize`](https://crates.io/crates/somatize)
facade or the [Python package](https://pypi.org/project/somatize/) rather
than this crate on its own.

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize-store>

Licensed under the Elastic License 2.0.
