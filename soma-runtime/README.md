# somatize-runtime

Execution: `GraphSession`, the node catalog, caching, streaming, search and training strategies.

Where things actually run. `GraphSession` is the primary orchestrator —
graph plus catalog plus cache plus events.

`NodeCatalog` is *the* registry: every node, filter or step, and the
trained states. Filters and steps used to live in two registries joined by
an adapter a caller had to remember to build, which is how `.compile()`
came to skip every step's schema while `.run()` checked them.

`run_node` is the one execution site — input resolution, the output cache,
`catch_unwind` and the start/complete/fail events happen once for filters
and steps alike. Its three primitives are what `StreamRun` composes per
chunk, so streaming is the same execution site rather than a parallel one.

Also here: LRU/local/tiered caches, the Grid/Random/Bayesian samplers,
Median/Percentile pruners, `StudyRunner`, `PbtRunner`, the effect driver
and journal, and the training strategies (local, data-parallel,
model-parallel and federated all run).

---

Part of [**Soma**](https://github.com/manucouto1/soma), a computational
graph runtime for research pipelines, agent orchestration and data
virtualization. Most users want the [`somatize`](https://crates.io/crates/somatize)
facade or the [Python package](https://pypi.org/project/somatize/) rather
than this crate on its own.

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize-runtime>

Licensed under the Elastic License 2.0.
