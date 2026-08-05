# somatize-memory

The experiment pool: records, lineage and retrieval over what has already been tried.

Every tracked run appends an `ExperimentRecord` to
`.soma/experiments.jsonl` with a deterministic conclusion, an
architecture fingerprint, and the `DerivationMove` from its parent —
VisTrails-style, where nodes are runs and edges are the changes applied to
the parent.

Retrieval is additive: `0.40·BM25 + 0.25·structural + 0.15·recency +
0.20·importance`, with importance floored at 0.6 for failures that carry a
conclusion. Dead ends have to stay retrievable, or the pool only remembers
what worked.

`KnowledgeBase` has three implementations: in-memory, file-backed, and
ChronosVector behind the `chronos` feature.

---

Part of [**Soma**](https://github.com/manucouto1/soma), a computational
graph runtime for research pipelines, agent orchestration and data
virtualization. Most users want the [`somatize`](https://crates.io/crates/somatize)
facade or the [Python package](https://pypi.org/project/somatize/) rather
than this crate on its own.

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize-memory>

Licensed under the Elastic License 2.0.
