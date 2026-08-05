# somatize-agent

The research loop as a `Step`: propose an experiment, run it, read the metrics, conclude.

`ResearchStep` is a single step whose actions are `RunExperiment` or
`Conclude`. Running one means emitting `Effect::Graph` — a pipeline is a
first-class tool for an agent — reading the metrics back, and deciding
whether to propose another.

Small on purpose: the interesting parts are the effect journal in
`somatize-runtime` and the experiment pool in `somatize-memory`.

---

Part of [**Soma**](https://github.com/manucouto1/soma), a computational
graph runtime for research pipelines, agent orchestration and data
virtualization. Most users want the [`somatize`](https://crates.io/crates/somatize)
facade or the [Python package](https://pypi.org/project/somatize/) rather
than this crate on its own.

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize-agent>

Licensed under the Elastic License 2.0.
