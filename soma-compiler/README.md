# somatize-compiler

`Graph` → `ExecutionPlan`: cache resolution, schema validation, distribution.

Turns a graph into something executable, and answers what it can *before*
anything runs: schemas that cannot connect, nodes nothing reaches, a
gradient that cannot flow.

`ExecutionPlan` variants: `Sequence`, `Parallel`, `Execute`, `Step`,
`Composite`, `Loop`, `Branch`, `Stream`, `Remote`, `Empty`. Loop bodies and
branch arms are claimed by **dominance**, and a loop carries `carry_from`
separately from `until` — what a loop carries and what stops it are
different questions.

The `Scheduler` reads a plan's topology and assigns it across workers:
sequential stays together, parallel distributes, differentiable groups.

---

Part of [**Soma**](https://github.com/manucouto1/soma), a computational
graph runtime for research pipelines, agent orchestration and data
virtualization. Most users want the [`somatize`](https://crates.io/crates/somatize)
facade or the [Python package](https://pypi.org/project/somatize/) rather
than this crate on its own.

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize-compiler>

Licensed under the Elastic License 2.0.
