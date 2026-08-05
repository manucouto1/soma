# somatize-worker

The worker daemon: runs plans, isolates Python, and moves data between machines.

A `LocalRunner` that listens on a port. Python filters execute in a **child
subprocess**, so the GIL is completely isolated from Rust and Tokio.

`EnvManager` keeps one isolated venv or conda environment per pipeline and
updates it incrementally — it hashes the requirements and installs,
upgrades or removes only what changed. Set `SOMA_LOCAL_PACKAGE` when
developing against a working tree, or the worker installs the last
*released* Soma and runs code different from the one that pickled the
filters.

Remote streaming drives the runtime's `StreamRun` rather than
reimplementing it, and the serialized plan carries the run seed so remote
cache keys are salted like local ones.

---

Part of [**Soma**](https://github.com/manucouto1/soma), a computational
graph runtime for research pipelines, agent orchestration and data
virtualization. Most users want the [`somatize`](https://crates.io/crates/somatize)
facade or the [Python package](https://pypi.org/project/somatize/) rather
than this crate on its own.

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize-worker>

Licensed under the Elastic License 2.0.
