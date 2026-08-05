# somatize-coordinator

Worker registry and placement, with a `soma-coordinator` binary.

Workers heartbeat every 10 seconds; the coordinator reaps whoever goes
quiet.

The design decision worth knowing: `/submit` **places**, it does not
proxy. It returns a worker and takes a lease, so the tensor payload goes
client → worker directly and never through the coordinator. `/complete`
releases the lease.

---

Part of [**Soma**](https://github.com/manucouto1/soma), a computational
graph runtime for research pipelines, agent orchestration and data
virtualization. Most users want the [`somatize`](https://crates.io/crates/somatize)
facade or the [Python package](https://pypi.org/project/somatize/) rather
than this crate on its own.

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize-coordinator>

Licensed under the Elastic License 2.0.
