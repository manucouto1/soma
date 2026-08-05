# somatize

The facade: one dependency that re-exports the workspace.

Start here. This crate pulls in the core types, the compiler, the runtime,
the memory layer and the rest, and re-exports them under one name, so a
`Cargo.toml` needs one line rather than eight.

```toml
[dependencies]
somatize = "0.5"
```

Reach for an individual crate only when you want less than all of it — for
example `somatize-core` alone, which carries no runtime and no network.

---

[**Soma**](https://github.com/manucouto1/soma) is a computational graph
runtime for research pipelines, agent orchestration and data
virtualization. If you would rather work in Python, the same runtime ships
as [`somatize` on PyPI](https://pypi.org/project/somatize/).

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize>

Licensed under the Elastic License 2.0.
