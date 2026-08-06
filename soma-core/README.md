# somatize-core

Types, traits and serialization. No runtime, no network, no optional heavy dependency.

The contracts every other crate agrees on: `Filter` (`fit` learns state,
`forward` transforms — both independently cacheable), `Step`, `Value`,
`Graph`, `Event`, `Schema`, `VirtualValue`, `Search`, `Study`,
`Effect`/`Transition`, `Message`/`ContentBlock`, `ToolSpec`,
`LoopCondition`, `TrainingStrategy` and the `DataStore` trait.

The rule is *no runtime, no network, no optional heavy dep* — **not** "no
I/O". `LocalDataStore` and its `std::fs` stay, because they cost a caller
nothing. Verify the rule holds with:

```sh
cargo tree -p somatize-core | grep tokio   # empty
```

Graph visualization lives here too (`to_mermaid`, `to_text`,
`to_svg`), because it is pure data → string and needs no runtime.

---

Part of [**Soma**](https://github.com/manucouto1/soma), a computational
graph runtime for research pipelines, agent orchestration and data
virtualization. Most users want the [`somatize`](https://crates.io/crates/somatize)
facade or the [Python package](https://pypi.org/project/somatize/) rather
than this crate on its own.

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize-core>

Licensed under the Elastic License 2.0.
