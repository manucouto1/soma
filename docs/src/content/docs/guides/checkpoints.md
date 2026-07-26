---
title: Checkpoints
description: Save and restore Soma graphs — full bundle (topology + weights) or state-only.
---

Soma graphs can be persisted in two granularities:

- **State-only** — `graph.state()` / `graph.load_state(sd)`. Snapshot of
  per-node runtime state. Cheap, in-process. Good for transfer learning,
  debugging, and swapping topology around fixed weights.
- **Full checkpoint** — `graph.save(path)` / `Graph.load(path)`. Zip
  bundle on disk: topology + per-node state + manifest. Re-creates the
  graph from scratch, no original Python objects required.

`save()` should be called after [`graph.freeze()`](../design/gradients/#native-training-loop-python)
so every `DifferentiableFilter._module` has its weights pushed into the
runtime state library first.

## State-only round-trip

```python
sd = g.state()                  # {node_id: state_value}
torch.save(sd, "states.pt")     # users pick their own serialiser

# Later, on a graph with the same topology:
g2 = build_same_topology()
g2.load_state(sd, strict=True)  # strict: every key must map to a node
```

`strict=False` allows partial overlap and reports the rest via
`warnings.warn`:

- keys in `sd` not present in `g2` → warned (or raised if `strict=True`)
- nodes in `g2` not covered by `sd` → warned (only when `strict=False`)

## Full checkpoint

```python
g.freeze()
g.save("ckpt.somack")           # zip bundle, ~1 KB + weights

# Anywhere with the filter classes importable:
g2 = Graph.load("ckpt.somack")
g2.eval()
preds = g2.forward(x_test)
```

`g.save(path, include_optimizer=True)` also bundles the optimiser
state. Restore happens lazily after a fresh optimiser is built:

```python
g2 = Graph.load("ckpt.somack")
g2.materialize(sample)
g2.train()
g2.make_optimizer(torch.optim.Adam, lr=1e-3)
applied = g2.restore_optimizer()  # True if a snapshot was bundled
```

## Bundle format

`.somack` is a regular zip archive:

```
ckpt.somack
├── manifest.json
├── states/
│   ├── <node_id>.safetensors      # tensor weights (one per node)
│   └── <node_id>.json             # non-tensor state (one per node)
└── optimizer.pt                   # optional, only with include_optimizer=True
```

### `manifest.json` schema

```jsonc
{
  "soma_checkpoint_format": 1,            // bundle format version
  "saved_at": "2026-04-26T12:00:00+00:00",
  "topology": {
    "nodes": [
      {
        "id": "encoder",
        "class_path": "myproj.filters.MyEncoder",  // import path
        "class_version": 1,                          // Filter.class_version
        "kwargs": {"hidden": 64, "lr": 1e-3}        // ctor args
      },
      // ...
    ],
    "edges": [
      {"source": "encoder", "target": "classifier"},
      // ...
    ]
  }
}
```

Tensor weights go to safetensors (memmap-friendly, no code execution on
load). Anything non-tensor in the runtime state goes to JSON next to it.
The optimiser, when bundled, uses `torch.save` / `torch.load`
(`weights_only=True`) — that path only runs in trusted environments
where the user already trusts the loaded filter classes.

## Versioning

Each `Filter` subclass declares `class_version: int = 1`. Bump it when
constructor kwargs or saved-state layout change in a non-backwards-compatible
way:

```python
class MyEncoder(Filter):
    class_version = 2  # was 1; renamed kwarg `dim` → `hidden`

    def __init__(self, hidden, lr=1e-3):
        super().__init__(hidden=hidden, lr=lr)
```

`Graph.load` compares the manifest's `class_version` against the
current code:

- `strict=True` (default) → mismatch raises `RuntimeError`.
- `strict=False` → warns and best-effort restores.

For migrations that need to rewrite kwargs or state, write a small
shim:

```python
def migrate(manifest_node):
    if manifest_node["class_version"] == 1:
        manifest_node["kwargs"]["hidden"] = manifest_node["kwargs"].pop("dim")
        manifest_node["class_version"] = 2
    return manifest_node
```

…then load the manifest manually, apply `migrate` to each node, and
feed the result through `Graph.load_state` after rebuilding the
topology.

## Filter contract

For a filter to round-trip cleanly:

1. **Constructor kwargs must be JSON-serialisable.** `Filter.kwargs()`
   returns the dict captured by `super().__init__(**kwargs)`. If your
   filter needs runtime objects (DB connections, large lookup tables,
   lazy embeddings), don't pass them as kwargs — build them inside
   `build_module` / `forward` from a serialisable seed (e.g. a path or
   a config dict).

2. **Class importable by `class_path`.** `Graph.load` does
   `importlib.import_module(...)`. Inline classes defined in a script
   work in the original process (no save needed) but cannot be
   reconstructed from a checkpoint elsewhere — keep filter classes in a
   stable module path.

3. **Override `kwargs()` if you need to drop fields.** Default behaviour
   reflects exactly what was passed to `__init__`. If a kwarg holds a
   non-serialisable object that you reconstruct inside, override
   `kwargs()` to return only the serialisable subset.

```python
class WithCallback(Filter):
    def __init__(self, hidden, on_event):
        super().__init__(hidden=hidden)   # don't store the callback
        self._cb = on_event               # attach separately

    def kwargs(self):
        return {"hidden": self.hidden}    # callback is reattached at runtime
```

## When to use which

| Need | API |
|---|---|
| Serve a trained pipeline elsewhere | `g.save(path)` → `Graph.load(path)` |
| Resume training from a checkpoint | `g.save(path, include_optimizer=True)` → load + `restore_optimizer()` |
| Transfer weights to a different topology | `g.state()` → build new graph → `g2.load_state(sd, strict=False)` |
| Debugging / inspecting a snapshot | `g.state()` (returns a plain dict) |
| Swap weights without rebuilding | `g.load_state(sd)` on an existing `g` |
