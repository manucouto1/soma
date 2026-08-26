# somatize

Declare a computation as a graph, run it, train it, and search over it — on this
machine or across several — with what happened written down as you go.

```bash
pip install somatize
```

```python
from somatize import Graph, Node

class Tokenise(Node):
    def forward(self, text, ctx):
        return text.lower().split()

class MeanLength(Node):
    def forward(self, words, ctx):
        return sum(len(w) for w in words) / len(words)

g = Graph.somatize(Tokenise().named("tokenise") >> MeanLength().named("mean"))
g.forward("The cat sat on the mat")   # 2.8333333333333335
```

`>>` chains, `|` fans out, and a node carries where it runs (`.at()`), whether
its result is kept (`.cached()`), and whether it learns (`.frozen()`). The graph
draws itself before it has ever run.

## What is here

The engine is Rust; the surface is Python. Nine crates, each one a thing you can
name:

| | |
|---|---|
| [`somatize-core`](soma-core) | the graph, the plan of how to run it, and the engine that walks it |
| [`somatize-store`](soma-store) | where a computed value is kept: bytes by content, names that point at them |
| [`somatize-data`](soma-data) | where the data comes from, and what it is once it arrives |
| [`somatize-study`](soma-study) | what is above one training run: a search over configurations |
| [`somatize-health`](soma-health) | whether what a run did is healthy — an opinion, and it says so |
| [`somatize-tree`](soma-tree) | what an edit did to a graph, said before anybody runs it |
| [`somatize-fabric-wire`](soma-fabric/wire) | carrying a slice of a plan to another process |
| [`somatize-fabric-broker`](soma-fabric/broker) | the name a graph gave a host, turned into a way of reaching it |

The API reference is at
[manucouto1.github.io/soma](https://manucouto1.github.io/soma/).

## Start here

[`examples/`](examples) holds twelve notebooks, **shipped with their outputs**,
so opening one shows what it does without running anything: declaring a graph,
watching a run, training, a study, the health of a network, one problem end to
end, a real architecture diagnosed in cycles, what can be said before a step is
taken, a fleet of machines, where the data comes from, what an edit did, and
where a kept value came from.

The optional halves are extras, because a graph declares, executes and trains
without either:

```bash
pip install 'somatize[viz]'      # plotly, for every figure
pip install 'somatize[remote]'   # cloudpickle, for a worker that starts empty
```

## Building it

```bash
uv run cargo test --workspace
cd soma-python && maturin develop && python -m pytest tests/ -q
```

`maturin develop` is not optional before `pytest`: the Python tests run against
the **installed** extension, so a change in `soma-python/src/` that is not
rebuilt means the suite is green about code that is not the code.

There is more, and it is opt-in because it is slow: a cluster of real containers
(`SOMA_CLUSTER=1 python -m pytest tests/cluster`) and a real bucket
(`docker compose -f soma-store/tests/docker/compose.yaml up -d`). Both are
described in [`CLAUDE.md`](CLAUDE.md).

## Licence

[Elastic License 2.0](LICENSE). The `somatize` versions up to 0.5.1 are a
different implementation, kept on the `legacy-0.5` branch.
