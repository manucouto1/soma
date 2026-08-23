# Examples

Four notebooks, in order. Each one runs on its own and needs nothing
downloaded — the data is made up in the cell that uses it, on purpose: an
example that fetches a dataset is an example that stops working.

| | what it is about |
|---|---|
| [1 — Declaring a graph](01-declaring-a-graph.ipynb) | the DSL, `.on()` / `.at()` / `.cached()` / `.frozen()` / `.mapped()`, and the figure a graph draws of itself before it has ever run |
| [2 — Watching a run](02-watching-a-run.ipynb) | `watching=`, a `Recorder`, reading a run back, `progress` / `spent` / `Live`, the cache seen, and a node on a real worker |
| [3 — Training](03-training.ipynb) | a `Trainer`, `Opaque`, the loss drawn live, gradient accumulation, freezing, and exporting what a run learnt |
| [4 — A study](04-a-study.ipynb) | `Space` / `Sampler` / `Pruner`, the distributed loop, a table of results, hyper-parameter influence and parallel coordinates |
| [5 — The health of a network](05-the-health-of-a-network.ipynb) | `auditing=`, each pathology built and caught, and the invariant: a diagnosis taken from the record, argued with by moving a bound, and taken again |

They are shipped **with their outputs**, so opening one shows what it does
without running anything.

Every figure is stored twice, and that is on purpose: the plotly JSON, so
JupyterLab and nbviewer draw it live and you can hover a node or zoom a curve;
and a PNG beside it, so a static viewer — GitHub, a diff, a preview — shows the
same figure instead of an empty cell. It is the renderer that decides:

```bash
PLOTLY_RENDERER="plotly_mimetype+png" jupyter lab      # or when re-executing them
```

## Running them

```bash
cd python && maturin develop          # the tests and the notebooks both run
pip install 'soma-next[viz]'          # plotly, for every figure here
pip install ipywidgets                # optional: `Live` redraws in place with it
```

`maturin develop` is not optional: the notebooks run against the **installed**
extension, so a change in `python/src/` that was not rebuilt means a notebook
that is green about code that is not the code.

Notebooks 3, 4 and 5 need `torch`. Notebook 2 starts a real worker process, which
needs nothing but the same interpreter. Notebooks 3 and 4 seed torch, so
re-executing them gives back the numbers that are stored here.

To re-execute them all after a change to the Python API:

```bash
python - <<'EOF'
import os, pathlib, nbformat
from nbclient import NotebookClient
os.environ["PLOTLY_RENDERER"] = "plotly_mimetype+png"
for path in sorted(pathlib.Path("examples").glob("*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3").execute()
    nbformat.write(nb, path)
    print(path.name, "ok")
EOF
```

## What is not here, and why

**The static half of health.** Notebook 5 diagnoses what *happened*, which needs
a training run. What a graph can be told about itself **before a GPU is spent** —
signal propagation at init, where a normalisation layer is missing, the zero-cost
proxies that rank architectures without training them — is the next slice. It is
a different question with different literature behind it, and pretending the two
are one is how a framework ends up scoring an architecture with a number that
mostly measures its parameter count.

**The overlay**: the graph figure from notebook 1 coloured by what notebook 5's
flags say. It needs a channel of its own, because in every other figure here hue
says *where a node runs* and never good-or-bad.
