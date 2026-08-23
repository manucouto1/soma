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

Notebook 3 and 4 need `torch`. Notebook 2 starts a real worker process, which
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

**A health audit** — gradients dying or exploding, a layer that is saturated,
channels nobody is updating. There is no notebook for it because **there is no
such thing yet**: it is CU21, the third of the three things observability was
split into, and writing an example of it would be writing an example of code
that does not exist.

What notebook 2 and 3 show is the row below it: the **record**, which is what a
diagnosis will be made of. The line between them is not a matter of taste and it
is already written down as an invariant — *a diagnosis has to be reproducible
from the stored record, without training again* — so everything a health audit
will need is either in the record already or is the reason CU21 is a slice of
its own.

Also not here: **`ctx.saw(...)`**, a node speaking for itself. The engine can see
which node ran and how long it took; it cannot see a gradient norm, and the node
can. That is what CU21 opens with.
