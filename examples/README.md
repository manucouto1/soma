# Examples

Eleven notebooks, in order. Each one runs on its own and needs nothing
downloaded — the data is made up in the cell that uses it, on purpose: an
example that fetches a dataset is an example that stops working.

| | what it is about |
|---|---|
| [1 — Declaring a graph](01-declaring-a-graph.ipynb) | the DSL, `.on()` / `.at()` / `.cached()` / `.frozen()` / `.mapped()`, and the figure a graph draws of itself before it has ever run |
| [2 — Watching a run](02-watching-a-run.ipynb) | `watching=`, a `Recorder`, reading a run back, `progress` / `spent` / `Live`, the cache seen, and a node on a real worker |
| [3 — Training](03-training.ipynb) | a `Trainer`, `Opaque`, the loss drawn live, gradient accumulation, freezing, and exporting what a run learnt |
| [4 — A study](04-a-study.ipynb) | `Space` / `Sampler` / `Pruner`, the distributed loop, a table of results, hyper-parameter influence and parallel coordinates |
| [5 — The health of a network](05-the-health-of-a-network.ipynb) | `auditing=`, each pathology built and caught, and the invariant: a diagnosis taken from the record, argued with by moving a bound, and taken again |
| [6 — A problem, end to end](06-a-problem-end-to-end.ipynb) | the whole loop on one problem: propose an architecture, find what is wrong, fix it, check the fix — five times, and the last one is not a bug in the network at all |
| [7 — A real architecture](07-a-real-architecture.ipynb) | convolutions with residuals, a transformer stack, a recurrent cell and a bottleneck — some inside one node, some across four — with the architecture drawn inside each; three problems, each with **problem → symptoms → solution → healthy**, and two that showed nothing |
| [8 — Before a step is taken](08-before-a-step-is-taken.ipynb) | `probe`, one recorded forward that never trained, and the rule: what separates is a runaway, what ranks is a proxy |
| [9 — A fleet](09-a-fleet.ipynb) | `fleet` and `machines`, working against waited on, what only a machine can say about itself, and the idle one that writes on a clock |
| [10 — Where the data comes from](10-a-dataset.ipynb) | a source is a node, a graph handed a **coordinate** instead of a batch, what that saves the cache, the version the store already knew, and a frame crossing a wire |
| [11 — What an edit did](11-what-an-edit-did.ipynb) | `foreseen.names` / `unneeded` / `changes` / `snapshot`, an afternoon of edits answered without running any of it, the three ways a name moves, and the half a notebook cannot answer about its own cells |

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

But it installs a **debug** build, and the numbers stored here were measured
against a release one. Re-executing a notebook that reports a timing — 3, 10 and
11 — against the installed extension writes numbers about ten times worse into
the record: notebook 10's 19 MB toll went from 121 ms to 1112 ms that way, and
what gave it away was the coordinate, which hashes nothing and slowed down just
as much. Build the package once and point at it:

```bash
cargo build --release -p soma-next-python
cp -r python/python/soma_next /tmp/relpkg/ && rm /tmp/relpkg/soma_next/_soma_next.*.so
cp target/release/lib_soma_next.so /tmp/relpkg/soma_next/_soma_next.cpython-313-x86_64-linux-gnu.so
PYTHONPATH=/tmp/relpkg python -  # the loop below
```

Notebooks 3 to 7 need `torch`. Notebook 2 starts a real worker process, which
needs nothing but the same interpreter. Notebooks 3 and 4 seed torch, so
re-executing them gives back the numbers that are stored here. Notebook 11
writes a module to a temporary directory and imports it, because a class defined
in a **cell** has no source to read and so no version — which is the thing that
notebook is partly about.

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

**A graph marked with what an edit did.** Notebook 11 answers with findings per
node and stops there. `overlaid` already puts findings on a figure, but its
channel is health: the outline turns red, and red means ill. A node whose recipe
changed is not ill, so saying **where** an edit landed needs a channel of its
own rather than borrowing one that means something else.

**A run spread over real machines.** Notebook 2 starts one worker on this
machine and notebook 9 reads a fleet back out of a record. Containers on
separate hosts, a GPU among them, and a study handed out of a shared folder live
in `python/tests/cluster/`, which needs docker and is opt-in — an example that
needs a cluster is an example nobody can open.
