"""What the compiled extension exposes.

`_soma` is a Rust extension module, so nothing that reads Python source —
mypy, an IDE, `help()` on an unimported name — can see inside it. This file
is that surface written down.

It is maintained by hand, which means it can lie. `tests/test_stubs.py`
checks it against the module that was actually built: same classes, same
methods and attributes on each, same parameter names in the same order,
same parameters carrying defaults. What that cannot check is whether a type
is *right* — that part is on whoever edits the Rust, and on review.

Two conventions worth knowing before editing:

* Defaults are written `= ...` even where the real default is `None` or
  `"inference"`. That is the stub convention: the value lives in the Rust,
  and repeating it here would be a second place to keep in sync.
* A class with no `#[new]` gets no `__init__` here. `Trial`, `Run` and
  `StepCtx` arrive as arguments and raise `TypeError` if you call them;
  declaring a constructor would bless code that cannot run.

The Python layer on top (`soma.Graph`, `soma.Study`, `soma.RunView`) is
ordinary source and needs no stub — it is annotated in place.
"""

from collections.abc import Callable
from typing import Any, Protocol, TypedDict, overload

__version__: str

class _SearchDim(Protocol):
    """Whatever `search(...)` returns.

    The Rust duck-types this: an argument is treated as a search dimension
    if it has both `to_dict` and `field_name`, otherwise it is passed
    through as a value. A Protocol says that; importing `soma.search` would
    also point this file at the pure-Python layer sitting above it.
    """

    field_name: str | None
    def to_dict(self) -> dict[str, Any]: ...

# ── Exceptions ──────────────────────────────────────────────────────

class SomaSuspended(RuntimeError):
    """A run stopped to wait for something outside it; resume with `Graph.resume`."""

class SomaPruned(RuntimeError):
    """A trial was stopped early by the study's pruner. Control flow, not failure."""

class SomaSchemaMismatch(RuntimeError):
    """Two connected nodes disagree about dtype or shape, caught at compile time."""

class SomaNodeNotFound(RuntimeError):
    """A node id that nothing in the graph answers to."""

# ── The graph ───────────────────────────────────────────────────────

class Graph:
    """A computational graph: the one user-facing API."""

    py_state: dict[str, Any]

    def __new__(
        cls,
        *,
        cache: str | None = ...,
        cache_path: str | None = ...,
        cache_max_bytes: int | None = ...,
    ) -> Graph:
        """`cache` is "memory" | "local" | "tiered"; the default is a persistent tiered cache."""

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    def _repr_html_(self) -> str:
        """Notebook display: the architecture as an inline SVG (a text tree past 80 nodes)."""

    def add_mcp_server(self, command: str, args: list[str] | None = ...) -> list[str]:
        """Start an MCP server and make its tools callable; returns the tool names found."""

    def add_tool(self, tool: Tool) -> None:
        """Register a tool without attaching it to a particular agent."""

    def add_worker(
        self, address: str, token: str | None = ..., tags: list[str] | None = ...
    ) -> None:
        """Register a remote worker to connect to directly."""

    def begin_run(
        self,
        name: str,
        root: str = ...,
        kind: str = ...,
        tags: list[str] | None = ...,
        params: dict[str, Any] | None = ...,
        parent: str | None = ...,
        hypothesis: str | None = ...,
    ) -> Run:
        """Start a tracked run: creates the run dir, snapshots the topology, attaches the sink."""

    def branch(
        self,
        node_id: str,
        condition: Any,
        arms: dict[str, Any],
        target: str | None = ...,
    ) -> str:
        """Add a routing node: runs `condition` and executes only the arm its output names."""

    def compile(self, mode: str = ...) -> dict[str, Any]:
        """Compile and return diagnostics; mode is "inference" | "differentiable" | "no_cache"."""

    def connect(self, source: str, target: str) -> None:
        """Alias for `edge`."""

    def edge(self, source: str, target: str) -> None:
        """Connect two nodes with a data edge."""

    def edges(self) -> list[tuple[str, str]]:
        """The data edges, in insertion order."""

    def emit_event(self, event: dict[str, Any]) -> None:
        """Emit an event onto the graph's bus; needs `event_type` plus that variant's fields."""

    def filter(self, node_id: str) -> Any | None:
        """The live Python filter registered under `node_id`, or None."""

    def filter_ids(self) -> list[str]:
        """Node ids that have a live Python filter, in topological order."""

    def filter_requirements(self, node_id: str) -> list[str] | None:
        """Distributions a worker must install to run this node."""

    def filter_source(self, node_id: str) -> str | None:
        """The module source captured for a filter node."""

    def filter_sources_dict(self) -> dict[str, str]:
        """Every captured filter source, keyed by node id."""

    def filters(self) -> list[tuple[str, Any]]:
        """The live Python filters as (node_id, filter), in topological order."""

    def fit(
        self,
        x: Any,
        y: Any | None = ...,
        batch_size: int | None = ...,
        mode: str = ...,
        seed: int | None = ...,
    ) -> None:
        """Fit every trainable filter; mode is "inference" | "differentiable"."""

    def forward(
        self,
        x: Any,
        stream: bool = ...,
        chunk_size: int = ...,
        seed: int | None = ...,
        run_id: str | None = ...,
    ) -> Any:
        """Push data through the graph and return the output of the leaf that ran."""

    def get_node_state(self, node_id: str) -> Any | None:
        """The stored state of a filter node."""

    def graph_json(self) -> str:
        """The topology as JSON."""

    def handoff(self, source: str, target: str) -> None:
        """Declare that `source` may hand control to `target` — what `Goto` needs."""

    def loop(
        self,
        node_id: str,
        body: Any,
        until: str | bool | None = ...,
        max_iterations: int | None = ...,
    ) -> str:
        """Repeat a body until it signals completion; `until=False` runs the full count."""

    def mark_fitted(self) -> None:
        """Mark the graph fitted without running `fit`."""

    @overload
    def node(self, filter: Any, /, *, target: str | None = ...) -> str:
        """Add a node and name it automatically. Returns the node id."""

    @overload
    def node(self, id: str, filter: Any, /, *, target: str | None = ...) -> str:
        """Add a node under an explicit id. Returns it."""

    def on_event(self, callback: Callable[[dict[str, Any]], object]) -> None:
        """Receive each event as a dict, on a background thread."""

    def optional(self, source: str, target: str) -> None:
        """Make an existing edge part of the search space: a study may keep it or cut it."""

    def optional_edges(self) -> list[tuple[str, str]]:
        """The edges a study is allowed to cut."""

    def register_graph(self, sub: Graph) -> None:
        """Make `sub`'s node implementations runnable by this graph's steps (soma.agentic.RunGraph)."""

    def register_step(self, step_id: str, obj: Any) -> str:
        """Register a spawn target without adding a node. Returns the id `Spawn` should name."""

    def resume(
        self, run_id: str, node_id: str, turn: int, reason: dict[str, Any], answer: Any
    ) -> None:
        """Answer what a suspended run was waiting for; every argument comes off `SomaSuspended`."""

    def run(self) -> dict[str, Any]:
        """Compile and execute, returning every node output."""

    def set_coordinator(self, url: str, token: str | None = ...) -> None:
        """Discover workers through a coordinator instead of naming them."""

    def set_data_store(
        self,
        store_type: str,
        path: str | None = ...,
        bucket: str | None = ...,
        prefix: str | None = ...,
        endpoint: str | None = ...,
        access_key: str | None = ...,
        secret_key: str | None = ...,
        cache_dir: str | None = ...,
    ) -> None:
        """Move data between workers through a store; `store_type` is "local" | "s3"."""

    def set_edge(self, source: str, target: str, enabled: bool) -> None:
        """Keep or cut one of the optional edges, restoring it whole and in place."""

    def set_node_state(self, node_id: str, state: Any) -> None:
        """Store a state value for a filter node."""

    def shutdown_worker(self, address: str, reason: str | None = ...) -> None:
        """Shut one worker down."""

    def shutdown_workers(self, reason: str | None = ...) -> None:
        """Shut every registered worker down."""

    def steps(self) -> list[tuple[str, Any]]:
        """The live object behind each step node, as (node_id, obj), sorted by id."""

    def to_graphviz(self, overlay: dict[str, Any] | None = ...) -> str:
        """Graphviz DOT, optionally annotated per node."""

    def to_mermaid(self, overlay: dict[str, Any] | None = ...) -> str:
        """A Mermaid diagram, optionally annotated per node."""

    def to_svg(self, overlay: dict[str, Any] | None = ...) -> str:
        """A self-contained SVG, with no JavaScript — notebooks strip that."""

    def to_text(self) -> str:
        """An ASCII tree."""

    def use_provider(self, provider: str) -> None:
        """Set the provider that serves model names given without a prefix."""

    def workers(self) -> list[dict[str, Any]]:
        """Known workers, whether added directly or found through a coordinator."""

# ── The agentic layer ───────────────────────────────────────────────

class Agent:
    """A ReAct agent: ask a model, run the tools it asks for, repeat."""

    def __new__(
        cls,
        model: str | _SearchDim,
        system: str | _SearchDim | None = ...,
        tools: list[Tool] | None = ...,
        max_turns: int | _SearchDim | None = ...,
        max_tokens: int | _SearchDim | None = ...,
        effort: str | _SearchDim | None = ...,
        text_only: bool = ...,
        schema: dict[str, Any] | None = ...,
        max_repairs: int = ...,
    ) -> Agent:
        """`model` is `provider/name`, or a bare name when the graph has a default provider."""

    model: str
    system: str | None
    max_turns: int
    max_tokens: int | None
    effort: str | None
    schema: dict[str, Any] | None
    tools: list[Tool]
    """A fresh list each time — mutating it does not change the agent."""

    def __repr__(self) -> str: ...
    def search_space(self) -> list[dict[str, Any]]:
        """The dimensions this agent contributes to a study."""

class Judge:
    """Grade something with a model against a rubric."""

    def __new__(
        cls,
        model: str | _SearchDim,
        rubric: str | _SearchDim,
        threshold: float | _SearchDim | None = ...,
    ) -> Judge:
        """How strictly to grade is a real thing to tune, and so is the wording of the rubric."""

    model: str
    rubric: str
    threshold: float

    def __repr__(self) -> str: ...
    def search_space(self) -> list[dict[str, Any]]:
        """The dimensions this judge contributes to a study."""

class Tool:
    """A tool backed by a Python callable."""

    def __new__(
        cls,
        func: Callable[..., Any],
        description: str | None = ...,
        name: str | None = ...,
        schema: dict[str, Any] | None = ...,
    ) -> Tool:
        """`schema` is JSON Schema for the arguments; without it one is derived from the signature.

        Raises `ValueError` when it can find neither a name nor a description —
        an undocumented lambda type-checks here and fails at runtime.
        """

    name: str
    description: str
    schema: dict[str, Any]

    def __repr__(self) -> str: ...

class StepCtx:
    """What a step's `poll` is handed: this turn, and everything before it."""

    node_id: str
    run_id: str
    turn: int
    input: Any
    results: list[dict[str, Any]]
    """What last turn's effects returned, in the order they were requested."""
    history: list[list[dict[str, Any]]]
    """Every turn's results, oldest first."""

    def __repr__(self) -> str: ...
    def result(self) -> dict[str, Any] | None:
        """The single result of a one-effect turn; None on turn 0."""

@overload
def tool(
    func: Callable[..., Any],
    *,
    description: str | None = ...,
    name: str | None = ...,
    schema: dict[str, Any] | None = ...,
) -> Tool:
    """Decorator form of `Tool`, used bare: `@tool`."""

@overload
def tool(
    func: None = ...,
    *,
    description: str | None = ...,
    name: str | None = ...,
    schema: dict[str, Any] | None = ...,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator form of `Tool`, called: `@tool(description=...)`."""

def providers() -> list[dict[str, Any]]:
    """The providers Soma knows about, and whether each is usable right now."""

def models(provider: str) -> list[str]:
    """The models a provider currently offers. Reaches the network."""

# ── Studies ─────────────────────────────────────────────────────────

class Study:
    """A search: a space, a strategy, and the trials it produced."""

    def __new__(
        cls,
        name: str,
        search_space: list[dict[str, Any]] | None = ...,
        strategy: str = ...,
        n_trials: int = ...,
        objectives: list[tuple[str, str]] | None = ...,
        seed: int | None = ...,
        objective: Callable[[dict[str, float]], float] | None = ...,
        direction: str = ...,
        pruning: str | tuple[str, int] | tuple[str, float, int] | None = ...,
        tracking: bool = ...,
        root: str = ...,
        tags: list[str] | None = ...,
        seeds: list[int] | None = ...,
        frozen: dict[str, Any] | None = ...,
    ) -> Study:
        """`seed` seeds the sampler; `seeds` are the experiment seeds to repeat each trial over."""

    name: str
    run_dir: str | None
    progress: float
    objectives: list[tuple[str, str]]
    """(metric, direction) pairs. Not a mirror of the constructor argument: a
    Python `objective=` collapses to a single `("score", direction)`."""
    n_trials: int
    """Trials recorded *so far* — not the constructor's planned count."""
    best_trial: dict[str, Any] | None
    trials: list[dict[str, Any]]
    """Records, not `Trial` objects: `study.trials[0]["params"]`, not `.params`.
    The `Trial` class is what an executor is handed; this is what was kept."""

    @staticmethod
    def load(
        run_dir: str, objective: Callable[[dict[str, float]], float] | None = ...
    ) -> Study:
        """Read a study back from a run directory.

        Returns the extension class, not `soma.Study` — so `progress=`, the
        `plot_*` methods and `trials_dataframe` are not on what comes back.
        """

    def run(
        self,
        executor: Callable[[Trial], dict[str, float] | float | None],
        on_event: Callable[[dict[str, Any]], object] | None = ...,
        resume: bool = ...,
    ) -> None:
        """`executor` gets a `Trial` and returns metrics, a bare number, or None."""

    def save(self, path: str | None = ...) -> None:
        """Write the study as JSON; defaults to `study.json` in its run dir."""

class Trial:
    """One point in the search space. Arrives as the executor's argument."""

    id: str
    params: dict[str, Any]
    """A fresh dict each time — mutating it does not change the trial."""

    def __getitem__(self, key: str) -> Any: ...
    def __contains__(self, key: str) -> bool: ...
    def get(self, key: str, default: Any = ...) -> Any:
        """The sampled value of `key`, or `default` when the space has no such dimension."""

    def keys(self) -> list[str]:
        """The names of the sampled parameters."""

    def report(self, name: str, value: float, step: int) -> bool:
        """Report an intermediate metric. True means the pruner decided against this trial."""

    def should_prune(self) -> bool:
        """Whether the pruner has decided to stop this trial."""

# ── Tracked runs ────────────────────────────────────────────────────

class Run:
    """A tracked run, bound to a graph's event bus."""

    id: str
    dir: str
    """Where the run is being written. Relative when the pool root was."""

    def finish(self, status: str = ...) -> None:
        """Finalize, detach the sink, append the summary to the journal. Idempotent."""

    def heartbeat(self) -> None:
        """Refresh the run's liveness for anything reading it from outside."""

    def log(
        self, name: str, value: float, step: int | None = ..., node: str | None = ...
    ) -> None:
        """Log a scalar, optionally scoped to a node."""

    def log_epoch(self, epoch: int, total: int | None = ...) -> None:
        """Mark the start of an epoch."""

    def log_epoch_completed(self, epoch: int, metrics: dict[str, float] | None = ...) -> None:
        """Mark the end of an epoch, with its summary metrics."""

    def step_completed(self, step: int, epoch: int | None = ...) -> None:
        """Mark one optimizer step."""

# ── Workers ─────────────────────────────────────────────────────────

class Worker:
    """A worker, startable from Python."""

    def __new__(
        cls,
        port: int = ...,
        tags: list[str] | None = ...,
        token: str | None = ...,
        cpus: int | None = ...,
        memory: int | None = ...,
        gpus: int | None = ...,
        max_concurrent: int = ...,
        worker_id: str | None = ...,
        coordinator: str | None = ...,
    ) -> Worker:
        """`memory` is bytes."""

    def __repr__(self) -> str: ...
    def info(self) -> str:
        """A one-line summary of what this worker can run."""

    def serve(self) -> None:
        """Serve requests. Blocks until shut down, releasing the GIL."""

# ── The persistent cache ────────────────────────────────────────────

class _CacheStats(TypedDict):
    """Two tables: action records, kept forever, and the blobs they point at."""

    root: str
    actions: int
    unique_outputs: int
    blobs_available: int
    blobs_evicted: int
    cas_bytes: int
    pinned: int
    saved_compute_ms: int

class _GcReport(TypedDict):
    bytes_before: int
    bytes_after: int
    blobs_evicted: int
    blobs_kept: int
    pinned_blobs: int

class _VerifyReport(TypedDict):
    ok: int
    missing: int
    corrupt: int

class _PurgeReport(TypedDict):
    removed_files: int
    removed_bytes: int

def cache_stats(path: str | None = ...) -> _CacheStats:
    """What the shared cache is holding."""

def cache_gc(max_bytes: int, min_age_secs: int = ..., path: str | None = ...) -> _GcReport:
    """Evict blobs down to `max_bytes`. Action records are kept regardless."""

def cache_pin(name: str, key_hex: str, path: str | None = ...) -> None:
    """Pin an action as a GC root, under a name a human can read."""

def cache_verify(path: str | None = ...) -> _VerifyReport:
    """Check every referenced blob against its content hash."""

def cache_purge_v1(path: str | None = ...) -> _PurgeReport:
    """Remove entries left by the pre-CAS cache."""

# ── The experiment pool ─────────────────────────────────────────────

def list_runs_json(root: str = ...) -> str:
    """Every run under the pool, newest first."""

def checkout_run(run_id: str, root: str = ...) -> None:
    """Point HEAD at a run, so the next one branches from it."""

def read_head_run(root: str = ...) -> str | None:
    """What HEAD points at; None when the next run starts its own line."""

def clear_head_run(root: str = ...) -> None:
    """Detach HEAD."""

def kb_reindex(root: str = ...) -> int:
    """Rebuild the journal from the run directories. Returns the records written."""

def kb_find_similar_json(
    query: str = ...,
    like_run: str | None = ...,
    limit: int = ...,
    research_line: str | None = ...,
    tags: list[str] | None = ...,
    half_life_days: float | None = ...,
    root: str = ...,
) -> str:
    """Rank past experiments against a query."""

def kb_record_conclusion(
    run_id: str,
    notes: str,
    hypothesis: str | None = ...,
    tags: list[str] | None = ...,
    root: str = ...,
) -> str:
    """Retain a conclusion as an append-only amendment. Returns its id."""

def kb_lineage_json(run_id: str, root: str = ...) -> str | None:
    """One experiment with its ancestors and descendants; None if the id is unknown."""

def kb_diff_json(a: str, b: str, root: str = ...) -> str:
    """The move from one experiment to another, related or not."""

# ── Reading a run directory ─────────────────────────────────────────

def run_info_json(dir: str) -> str:
    """Identity and derived state for one run."""

def run_manifest_json(dir: str) -> str:
    """The run's manifest."""

def run_summary_json(dir: str) -> str:
    """One run folded down: identity, cost, metrics, conclusion."""

def run_events_json(dir: str) -> str:
    """Every parseable event, in log order."""

def run_metric_series_json(dir: str, name: str | None = ...) -> str:
    """Metric time series, optionally just one."""

def run_node_timings_json(dir: str) -> str:
    """Per-node execution spans."""

def run_cache_activity_json(dir: str) -> str:
    """Hits and misses, total and per node."""

def run_health_flags_json(dir: str) -> str:
    """Health flags with their wall time."""

def run_trial_timeline_json(dir: str) -> str:
    """Trial lifetimes. Empty for a run that is not a study."""

def run_agentic_activity_json(dir: str) -> str:
    """Agent-step activity per node: turns, tokens, effects, tools."""

def run_agentic_timeline_json(dir: str) -> str:
    """Per-effect execution spans for agent runs."""

def run_overlay_json(dir: str) -> str:
    """This run's events folded into a rendering overlay."""

# ── Diagrams ────────────────────────────────────────────────────────

def run_to_mermaid(dir: str, overlay: bool = ...) -> str:
    """The run's graph as Mermaid, annotated with how it went."""

def run_to_graphviz(dir: str, overlay: bool = ...) -> str:
    """The run's graph as Graphviz DOT, annotated with how it went."""

def run_to_svg(dir: str, overlay: bool = ...) -> str:
    """The run's graph as a self-contained SVG, annotated with how it went."""

def graph_json_to_mermaid(graph_json: str, overlay: dict[str, Any] | None = ...) -> str:
    """Render a serialized graph to Mermaid."""

def graph_json_to_svg(graph_json: str, overlay: dict[str, Any] | None = ...) -> str:
    """Render a serialized graph to a self-contained SVG."""
