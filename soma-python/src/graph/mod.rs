//! `Graph` — the primary API.
//!
//! What a `Graph` *is*: the topology, the implementations behind each node
//! id, the cache, the bus, and the places it may run. What a `Graph`
//! *does* lives one module down, split by which of those it owns —
//! [`topology`], [`registry`], [`execution`], [`agentic`],
//! [`distributed`], [`tracking`], [`viz`].
//!
//! The `#[pymethods]` block below is therefore signatures and
//! documentation, and almost no code. That is not a style choice: pyo3
//! without the `multiple-pymethods` feature allows exactly one such block
//! per class, so a class this size can only be split by moving the bodies
//! out. Keeping the doc comments *here* matters — they are what
//! `help(soma.Graph)` prints.

mod agentic;
pub(crate) mod bridge;
mod distributed;
mod execution;
mod registry;
mod topology;
mod tracking;
mod viz;

use crate::prelude::*;
use crate::tracking::run::PyRun;
use distributed::RemoteWorker;
use registry::Registry;

// ── PyGraph ──

#[pyclass(name = "Graph", subclass)]
pub(crate) struct PyGraph {
    graph: Graph,
    library: NodeCatalog,
    cache: Arc<dyn somatize_core::cache::CacheStore>,
    event_bus: Arc<EventBus>,
    fitted: bool,
    /// What each registered node id actually is, in Python — see
    /// [`registry::NodeRecord`].
    nodes: Registry,
    /// Workers this graph may dispatch to, in registration order.
    workers: Vec<RemoteWorker>,
    /// Coordinator URL + token.
    coordinator: Option<(String, Option<String>)>,
    /// Optional DataStore for persistent data transport (opt-in, costs storage).
    data_store: Option<Arc<dyn somatize_core::data::store::DataStore>>,
    /// Data edges a study may cut, in declaration order.
    optional_edges: Vec<(String, String)>,
    /// Optional edges currently cut, held whole together with the position
    /// they came from, so restoring one restores its id, kind, label *and*
    /// place — a trial that cuts an edge has to leave the graph the next
    /// trial starts from byte-identical.
    cut_edges: HashMap<(String, String), (usize, Edge)>,
    /// Tools every agent in this graph may call, by name. Collected from the
    /// agents as they are added, so a tool declared once is callable by any
    /// node that lists it.
    tools: HashMap<String, PyTool>,
    /// Which provider serves a bare (unqualified) model name.
    default_provider: Option<String>,
    /// Tool sets from MCP servers. Held so the servers stay alive for the
    /// graph's lifetime — dropping a client kills its subprocess.
    mcp_toolboxes: Vec<somatize_llm::Toolbox>,
    /// Generic Python-side scratch dict for orchestration state that
    /// doesn't belong on the Rust struct (e.g. the registered optimiser).
    /// Lazily initialised on first access. PyGraph deliberately doesn't
    /// expose `__dict__`, so this dict is the supported way to attach
    /// per-graph Python state.
    py_state: Option<Py<PyDict>>,
}

impl PyGraph {
    /// The core graph, for the effect parser: `RunGraph` names a sub-graph
    /// by passing the live object.
    pub(crate) fn core_graph(&self) -> &Graph {
        &self.graph
    }
}

#[pymethods]
impl PyGraph {
    /// Create a new Graph.
    ///
    /// Optional keyword arguments:
    ///
    /// * `cache` — `"memory"` for an in-process LRU that dies with the
    ///   process, or a **directory path** for a persistent store with that
    ///   LRU in front of it. The default is the same persistent store at
    ///   `$SOMA_CACHE_DIR`, or `~/.soma/cache`, so fit states and forward
    ///   outputs survive crashes and are shared across processes and
    ///   projects.
    ///
    ///   There used to be a `cache` *kind* (`"local"`, `"tiered"`) and a
    ///   separate `cache_path`, which made three arguments out of one
    ///   question — where do I want this kept? A memory LRU in front of a
    ///   disk store costs nothing and is what the default already did, so
    ///   `"local"` was a way to ask for something strictly worse.
    /// * `cache_max_bytes` — max bytes for the in-memory LRU (default 1 GB).
    #[new]
    #[pyo3(signature = (*, cache=None, cache_max_bytes=None))]
    fn new(cache: Option<&str>, cache_max_bytes: Option<usize>) -> PyResult<Self> {
        let max_bytes = cache_max_bytes.unwrap_or(1024 * 1024 * 1024);
        let tiered = |dir: &str| -> PyResult<Arc<dyn somatize_core::cache::CacheStore>> {
            let local = FsActionStore::new(dir)
                .map_err(|e| PyRuntimeError::new_err(format!("cache init at {dir:?}: {e}")))?;
            Ok(Arc::new(TieredCache::memory_and_local(
                Box::new(MemoryCache::new(max_bytes)),
                Box::new(local),
            )))
        };
        let cache_store: Arc<dyn somatize_core::cache::CacheStore> = match cache {
            Some("memory") => Arc::new(MemoryCache::new(max_bytes)),
            Some(path) => tiered(path)?,
            // No writable cache dir (sandbox, read-only home): degrade to
            // memory-only rather than failing.
            None => match default_cache_dir() {
                Some(dir) => match tiered(&dir.to_string_lossy()) {
                    Ok(store) => store,
                    Err(_) => Arc::new(MemoryCache::new(max_bytes)),
                },
                None => Arc::new(MemoryCache::new(max_bytes)),
            },
        };

        Ok(Self {
            graph: Graph::new(),
            library: NodeCatalog::new(),
            cache: cache_store,
            event_bus: Arc::new(EventBus::new(256)),
            fitted: false,
            nodes: Registry::default(),
            workers: Vec::new(),
            coordinator: None,
            data_store: None,
            optional_edges: Vec::new(),
            cut_edges: HashMap::new(),
            tools: HashMap::new(),
            default_provider: None,
            mcp_toolboxes: Vec::new(),
            py_state: None,
        })
    }

    // ── Topology ──

    /// Add a filter node. Returns the node id.
    ///
    /// Usage:
    ///   g.node(MyFilter())                        # auto-named
    ///   g.node("scaler", MyFilter())              # explicit id
    ///   g.node(MyFilter(), target="gpu")           # route to gpu worker
    ///   g.node("model", MyFilter(), target="local") # force local execution
    #[pyo3(signature = (*args, target=None))]
    fn node(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        target: Option<String>,
    ) -> PyResult<String> {
        topology::node(self, py, args, target)
    }

    /// Connect two nodes with a data edge.
    fn edge(&mut self, source: String, target: String) {
        topology::edge(self, source, target)
    }

    /// List data edges as ``[(source, target), ...]`` in insertion order.
    ///
    /// Used by :meth:`Graph.save` to record topology in the manifest so
    /// :meth:`Graph.load` can reconstruct non-linear graphs (forks,
    /// joins) instead of falling back to a linear chain.
    fn edges(&self) -> Vec<(String, String)> {
        topology::edges(self)
    }

    /// Declare that `source` may hand control to `target`.
    ///
    /// This is what `soma.Goto(target)` needs: a handoff transfers control
    /// rather than passing data, so it is a control edge and not a data
    /// one. Declaring it is deliberate — a step that hands control
    /// somewhere the graph never said it could is an error rather than a
    /// silent jump, which is the inter-agent misalignment the multi-agent
    /// literature keeps finding.
    ///
    /// ```python
    /// g.node("triage", Triage())
    /// g.node("billing", soma.Agent(model="ollama/qwen2.5"))
    /// g.handoff("triage", "billing")   # now Goto("billing") is allowed
    /// ```
    fn handoff(&mut self, source: &str, target: &str) {
        topology::handoff(self, source, target)
    }

    /// Add a node that routes: it runs `condition`, reads the arm label out
    /// of the result, and executes only that arm.
    ///
    /// ```python
    /// g.branch("router", Classifier(), {
    ///     "billing": soma.Agent(model="ollama/llama3.2", system="Billing."),
    ///     "tech":    "tech_team",     # a node already in the graph
    ///     "default": Escalate(),
    /// })
    /// ```
    ///
    /// The arms are declared, so the compiler rejects one that no edge
    /// reaches and one that no arm declares — the silent-drop failure that
    /// the multi-agent literature files under inter-agent misalignment.
    /// An arm labelled `default` (or `else`) catches anything unmatched;
    /// without one, an unrecognised label is an error rather than a guess.
    #[pyo3(signature = (node_id, condition, arms, target=None))]
    fn branch(
        &mut self,
        py: Python<'_>,
        node_id: String,
        condition: &Bound<'_, PyAny>,
        arms: &Bound<'_, PyDict>,
        target: Option<String>,
    ) -> PyResult<String> {
        topology::branch(self, py, node_id, condition, arms, target)
    }

    /// Add a node that repeats a body until it signals completion.
    ///
    /// ```python
    /// g.node("draft", Draft())
    /// g.node("critic", soma.Judge(model="ollama/llama3.2", rubric="..."))
    /// g.edge("draft", "critic")
    /// g.loop("refine", body="draft", until="critic", max_iterations=3)
    /// ```
    ///
    /// `body` names the entry node(s); the loop owns those and everything
    /// only reachable through them.
    ///
    /// `until` says when to stop:
    ///
    /// - a node id — that node's output carries the signal: a bool,
    ///   `"done"`/`"stop"`, or a mapping with a `done` key, which is exactly
    ///   what `Judge` emits;
    /// - unset (the default) — the body's single terminal node is used, and
    ///   a body with several terminals is a compile error rather than a race;
    /// - `False` — never stop early; run the full `max_iterations`.
    ///
    /// The loop's value is its *carry*: seeded from the loop's input, then
    /// replaced after each pass by the condition node's output. That is what
    /// the body reads on the next round, so a refine loop refines instead of
    /// redrafting the same thing.
    #[pyo3(name = "loop", signature = (node_id, body, until=None, max_iterations=None))]
    fn loop_(
        &mut self,
        py: Python<'_>,
        node_id: String,
        body: &Bound<'_, PyAny>,
        until: Option<&Bound<'_, PyAny>>,
        max_iterations: Option<usize>,
    ) -> PyResult<String> {
        topology::loop_(self, py, node_id, body, until, max_iterations)
    }

    /// Make a data edge part of the search space: a study may keep it or
    /// cut it.
    ///
    /// This is topology as a hyperparameter — whether the critic should see
    /// the retriever's output at all is exactly the kind of question a
    /// search answers better than an argument does. Control edges are not
    /// eligible: they are what makes a loop a loop, not a design choice.
    ///
    /// ```python
    /// g.optional("retriever", "critic")
    /// study = g.study("shape", n_trials=20)   # gains `edge:retriever->critic`
    /// ```
    fn optional(&mut self, source: String, target: String) -> PyResult<()> {
        topology::optional(self, source, target)
    }

    /// The edges a study may cut, as `(source, target)`.
    fn optional_edges(&self) -> Vec<(String, String)> {
        self.optional_edges.clone()
    }

    /// Keep or cut one of the optional edges.
    ///
    /// A cut edge is set aside whole, so restoring it restores its id, kind
    /// and label — a trial that cuts an edge must leave the graph identical
    /// to the one the next trial starts from.
    fn set_edge(&mut self, source: String, target: String, enabled: bool) -> PyResult<()> {
        topology::set_edge(self, source, target, enabled)
    }

    // ── What a node is ──

    /// Make `sub`'s node implementations runnable by this graph's steps.
    ///
    /// A step invokes a pipeline with `soma.agentic.RunGraph(sub, ...)`. The
    /// effect carries the *structure* — that is what the journal keys on —
    /// but a graph names its nodes rather than carrying their code, so the
    /// implementations have to live somewhere the runtime can find them:
    /// here. Filters, steps and tools are merged; the same node id behind a
    /// different configuration is an error, because whichever one lost would
    /// silently answer for the other's cache entries.
    ///
    /// ```python
    /// pipeline = soma.Graph()
    /// pipeline.node("scale", Scaler())
    ///
    /// g = soma.Graph()
    /// g.node("planner", PlannerStep())   # awaits RunGraph(pipeline, ...)
    /// g.register_graph(pipeline)
    /// ```
    ///
    /// Register after the sub-graph is fully built; nodes added to it later
    /// are not seen until it is registered again.
    fn register_graph(&mut self, py: Python<'_>, sub: PyRef<'_, PyGraph>) -> PyResult<()> {
        registry::register_graph(self, py, sub)
    }

    /// Register a step that can be *spawned* but is not a node in the graph.
    ///
    /// `Spawn` names the work it wants by id, and that id is looked up in the
    /// step library — which `node()` also fills. But a node with no edges is
    /// a root, so registering a spawn target with `node()` makes it run once
    /// on the graph's own input as well, which is wasted work and a confusing
    /// reading of the diagram.
    ///
    /// ```python
    /// g.node("fanout", Planner())        # decides the width at runtime
    /// g.register_step("worker", Worker())  # spawnable, never a root
    /// ```
    ///
    /// The returned id is the one `Spawn` should name.
    fn register_step(
        &mut self,
        py: Python<'_>,
        step_id: &str,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<String> {
        registry::register_step(self, py, step_id, obj)
    }

    /// Get the full module source code for a filter node (for Nous agent
    /// introspection). Returns None if the node has no captured source.
    fn filter_source(&self, node_id: String) -> Option<String> {
        registry::filter_source(self, &node_id)
    }

    /// Third-party distributions the worker must install to run `node_id`.
    ///
    /// Detected from the filter module's imports when the node was added.
    /// This is what a remote plan ships to the worker's ``EnvManager``, and
    /// it is part of the filter's cache identity — the same code under a
    /// different dependency set can produce different results. Returns
    /// ``None`` for a node with no live Python filter.
    fn filter_requirements(&self, node_id: String) -> Option<Vec<String>> {
        registry::filter_requirements(self, &node_id)
    }

    /// Get all filter sources as a dict: {node_id: source_code}.
    fn filter_sources_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        registry::filter_sources_dict(self, py)
    }

    /// Retrieve the live Python filter instance registered under `node_id`.
    ///
    /// Returns ``None`` if the node doesn't exist or wasn't added through
    /// the Python `node()` API (e.g. nodes materialised from a serialised
    /// graph have only pickled bytes, not a live instance).
    ///
    /// Used by the in-process training path so callers can manipulate the
    /// filter directly — e.g. toggle `self.training`, read `_module`, or
    /// extract `state_dict()` — without round-tripping through a pickle.
    fn filter(&self, py: Python<'_>, node_id: String) -> Option<PyObject> {
        registry::filter(self, py, &node_id)
    }

    /// List node ids with live Python filter instances, in topological order.
    ///
    /// Falls back to insertion order if topological sort fails (e.g. the
    /// graph hasn't been validated yet — possible during construction).
    /// Callers that drive training need the topo order so output of one
    /// filter feeds the next.
    fn filter_ids(&self) -> Vec<String> {
        registry::filter_ids(self)
    }

    /// Return live Python filter instances as an ordered list of
    /// ``(node_id, filter)`` tuples in topological order.
    ///
    /// Returning a list (vs. a dict) preserves the order — callers
    /// iterating to chain forwards get inputs threaded correctly.
    fn filters(&self, py: Python<'_>) -> PyResult<PyObject> {
        registry::filters(self, py)
    }

    /// The live `Agent`/`Judge` behind each step node, as `(node_id, obj)`.
    ///
    /// The counterpart of `filters()`. A study reads their search spaces and
    /// writes sampled values straight onto them.
    fn steps(&self, py: Python<'_>) -> Vec<(String, PyObject)> {
        registry::steps(self, py)
    }

    /// Store a Python state value for a filter node.
    ///
    /// Used by ``Graph.freeze()`` (Python side) to push each live
    /// ``DifferentiableFilter`` module's serialised ``state_dict`` into
    /// the runtime's filter-state library, so subsequent eval calls go
    /// through the Rust forward path with state pre-populated.
    fn set_node_state(
        &mut self,
        py: Python<'_>,
        node_id: String,
        state: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        registry::set_node_state(self, py, node_id, state)
    }

    /// Retrieve the stored state value for a filter node, or ``None``.
    ///
    /// Mirror of :meth:`set_node_state`. Used by ``Graph.state()`` to
    /// snapshot every node's state for checkpointing.
    fn get_node_state(&self, py: Python<'_>, node_id: String) -> PyResult<Option<PyObject>> {
        registry::get_node_state(self, py, &node_id)
    }

    /// Mark the graph as fitted without running ``fit()``.
    ///
    /// The Rust ``forward`` path refuses to run on an un-fitted graph.
    /// When the user trains via the Python autograd loop (``train()`` /
    /// ``forward(x)`` / ``backward`` / ``step``) and then calls
    /// ``freeze()``, no Rust ``fit()`` ran — but state has been pushed
    /// via ``set_node_state``. ``freeze()`` calls this so the
    /// subsequent eval ``forward`` is allowed.
    fn mark_fitted(&mut self) {
        self.fitted = true;
    }

    // ── Agentic ──

    /// Set the provider that serves model names given without a prefix.
    ///
    /// ```python
    /// g.use_provider("ollama")
    /// g.node("a", soma.Agent(model="llama3.2"))   # → ollama/llama3.2
    /// ```
    fn use_provider(&mut self, provider: String) {
        self.default_provider = Some(provider);
    }

    /// Register a tool without attaching it to a particular agent.
    fn add_tool(&mut self, tool: PyTool) {
        self.tools.insert(tool.tool_name().to_string(), tool);
    }

    /// Start an MCP server and make everything it publishes callable.
    ///
    /// Returns the tool names discovered. Discovery happens now, so a
    /// misconfigured server fails here rather than mid-run.
    #[pyo3(signature = (command, args=None))]
    fn add_mcp_server(
        &mut self,
        py: Python<'_>,
        command: String,
        args: Option<Vec<String>>,
    ) -> PyResult<Vec<String>> {
        agentic::add_mcp_server(self, py, command, args)
    }

    /// Answer what a suspended run was waiting for.
    ///
    /// Every argument comes off the `SomaSuspended` exception that stopped
    /// the run, `reason` included — it is part of the journal key, so the
    /// answer has to be filed against the same pause the step described,
    /// not one reconstructed from a guess.
    ///
    /// The answer lands at the exact site the step paused. Running the
    /// graph again replays every prior effect from the record, reaches
    /// that point, and finds it waiting. There is no checkpoint file: the
    /// journal is the checkpoint.
    ///
    /// This existed in Rust and nowhere else, which meant nowhere at all —
    /// the only entry point that runs steps is this one.
    fn resume(
        &mut self,
        py: Python<'_>,
        run_id: &str,
        node_id: &str,
        turn: usize,
        reason: &Bound<'_, PyAny>,
        answer: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        agentic::resume(self, py, run_id, node_id, turn, reason, answer)
    }

    // ── Running ──

    /// Fit all trainable filters in topological order.
    ///
    /// If `batch_size` is set, the input is split into batches and each batch
    /// is processed through the entire pipeline (encoder → classifier) before
    /// moving to the next. This keeps memory bounded.
    ///
    /// If workers are registered and no node forces local, training is
    /// dispatched to a remote worker.
    #[pyo3(signature = (x, y=None, batch_size=None, mode="inference", seed=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        y: Option<&Bound<'_, pyo3::types::PyAny>>,
        batch_size: Option<usize>,
        mode: &str,
        seed: Option<i64>,
    ) -> PyResult<()> {
        execution::fit(self, py, x, y, batch_size, mode, seed)
    }

    /// Forward data through the compiled graph (inference mode).
    ///
    /// Routing:
    /// - stream=True → chunks sent via WS Binary to StreamExecutor on worker
    /// - No workers → local execution
    /// - Workers + all nodes non-local → entire plan dispatched to worker
    /// - Workers + mixed (some local) → local execution with remote fallback
    #[pyo3(signature = (x, stream=false, chunk_size=None, seed=None, run_id=None))]
    fn forward(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        stream: bool,
        chunk_size: Option<usize>,
        seed: Option<i64>,
        run_id: Option<String>,
    ) -> PyResult<PyObject> {
        execution::forward(slf, py, x, stream, chunk_size, seed, run_id)
    }

    /// Compile the graph and return diagnostic information.
    #[pyo3(signature = (mode="inference"))]
    fn compile(&self, py: Python<'_>, mode: &str) -> PyResult<PyObject> {
        execution::compile_info(self, py, mode)
    }

    // ── Visualization ──

    /// Render the graph as a Mermaid diagram string.
    ///
    /// `overlay` is an optional dict of per-node execution annotations
    /// (the shape `RunView.overlay()` returns): status coloring plus a
    /// duration/cache/flags label line per node.
    #[pyo3(signature = (overlay=None))]
    fn to_mermaid(&self, py: Python<'_>, overlay: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        viz::to_mermaid(self, py, overlay)
    }

    /// Render the graph as a self-contained SVG diagram (same optional
    /// `overlay` as `to_mermaid`). No JavaScript — displays inline in
    /// any notebook viewer.
    #[pyo3(signature = (overlay=None))]
    fn to_svg(&self, py: Python<'_>, overlay: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        viz::to_svg(self, py, overlay)
    }

    /// Render the graph as an ASCII text tree.
    fn to_text(&self) -> String {
        viz::to_text(self)
    }

    /// Notebook display: the architecture as an inline SVG diagram
    /// (falls back to the text tree for very large graphs).
    fn _repr_html_(&self) -> String {
        viz::repr_html(self)
    }

    /// Serialized graph topology (nodes/edges) as JSON — written into
    /// run directories so a front-end can draw the architecture.
    fn graph_json(&self) -> PyResult<String> {
        viz::graph_json(self)
    }

    // ── Tracking ──

    /// Register a Python callback to receive events during execution.
    ///
    /// The callback is called with a dict for each event. Events are
    /// delivered in a background thread; the callback must be thread-safe.
    ///
    /// Usage:
    /// ```python
    /// def on_event(event):
    ///     print(event["event_type"], event.get("node_id", ""))
    /// g.on_event(on_event)
    /// g.fit(data)
    /// ```
    fn on_event(&self, callback: PyObject) -> PyResult<()> {
        tracking::on_event(self, callback)
    }

    /// Emit an event onto the graph's bus from Python.
    ///
    /// The dict must carry an `event_type` matching a Soma event
    /// variant (e.g. `StepCompleted`, `MetricReported`, `HealthFlag`)
    /// plus that variant's fields. Used by the native training loop and
    /// the gradient audit to make Python-side progress visible to
    /// trackers and subscribers.
    fn emit_event(&self, py: Python<'_>, event: &Bound<'_, PyDict>) -> PyResult<()> {
        tracking::emit_event(self, py, event)
    }

    /// Start a tracked run: creates `.soma/runs/<run_id>/`, snapshots
    /// the graph topology into it (`graph.json`, `graph.mmd`,
    /// `fingerprint.json`) and attaches its lossless sink to this
    /// graph's event bus. Prefer the `graph.track_run(...)` context
    /// manager from Python.
    ///
    /// This is the only writer of those three files: it is the one place
    /// where the `Graph` and the `NodeCatalog` are both in scope, so
    /// it is the only place that can stamp per-node config hashes into
    /// the fingerprint.
    ///
    /// `params` are the hyperparameters that live outside the graph;
    /// they are what makes a `ParamChanged` derivation possible when a
    /// later run varies one. `parent` names the run this one descends
    /// from — omit it and soma resolves one from `$SOMA_PARENT_RUN` or
    /// `.soma/HEAD` (see `soma.checkout`).
    #[pyo3(signature = (name, root=".soma".to_string(), kind="train".to_string(), tags=None, params=None, parent=None, hypothesis=None))]
    #[allow(clippy::too_many_arguments)]
    fn begin_run(
        &self,
        py: Python<'_>,
        name: String,
        root: String,
        kind: String,
        tags: Option<Vec<String>>,
        params: Option<&Bound<'_, PyDict>>,
        parent: Option<String>,
        hypothesis: Option<String>,
    ) -> PyResult<PyRun> {
        tracking::begin_run(self, py, name, root, kind, tags, params, parent, hypothesis)
    }

    // ── Where it runs ──

    /// Register a remote worker for direct connection (mode B).
    ///
    /// Usage:
    ///   g.add_worker("ws://gpu-0:8080", token="sk-xxx", tags=["gpu"])
    #[pyo3(signature = (address, token=None, tags=None))]
    fn add_worker(&mut self, address: String, token: Option<String>, tags: Option<Vec<String>>) {
        distributed::add_worker(self, address, token, tags)
    }

    /// Set a coordinator for auto-discovery (mode C).
    ///
    /// Usage:
    ///   g.set_coordinator("http://coord:9090", token="sk-xxx")
    #[pyo3(signature = (url, token=None))]
    fn set_coordinator(&mut self, url: String, token: Option<String>) {
        self.coordinator = Some((url, token));
    }

    /// List known workers (from add_worker or coordinator).
    ///
    /// Returns a list of dicts with worker info.
    fn workers(&self, py: Python<'_>) -> PyResult<PyObject> {
        distributed::workers(self, py)
    }

    /// Shutdown a specific worker by address.
    ///
    /// Usage:
    ///   g.shutdown_worker("ws://worker:8080")
    ///   g.shutdown_worker("ws://worker:8080", reason="maintenance")
    #[pyo3(signature = (address, reason=None))]
    fn shutdown_worker(&self, address: String, reason: Option<String>) -> PyResult<()> {
        distributed::shutdown_worker(self, &address, reason)
    }

    /// Shutdown all registered workers.
    ///
    /// Usage:
    ///   g.shutdown_workers()
    ///   g.shutdown_workers(reason="end of experiment")
    #[pyo3(signature = (reason=None))]
    fn shutdown_workers(&self, reason: Option<String>) -> PyResult<()> {
        distributed::shutdown_workers(self, reason)
    }

    /// Configure a DataStore for persistent data transport (opt-in).
    ///
    /// When set, large payloads are uploaded to the store and workers read
    /// via DataRef instead of receiving data inline or via HTTP upload.
    ///
    /// Usage:
    ///   g.set_data_store("local", path="/data/soma")
    ///   g.set_data_store("s3", bucket="my-lab", prefix="exp/",
    ///                    endpoint="s3.amazonaws.com",
    ///                    access_key="AK...", secret_key="SK...")
    #[pyo3(signature = (store_type, path=None, bucket=None, prefix=None, endpoint=None, access_key=None, secret_key=None, cache_dir=None))]
    #[allow(clippy::too_many_arguments)]
    fn set_data_store(
        &mut self,
        store_type: String,
        path: Option<String>,
        bucket: Option<String>,
        prefix: Option<String>,
        endpoint: Option<String>,
        access_key: Option<String>,
        secret_key: Option<String>,
        cache_dir: Option<String>,
    ) -> PyResult<()> {
        self.data_store = Some(crate::data::store::build_data_store(
            &store_type,
            path,
            bucket,
            prefix,
            endpoint,
            access_key,
            secret_key,
            cache_dir,
        )?);
        Ok(())
    }

    /// Set the graph's training strategy.
    ///
    ///   g.set_strategy("federated", num_clients=2, rounds=3)
    ///   g.set_strategy("data_parallel", num_replicas=4)
    ///
    /// The strings are the ones the documentation has always shown. Until
    /// now the method they were shown on did not exist in Python at all,
    /// and in Rust nothing read the attribute back — see the guide for
    /// what runs today: `federated` does, `data_parallel` needs gradients
    /// the worker cannot yet hand over, and the other two are unwritten.
    #[pyo3(signature = (kind, num_replicas=None, num_clients=None, rounds=None, aggregation=None, generations=None, population_size=None, partitions=None))]
    #[allow(clippy::too_many_arguments)]
    fn set_strategy(
        &mut self,
        kind: &str,
        num_replicas: Option<usize>,
        num_clients: Option<usize>,
        rounds: Option<usize>,
        aggregation: Option<&str>,
        generations: Option<usize>,
        population_size: Option<usize>,
        partitions: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<()> {
        distributed::set_strategy(
            self,
            kind,
            num_replicas,
            num_clients,
            rounds,
            aggregation,
            generations,
            population_size,
            partitions,
        )
    }

    /// The graph's training strategy, as the string `set_strategy` takes.
    fn strategy(&self) -> String {
        distributed::strategy(self)
    }

    // ── Python-side odds and ends ──

    /// Per-graph scratch dict for Python-side orchestration state.
    ///
    /// PyGraph doesn't expose ``__dict__``, so callers (e.g. the
    /// _orchestrator module) use this dict to attach things like the
    /// registered optimiser without monkey-patching the class.
    /// Lazily created on first access.
    #[getter]
    fn py_state(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        if self.py_state.is_none() {
            self.py_state = Some(PyDict::new(py).unbind());
        }
        Ok(self.py_state.as_ref().unwrap().clone_ref(py))
    }

    /// Number of nodes in the graph.
    fn __len__(&self) -> usize {
        self.graph.nodes.len()
    }

    fn __repr__(&self) -> String {
        let n = self.graph.nodes.len();
        let e = self.graph.edges.len();
        format!(
            "Graph({n} nodes, {e} edges, fitted={fitted})",
            fitted = self.fitted
        )
    }

    fn __str__(&self) -> String {
        self.graph.to_text()
    }
}
