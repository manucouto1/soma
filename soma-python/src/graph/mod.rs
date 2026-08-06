//! `Graph` — the primary API.

pub(crate) mod bridge;
mod distributed;
mod registry;

use crate::prelude::*;
use crate::tracking::readers::py_overlay;
use crate::tracking::run::PyRun;
use distributed::RemoteWorker;
use registry::Registry;

// ── PyGraph ──

/// How a finished fit's learned states are keyed, which depends on who
/// produced them.
///
/// The two shapes are not interchangeable, and reading one as the other is
/// not a matter of taste: a runner's map carries every node's *output*
/// under its bare id beside the states, so taking it whole files a
/// scaler's transformed data as the scaler's learned mean — and which of
/// the two wins depends on `HashMap` order.
enum FittedStates {
    /// A [`Runner`]'s output map: node outputs under bare ids, learned
    /// state under `__state_<id>`. Only the prefixed entries are states.
    Runner(HashMap<String, Value>),
    /// A worker's `PlanResult`, or a strategy's round: states only, keyed
    /// by node id — the producer strips the prefix before it answers.
    Trained(HashMap<String, Value>),
}

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
    cut_edges: std::collections::HashMap<(String, String), (usize, Edge)>,
    /// Tools every agent in this graph may call, by name. Collected from the
    /// agents as they are added, so a tool declared once is callable by any
    /// node that lists it.
    tools: std::collections::HashMap<String, PyTool>,
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

    /// File what a fit learned, and mark the graph fitted.
    ///
    /// The tail of every `fit` path. It was written out at each of the five
    /// returns, and the five copies had drifted: the local one filtered on
    /// the `__state_` prefix, the differentiable one — reading a map from
    /// the same runner — fell back to the bare key, which stored every
    /// node's output as its state. [`FittedStates`] is what says which map
    /// this is, so the answer is now the type's rather than each caller's.
    fn absorb(&mut self, states: FittedStates) -> PyResult<()> {
        let learned: Vec<(String, Value)> = match states {
            FittedStates::Runner(map) => map
                .into_iter()
                .filter_map(|(key, state)| {
                    Some((
                        somatize_core::data::keys::node_of_state_key(&key)?.to_string(),
                        state,
                    ))
                })
                .collect(),
            FittedStates::Trained(map) => map.into_iter().collect(),
        };
        for (node_id, state) in learned {
            self.library
                .try_set_state(node_id, state)
                .map_err(soma_err_to_py)?;
        }
        self.fitted = true;
        Ok(())
    }

    /// Rebuild plan and run it here (or on workers). The non-autograd path.
    fn forward_local(
        &self,
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        stream: bool,
        chunk_size: usize,
        seed: Option<i64>,
        run_id: Option<String>,
    ) -> PyResult<PyObject> {
        // A step has no fit phase — its behaviour comes from a model and a
        // prompt, not from learned state. A graph with nothing trainable in
        // it therefore has nothing to fit, and demanding a fit first would
        // be asking for a no-op.
        if !self.fitted && registry::has_trainable_filters(self) {
            return Err(PyRuntimeError::new_err(
                "graph must be fitted before forward",
            ));
        }
        let x_val = py_to_value(py, x)?;

        // Remote streaming: chunks over WS Binary to the worker.
        if stream && !self.workers.is_empty() {
            // Release GIL during WS dispatch so worker thread can acquire
            // it for Python execution.
            let output = py
                .allow_threads(|| distributed::dispatch_streamed(self, &x_val, chunk_size, seed))?;
            return value_to_py(py, &output);
        }

        // Dispatch entire plan remotely if workers registered and no node forces local
        if !stream && !self.workers.is_empty() && self.graph.nodes.iter().all(|n| !n.is_local()) {
            let (output, _states) = py.allow_threads(|| {
                distributed::dispatch_to_worker(
                    self,
                    &x_val,
                    somatize_worker::protocol::ExecutionMode::Forward,
                    seed,
                )
            })?;
            return value_to_py(py, &output);
        }

        // Local execution — one path whether chunked or not. Streaming
        // used to be a hand-rolled sibling that attached no driver, no
        // transport, ignored a resumed run's id and picked its output
        // differently; now the ONLY difference is which compiler entry
        // produced the plan.
        let catalog = registry::rebuild_catalog(self, py)?;
        let compile_result = if stream {
            somatize_compiler::compile_stream(&self.graph, &catalog, chunk_size)
        } else {
            somatize_compiler::compile(
                &self.graph,
                &catalog,
                CompileMode::Inference,
                Some(self.cache.as_ref()),
            )
        }
        .map_err(soma_err_to_py)?;

        let graph_info = GraphInfo::from_graph(&self.graph);
        // A caller resuming a suspended run passes its id back. The
        // journal keys an impure effect by `(run, node, turn, index)`, so
        // a fresh id would replay nothing and the answer already recorded
        // would never be found — which is why resuming did not work.
        let run_id = run_id.unwrap_or_else(|| somatize_core::util::timestamp_id("graph_forward"));
        let mut ctx = Context::new(self.event_bus.clone(), run_id)
            .with_graph_info(graph_info)
            .with_seed(seed);

        if let Some(driver) = self.step_runtime(py, &catalog)? {
            ctx = ctx.with_driver(driver);
        }
        if let Some(transport) = distributed::make_transport(self) {
            ctx = ctx.with_transport(transport);
        }

        let roots = self.graph.roots();
        if roots.len() == 1 {
            ctx.set(
                somatize_core::data::keys::input_key(roots[0]),
                x_val.clone(),
            );
        }
        ctx.set(somatize_core::data::keys::GRAPH_INPUT, x_val);

        // Release the GIL: Parallel plans run branches on scoped threads
        // whose Python filters must acquire it — holding it here would
        // deadlock the join.
        py.allow_threads(|| {
            executor::execute(
                &compile_result.plan,
                &mut ctx,
                &catalog,
                self.cache.as_ref(),
            )
        })
        .map_err(soma_err_to_py)?;

        // Which leaf is "the output" when there are several? Prefer one that
        // actually ran. A branch makes every arm a leaf, so declaration
        // order alone would return the arm that was *not* taken — an empty
        // value, from a node that never executed.
        //
        // Among leaves that did produce something, declaration order still
        // decides, so a parallel fan-out answers the same as it always has.
        let leaves = self.graph.leaves();
        let output = leaves
            .iter()
            .find_map(|id| ctx.get(id).cloned())
            .or_else(|| leaves.first().and_then(|id| ctx.get(id).cloned()))
            .or_else(|| {
                // The last node that actually ran — skipping the run's own
                // reserved entries, which `last()` alone would happily
                // return as though a node had produced them.
                ctx.execution_order()
                    .iter()
                    .rev()
                    .find(|id| !somatize_core::data::keys::is_reserved(id))
                    .and_then(|id| ctx.get(id).cloned())
            })
            .unwrap_or(Value::Empty);

        value_to_py(py, &output)
    }

    /// A node id not yet taken, suffixing `_2`, `_3`, … as needed.
    fn free_id(&self, wanted: &str) -> String {
        if self.graph.node(wanted).is_none() {
            return wanted.to_string();
        }
        let mut i = 2;
        loop {
            let candidate = format!("{wanted}_{i}");
            if self.graph.node(&candidate).is_none() {
                return candidate;
            }
            i += 1;
        }
    }

    /// Resolve one arm of a branch or one entry of a loop body: either the
    /// id of a node already in the graph, or a filter/agent to add as one.
    fn resolve_member(
        &mut self,
        py: Python<'_>,
        fallback_id: &str,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<String> {
        if let Ok(existing) = obj.extract::<String>() {
            if self.graph.node(&existing).is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "`{existing}` names no node in this graph. Pass a filter or an \
                     agent to create one, or an id already added with node()"
                )));
            }
            return Ok(existing);
        }

        let id = self.free_id(fallback_id);
        let node = registry::register_behaviour(self, py, &id, obj)?.node(&id);
        self.graph.add_node(node);
        Ok(id)
    }

    /// Add a labelled control edge — the wire the compiler reads to decide
    /// which nodes a loop or branch owns.
    fn control_edge(&mut self, source: &str, target: &str, label: Option<&str>) {
        let id = format!("e_{}", self.graph.edges.len());
        let mut edge = Edge::control(id, source, target);
        if let Some(label) = label {
            edge = edge.with_label(label);
        }
        self.graph.add_edge(edge);
    }

    /// Build the step library and effect driver an agentic plan needs.
    ///
    /// Returns `None` for a graph with no steps, so a purely computational
    /// pipeline never constructs a provider router, reads a catalog, or
    /// touches an environment variable.
    fn step_runtime(
        &self,
        py: Python<'_>,
        catalog: &NodeCatalog,
    ) -> PyResult<Option<EffectDriver>> {
        if !catalog.has_steps() {
            return Ok(None);
        }
        // Captured before `somatize_llm::Catalog` shadows the name below.
        let node_catalog = Arc::new(catalog.clone());

        // Python tools and MCP tools land in one toolbox: to a model they
        // are the same thing, and a step names them the same way. Tools
        // declared on a live agent are collected here too, so an agent that
        // gained one since the graph was built can still call it.
        let mut toolbox = somatize_llm::Toolbox::new();
        for tool in self.tools.values() {
            toolbox.add(Arc::new(PyToolAdapter { tool: tool.clone() }));
        }
        for (_, obj) in self.nodes.steps() {
            for tool in to_step_spec(py, obj.bind(py))?.tools() {
                toolbox.add(Arc::new(PyToolAdapter { tool: tool.clone() }));
            }
        }
        for mcp in &self.mcp_toolboxes {
            toolbox.merge_from(mcp);
        }

        let catalog = somatize_llm::Catalog::load().map_err(soma_err_to_py)?;
        let mut router = somatize_llm::Router::from_catalog(catalog).map_err(soma_err_to_py)?;
        if let Some(default) = &self.default_provider {
            router = router.with_default(default);
        }

        // The journal shares the graph's cache directory, so an agentic run
        // is resumable by the same mechanism a computational one is.
        let cache_dir = default_cache_dir().ok_or_else(|| {
            PyRuntimeError::new_err(
                "an agentic graph needs somewhere to journal its effects; \
                 set SOMA_CACHE_DIR or HOME",
            )
        })?;
        let store = Arc::new(FsActionStore::new(cache_dir).map_err(soma_err_to_py)?);
        let journal = EffectJournal::new(store.clone(), store);

        // The base handlers, shared with the graph handler below so a
        // sub-pipeline's own agents reach the same providers, tools and
        // journal — that is what makes agent → pipeline → agent one run.
        let base: Vec<Arc<dyn somatize_core::agentic::effect::EffectHandler>> = vec![
            Arc::new(somatize_llm::LlmHandler::new(router)),
            Arc::new(toolbox),
            Arc::new(somatize_runtime::agentic::SleepHandler),
        ];
        let graph_handler = somatize_runtime::agentic::GraphHandler::new((*node_catalog).clone())
            .with_cache(self.cache.clone())
            .with_step_runtime(base.clone(), journal.clone())
            .with_event_bus(self.event_bus.clone());

        let mut driver = EffectDriver::new(journal)
            .with_event_bus(self.event_bus.clone())
            .with_handler(Arc::new(graph_handler))
            // The driver carries its own catalog: this is where a
            // `Spawn` transition finds the nodes it names.
            .with_catalog(node_catalog);
        for handler in base {
            driver = driver.with_handler(handler);
        }

        Ok(Some(driver))
    }

    /// Write the run's topology snapshot: `graph.json` (the machine
    /// contract), `graph.mmd` (the human one) and `fingerprint.json`
    /// (structural identity, with each node's filter config hash).
    ///
    /// Called from `begin_run` — the single writer. The fingerprint is
    /// best-effort: a graph whose canonical form will not serialize
    /// must not stop a run from starting.
    fn snapshot_topology(&self, tracker: &LocalTracker) -> PyResult<()> {
        let graph_json = serde_json::to_string_pretty(&self.graph)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        tracker
            .save_artifact("graph.json", graph_json.as_bytes())
            .map_err(soma_err_to_py)?;
        tracker
            .save_artifact("graph.mmd", self.graph.to_mermaid().as_bytes())
            .map_err(soma_err_to_py)?;

        if let Ok(fingerprint) = ArchitectureFingerprint::of(&self.graph) {
            let node_config: std::collections::BTreeMap<String, String> = self
                .graph
                .nodes
                .iter()
                .filter_map(|node| {
                    let hash = self.library.get(&node.id)?.config_hash();
                    Some((node.id.clone(), hash.to_hex()))
                })
                .collect();
            let fingerprint = fingerprint.with_node_config(node_config);
            if let Ok(json) = serde_json::to_string_pretty(&fingerprint) {
                tracker
                    .save_artifact("fingerprint.json", json.as_bytes())
                    .map_err(soma_err_to_py)?;
            }
        }
        Ok(())
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
            cut_edges: std::collections::HashMap::new(),
            tools: std::collections::HashMap::new(),
            default_provider: None,
            mcp_toolboxes: Vec::new(),
            py_state: None,
        })
    }

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
        let (node_id, filter_obj) = match args.len() {
            1 => {
                let filter_obj = args.get_item(0)?;
                let class_name = filter_obj
                    .getattr("__class__")?
                    .getattr("__name__")?
                    .extract::<String>()?;
                let snake = to_snake_case(&class_name);
                (snake, filter_obj.to_owned())
            }
            2 => {
                let id = args.get_item(0)?.extract::<String>()?;
                let filter_obj = args.get_item(1)?;
                (id, filter_obj.to_owned())
            }
            n => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "node() takes 1 or 2 positional arguments, got {n}"
                )));
            }
        };

        // An Agent or a Judge is a node too — it just runs a turn loop
        // instead of a function. `register_behaviour` dispatches, so there
        // is one way to add a node rather than a second method whose name
        // would collide with the optimiser's `step()`.
        let actual_id = self.free_id(&node_id);
        let mut node =
            registry::register_behaviour(self, py, &actual_id, &filter_obj)?.node(&actual_id);
        if let Some(t) = target {
            node = node.with_target(t);
        }
        self.graph.add_node(node);

        Ok(actual_id)
    }

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
        if arms.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "branch() needs at least one arm; a router with nowhere to \
                 route is just a node",
            ));
        }

        let actual_id = self.free_id(&node_id);
        // The branch node *is* the condition: the executor runs it and reads
        // the arm label from its output.
        registry::register_behaviour(self, py, &actual_id, condition)?;

        let labels: Vec<String> = arms
            .keys()
            .iter()
            .map(|k| k.extract::<String>())
            .collect::<PyResult<_>>()?;

        let mut node = Node::branch_over(&actual_id, labels);
        if let Some(t) = target {
            node = node.with_target(t);
        }
        self.graph.add_node(node);

        for (key, value) in arms.iter() {
            let label = key.extract::<String>()?;
            let arm_id = self.resolve_member(py, &label, &value)?;
            self.control_edge(&actual_id, &arm_id, Some(&label));
        }

        Ok(actual_id)
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
        // One entry or several: a list is the general case, a bare value the
        // one people write.
        let entries: Vec<Bound<'_, PyAny>> = match body.try_iter() {
            Ok(iter) if !body.is_instance_of::<pyo3::types::PyString>() => {
                iter.collect::<PyResult<_>>()?
            }
            _ => vec![body.clone()],
        };
        if entries.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "loop() needs a body",
            ));
        }

        let actual_id = self.free_id(&node_id);
        use somatize_core::graph::control::LoopCondition;
        let until = match until {
            None => LoopCondition::BodyTerminal,
            // `False` is the only bool that means anything here: "run the
            // whole count". `True` would have to mean "stop immediately",
            // which nobody writes on purpose.
            Some(u) if u.is_instance_of::<pyo3::types::PyBool>() => {
                if u.extract::<bool>()? {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "until=True says the loop stops before it runs. Pass a node \
                         id to read the signal from, or False to run the full count",
                    ));
                }
                LoopCondition::Exhaust
            }
            Some(u) => {
                let cond = u.extract::<String>()?;
                if self.graph.node(&cond).is_none() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "`{cond}` names no node in this graph, so it cannot be the \
                         loop's stop condition"
                    )));
                }
                LoopCondition::WhenSignaled(cond)
            }
        };

        self.graph
            .add_node(Node::loop_until(&actual_id, max_iterations, until));

        for (i, entry) in entries.iter().enumerate() {
            let fallback = format!("{actual_id}_body_{i}");
            let entry_id = self.resolve_member(py, &fallback, entry)?;
            self.control_edge(&actual_id, &entry_id, None);
        }

        Ok(actual_id)
    }

    /// Set the provider that serves model names given without a prefix.
    ///
    /// ```python
    /// g.use_provider("ollama")
    /// g.step("a", soma.Agent(model="llama3.2"))   # → ollama/llama3.2
    /// ```
    fn use_provider(&mut self, provider: String) {
        self.default_provider = Some(provider);
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
        let found = self
            .graph
            .edges
            .iter()
            .find(|e| e.source == source && e.target == target);

        match found {
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "there is no edge `{source}` → `{target}` to make optional"
            ))),
            Some(e) if e.kind != somatize_core::graph::EdgeKind::Data => {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "`{source}` → `{target}` is a control edge; cutting it would \
                     change what the loop or branch owns, not just what flows"
                )))
            }
            Some(_) => {
                let pair = (source, target);
                if !self.optional_edges.contains(&pair) {
                    self.optional_edges.push(pair);
                }
                Ok(())
            }
        }
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
        let pair = (source, target);
        if !self.optional_edges.contains(&pair) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "`{}` → `{}` was never declared optional; call optional() first",
                pair.0, pair.1
            )));
        }

        if enabled {
            if let Some((at, edge)) = self.cut_edges.remove(&pair) {
                // Back where it was, not on the end. Appending would leave a
                // graph that is semantically the same and renders, hashes and
                // fingerprints differently — so two trials of the same
                // topology would not compare equal.
                self.graph
                    .edges
                    .insert(at.min(self.graph.edges.len()), edge);
            }
        } else if !self.cut_edges.contains_key(&pair)
            && let Some(i) = self
                .graph
                .edges
                .iter()
                .position(|e| e.source == pair.0 && e.target == pair.1)
        {
            let edge = self.graph.edges.remove(i);
            self.cut_edges.insert(pair, (i, edge));
        }
        Ok(())
    }

    /// The live `Agent`/`Judge` behind each step node, as `(node_id, obj)`.
    ///
    /// The counterpart of `filters()`. A study reads their search spaces and
    /// writes sampled values straight onto them.
    fn steps(&self, py: Python<'_>) -> Vec<(String, PyObject)> {
        registry::steps(self, py)
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
        let args = args.unwrap_or_default();
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();

        // Spawning and handshaking is I/O; do not hold the GIL for it.
        let mut toolbox = somatize_llm::Toolbox::new();
        py.allow_threads(|| toolbox.add_mcp_server(&command, &refs))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let names: Vec<String> = toolbox.names().into_iter().map(String::from).collect();
        self.mcp_toolboxes.push(toolbox);
        Ok(names)
    }

    /// Connect two nodes with a data edge.
    fn edge(&mut self, source: String, target: String) {
        let id = format!("e_{}", self.graph.edges.len());
        self.graph.add_edge(Edge::data(id, source, target));
    }
    /// Declare that `source` may hand control to `target`.
    ///
    /// This is what `soma.Goto(target)` needs: a handoff transfers control
    /// rather than passing data, so it is a control edge and not a
    /// `connect`. Declaring it is deliberate — a step that hands control
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
        self.control_edge(source, target, None);
    }

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
        let x_val = py_to_value(py, x)?;
        let y_val = match y {
            Some(v) => Some(py_to_value(py, v)?),
            None => None,
        };

        // Differentiable mode: compile with CompileMode::Differentiable (which
        // collapses consecutive differentiable filters into a Composite block)
        // and execute via LocalRunner, which delegates the block to the first
        // filter's ``composite_fit``. Gradients flow end-to-end inside the
        // user-provided composite_fit implementation.
        if mode == "differentiable" {
            // `mode="differentiable"` is the *local* loop: the caller drives
            // `context`/`backward`/`step` and owns when the parameters move.
            // A worker cannot be driven that way — it would need distributed
            // autograd — so this stays refused rather than running a fit
            // that computes gradients and never steps.
            //
            // Training a differentiable graph on workers is
            // `set_strategy("data_parallel")`, which is a complete round:
            // each replica fits its own shard, the gradients are averaged
            // across replicas and applied, and the stepped weights are read
            // back. See `guides/execution-modes.md`.
            if !self.workers.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "mode='differentiable' drives the training loop locally, so \
                     it cannot run on workers. To train this graph on the \
                     workers you registered, set a strategy instead:\n    \
                     g.set_strategy(\"data_parallel\", num_replicas=N)\n    \
                     g.fit(x, y)",
                ));
            }
            self.graph.validate().map_err(soma_err_to_py)?;
            let catalog = registry::rebuild_catalog(self, py)?;
            let compile_result = compile(
                &self.graph,
                &catalog,
                CompileMode::Differentiable,
                Some(self.cache.as_ref()),
            )
            .map_err(soma_err_to_py)?;
            let runner = LocalRunner;
            let run_id = somatize_core::util::timestamp_id("fit");
            self.event_bus
                .emit(somatize_core::tracking::event::Event::RunStarted {
                    run_id: run_id.clone(),
                    plan_summary: compile_result.plan.summary(),
                });
            let run_start = std::time::Instant::now();
            let run_ctx = somatize_runtime::execution::runner::RunContext::new(
                &catalog,
                self.cache.as_ref(),
                &self.event_bus,
                &run_id,
                GraphInfo::from_graph(&self.graph),
            );
            let result = runner.fit(&compile_result.plan, &run_ctx, &x_val, y_val.as_ref());
            let (_output, states) = match result {
                Ok(out) => {
                    self.event_bus
                        .emit(somatize_core::tracking::event::Event::RunCompleted {
                            run_id,
                            duration: run_start.elapsed(),
                        });
                    out
                }
                Err(e) => {
                    self.event_bus
                        .emit(somatize_core::tracking::event::Event::RunFailed {
                            run_id,
                            error: e.to_string(),
                        });
                    return Err(soma_err_to_py(e));
                }
            };
            return self.absorb(FittedStates::Runner(states));
        }
        if mode != "inference" {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown mode={mode:?}. Use 'inference' or 'differentiable'."
            )));
        }

        // A strategy over several workers goes through the runtime's
        // StrategyExecutor rather than a single dispatch: it shards the
        // input, runs a round per client and aggregates between rounds.
        // One worker is not a strategy — it is the ordinary path below.
        if self.graph.effective_strategy_is_distributed()
            && self.workers.len() > 1
            && self.graph.nodes.iter().all(|n| !n.is_local())
        {
            distributed::register_filters_on_all(self)?;
            let transports = distributed::transports(self);
            let states = py.allow_threads(|| {
                distributed::session_with_transports(self, transports)
                    .and_then(|mut session| session.fit(&x_val, y_val.as_ref()))
            });
            let states = states.map_err(soma_err_to_py)?;
            return self.absorb(FittedStates::Trained(states));
        }

        // Dispatch fit to a worker if possible. Batching is the worker's
        // business either way — `batch_size` travels inside the mode — so
        // the batched and unbatched dispatches were the same call written
        // twice.
        //
        // Release the GIL during WS dispatch so the worker thread can
        // acquire it for Python execution.
        if !self.workers.is_empty() && self.graph.nodes.iter().all(|n| !n.is_local()) {
            let mode = somatize_worker::protocol::ExecutionMode::Fit {
                y: y_val.clone(),
                batch_size,
            };
            let result =
                py.allow_threads(|| distributed::dispatch_to_worker(self, &x_val, mode, seed));
            let (_output, states) = result?;
            return self.absorb(FittedStates::Trained(states));
        }

        // Local fit.
        //
        // Through the compiler and the runner, like every other entry
        // point. This used to be a topological loop written here, walking
        // `graph.topological_sort()` and calling fit/forward node by node
        // — so it ignored parallelism, loops and branches, and it was the
        // only fit anywhere that salted its state keys with the seed. Now
        // the runner salts, and the loop is gone.
        self.graph.validate().map_err(soma_err_to_py)?;
        let catalog = registry::rebuild_catalog(self, py)?;
        let compile_result = compile(
            &self.graph,
            &catalog,
            CompileMode::NoCache,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        let run_id = somatize_core::util::timestamp_id("graph_fit");
        self.event_bus
            .emit(somatize_core::tracking::event::Event::RunStarted {
                run_id: run_id.clone(),
                plan_summary: compile_result.plan.summary(),
            });
        let run_start = std::time::Instant::now();

        let mut run_ctx = somatize_runtime::execution::runner::RunContext::new(
            &catalog,
            self.cache.as_ref(),
            &self.event_bus,
            &run_id,
            GraphInfo::from_graph(&self.graph),
        )
        .with_seed(seed);

        // A fit reaches steps now, so it has to be able to drive them —
        // `forward` has always attached this. Without it a graph mixing
        // filters and steps fitted the filters and then stopped at the
        // first step for want of a driver.
        if let Some(driver) = self.step_runtime(py, &catalog)? {
            run_ctx = run_ctx.with_driver(driver);
        }

        // Release the GIL: a Parallel plan runs branches on scoped threads
        // whose Python filters must acquire it.
        let result = py.allow_threads(|| {
            LocalRunner.fit(&compile_result.plan, &run_ctx, &x_val, y_val.as_ref())
        });

        let (_output, states) = match result {
            Ok(out) => {
                self.event_bus
                    .emit(somatize_core::tracking::event::Event::RunCompleted {
                        run_id,
                        duration: run_start.elapsed(),
                    });
                out
            }
            Err(e) => {
                self.event_bus
                    .emit(somatize_core::tracking::event::Event::RunFailed {
                        run_id,
                        error: e.to_string(),
                    });
                return Err(soma_err_to_py(e));
            }
        };

        self.absorb(FittedStates::Runner(states))
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
        // A graph whose filters carry torch modules is walked in Python,
        // because autograd does not survive the `Value` boundary: a tensor
        // that becomes a vector of f64 and back has lost the graph the
        // optimiser needs.
        //
        // That walk used to *replace* this method at import time, so which
        // engine ran depended on which modules had been imported, two
        // implementations answered to one name, and neither
        // `help(Graph.forward)` nor any static analysis could see it. The
        // dispatch belongs here, where the graph knows what it holds; the
        // walk is a named function this calls.
        if registry::has_differentiable_filters(&slf, py) {
            // The torch walk honours none of these: it does not chunk, it
            // does not salt a cache key (nothing it produces is cached),
            // and it emits no run bracket. They used to be accepted and
            // discarded, so `g.forward(x, seed=42)` on a torch graph
            // reported success having ignored the seed — and a seed that
            // is silently ignored is worse than one that is refused,
            // because the run looks reproducible.
            let ignored: Vec<&str> = [
                ("stream", stream),
                ("chunk_size", chunk_size.is_some()),
                ("seed", seed.is_some()),
                ("run_id", run_id.is_some()),
            ]
            .iter()
            .filter(|(_, given)| *given)
            .map(|(name, _)| *name)
            .collect();
            if !ignored.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "this graph holds differentiable filters, so it is walked in \
                     Python for autograd — and that walk cannot honour {}. Run the \
                     graph in eval mode (`g.eval()`, after `g.freeze()`) to reach the \
                     Rust path, which can.",
                    ignored.join(", ")
                )));
            }
            let graph = Py::from(slf);
            let walk = py.import("soma._orchestrator")?;
            return walk
                .call_method1("differentiable_forward", (graph, x))
                .map(|v| v.unbind());
        }
        slf.forward_local(py, x, stream, chunk_size.unwrap_or(1024), seed, run_id)
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
        let reason: somatize_core::agentic::effect::SuspendReason =
            serde_json::from_value(py_any_to_json(reason)?).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "`reason` should be the one from the SomaSuspended exception: {e}"
                ))
            })?;

        let catalog = registry::rebuild_catalog(self, py)?;
        let driver = self.step_runtime(py, &catalog)?.ok_or_else(|| {
            PyRuntimeError::new_err(
                "this graph has no effectful nodes, so nothing in it can suspend",
            )
        })?;

        // Release the GIL, like every sibling entry point. Resuming
        // drives the effect journal forward, which can perform a model
        // call — holding the GIL across that blocks every other Python
        // thread in the process for the length of an HTTP request. `fit`,
        // `forward` and `run` all release it; this one did not.
        let answer = py_to_value(py, answer)?;
        py.allow_threads(|| driver.resume_with(run_id, node_id, turn, &reason, answer))
            .map_err(soma_err_to_py)
    }

    /// Compile the graph and return diagnostic information.
    #[pyo3(signature = (mode="inference"))]
    fn compile(&self, py: Python<'_>, mode: &str) -> PyResult<PyObject> {
        let compile_mode = match mode {
            "inference" => CompileMode::Inference,
            "differentiable" => CompileMode::Differentiable,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown mode: {mode}. Use 'inference' or 'differentiable' — \
                     the same two `fit` takes."
                )));
            }
        };

        // The rebuilt catalog, not `self.library`: passing the filter half
        // alone is how `.compile()` came to skip every step's schema while
        // `.run()` checked them.
        let catalog = registry::rebuild_catalog(self, py)?;
        let result = somatize_compiler::compile(
            &self.graph,
            &catalog,
            compile_mode,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        let dict = PyDict::new(py);
        let summary = result.plan.summary();
        dict.set_item("total_nodes", summary.total_nodes)?;
        dict.set_item("cached_nodes", summary.cached_nodes)?;
        dict.set_item("parallel_branches", summary.parallel_branches)?;

        // Structured diagnostics: {node, level, message} dicts, not
        // Debug strings — readable and machine-consumable.
        let diags = PyList::empty(py);
        for d in &result.diagnostics {
            let entry = PyDict::new(py);
            entry.set_item("node", &d.node_id)?;
            entry.set_item(
                "level",
                match d.level {
                    somatize_compiler::DiagnosticLevel::Warning => "warning",
                    somatize_compiler::DiagnosticLevel::Info => "info",
                },
            )?;
            entry.set_item("message", &d.message)?;
            diags.append(entry)?;
        }
        dict.set_item("diagnostics", diags)?;
        dict.set_item("plan_text", format!("{}", result.plan))?;
        dict.set_item("plan_mermaid", result.plan.to_mermaid())?;
        dict.set_item("plan_svg", result.plan.to_graph().to_svg())?;

        Ok(dict.into_any().unbind())
    }

    // ── Visualization ──

    /// Render the graph as a Mermaid diagram string.
    ///
    /// `overlay` is an optional dict of per-node execution annotations
    /// (the shape `RunView.overlay()` returns): status coloring plus a
    /// duration/cache/flags label line per node.
    #[pyo3(signature = (overlay=None))]
    fn to_mermaid(&self, py: Python<'_>, overlay: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        match overlay {
            None => Ok(self.graph.to_mermaid()),
            Some(ov) => Ok(self.graph.to_mermaid_with(&py_overlay(py, ov)?)),
        }
    }

    /// Render the graph as a self-contained SVG diagram (same optional
    /// `overlay` as `to_mermaid`). No JavaScript — displays inline in
    /// any notebook viewer.
    #[pyo3(signature = (overlay=None))]
    fn to_svg(&self, py: Python<'_>, overlay: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        match overlay {
            None => Ok(self.graph.to_svg()),
            Some(ov) => Ok(self.graph.to_svg_with(&py_overlay(py, ov)?)),
        }
    }

    /// Notebook display: the architecture as an inline SVG diagram
    /// (falls back to the text tree for very large graphs).
    fn _repr_html_(&self) -> String {
        if self.graph.nodes.is_empty() {
            return "<i>empty graph — add nodes with g.node(...)</i>".to_string();
        }
        if self.graph.nodes.len() > 80 {
            return format!(
                "<pre style='font-family:ui-monospace,monospace'>{}</pre>",
                self.graph
                    .to_text()
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
            );
        }
        self.graph.to_svg()
    }

    /// Render the graph as an ASCII text tree.
    fn to_text(&self) -> String {
        self.graph.to_text()
    }

    // ── Events ──

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
        let mut rx = self.event_bus.subscribe();
        std::thread::spawn(move || {
            while let Ok(event) = rx.blocking_recv() {
                if let Ok(json_str) = serde_json::to_string(&event) {
                    Python::with_gil(|py| {
                        // Parse JSON string into Python dict via json.loads
                        let json_mod = py.import("json").unwrap();
                        if let Ok(dict) = json_mod.call_method1("loads", (json_str,)) {
                            let _ = callback.call1(py, (dict,));
                        }
                    });
                }
            }
        });
        Ok(())
    }

    /// Emit an event onto the graph's bus from Python.
    ///
    /// The dict must carry an `event_type` matching a Soma event
    /// variant (e.g. `StepCompleted`, `MetricReported`, `HealthFlag`)
    /// plus that variant's fields. Used by the native training loop and
    /// the gradient audit to make Python-side progress visible to
    /// trackers and subscribers.
    fn emit_event(&self, py: Python<'_>, event: &Bound<'_, PyDict>) -> PyResult<()> {
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (event,))?.extract()?;
        let value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid event JSON: {e}")))?;
        let event: somatize_core::tracking::event::Event = serde_json::from_value(value)
            .map_err(|e| PyRuntimeError::new_err(format!("unknown or malformed event: {e}")))?;
        self.event_bus.emit(event);
        Ok(())
    }

    /// Serialized graph topology (nodes/edges) as JSON — written into
    /// run directories so a front-end can draw the architecture.
    fn graph_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.graph)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
        let kind = match kind.as_str() {
            "fit" => RunKind::Fit,
            "train" => RunKind::Train,
            "study" => RunKind::Study,
            "trial" => RunKind::Trial,
            _ => RunKind::Other,
        };
        let tracker = LocalTracker::create(&root, kind, &name).map_err(soma_err_to_py)?;
        self.snapshot_topology(&tracker)?;

        // Enrich the manifest with Python-side context.
        let mut manifest = load_manifest(tracker.run_dir()).map_err(soma_err_to_py)?;
        manifest.tags = tags.unwrap_or_default();
        manifest.python_version = Some(py.version().split_whitespace().next().unwrap_or("").into());
        manifest.params = match params {
            Some(dict) => match py_any_to_json(dict.as_any())? {
                serde_json::Value::Object(map) => map.into_iter().collect(),
                _ => HashMap::new(),
            },
            None => HashMap::new(),
        };
        manifest.parent_run_id = resolve_parent(&root, parent.as_deref());
        manifest.hypothesis = hypothesis;
        manifest.graph = Some(GraphSummaryInfo {
            n_nodes: self.graph.nodes.len(),
            node_ids: self.graph.nodes.iter().map(|n| n.id.clone()).collect(),
            graph_path: Some("graph.json".into()),
            mermaid_path: Some("graph.mmd".into()),
        });
        tracker.save_manifest(&manifest).map_err(soma_err_to_py)?;

        let sink = tracker.sink();
        self.event_bus.add_sink(sink.clone());
        Ok(PyRun {
            tracker: Arc::new(tracker),
            bus: self.event_bus.clone(),
            sink,
            finished: std::sync::atomic::AtomicBool::new(false),
            summary: std::sync::Mutex::new(HashMap::new()),
        })
    }

    // ── Workers ──

    /// Register a remote worker for direct connection (mode B).
    ///
    /// Usage:
    ///   g.add_worker("ws://gpu-0:8080", token="sk-xxx", tags=["gpu"])
    #[pyo3(signature = (address, token=None, tags=None))]
    fn add_worker(&mut self, address: String, token: Option<String>, tags: Option<Vec<String>>) {
        distributed::add_worker(self, address, token, tags)
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

    /// Get the full module source code for a filter node (for Nous agent introspection).
    /// Returns None if the node has no captured source.
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

    /// List data edges as ``[(source, target), ...]`` in insertion order.
    ///
    /// Used by :meth:`Graph.save` to record topology in the manifest so
    /// :meth:`Graph.load` can reconstruct non-linear graphs (forks,
    /// joins) instead of falling back to a linear chain.
    fn edges(&self) -> Vec<(String, String)> {
        self.graph
            .edges
            .iter()
            .map(|e| (e.source.clone(), e.target.clone()))
            .collect()
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

fn to_snake_case(name: &str) -> String {
    name.chars()
        .enumerate()
        .fold(String::new(), |mut s, (i, c)| {
            if c.is_uppercase() && i > 0 {
                s.push('_');
            }
            s.push(c.to_ascii_lowercase());
            s
        })
}
