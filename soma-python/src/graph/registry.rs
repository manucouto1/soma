//! What the graph knows about each node it registered.
//!
//! One record per node, and the queries that read it. The `Graph` itself —
//! nodes, edges, what connects to what — is [`super::topology`]; this is
//! the other half: *what a node id actually is*, in Python.

use super::PyGraph;
use crate::prelude::*;

/// What a registered node *does*, once [`register_behaviour`] has filed it
/// away — enough to build the graph node, and nothing else.
pub(super) enum Behaviour {
    /// An effectful step, carrying its `step_name`.
    Step(String),
    /// An ordinary filter, carrying its `filter_name`.
    Filter(String),
}

impl Behaviour {
    pub(super) fn node(&self, id: &str) -> Node {
        match self {
            Behaviour::Step(kind) => Node::step(id, kind),
            Behaviour::Filter(name) => Node::filter_with_id(id, name),
        }
    }
}

/// Everything registration read off a Python filter.
pub(super) struct FilterRecord {
    /// The live Python instance, retained by node id. The in-process
    /// training path (`graph.train`/`forward`/`freeze`) needs a filter's
    /// persistent state — an `nn.Module` attached to `self` — to survive
    /// across forward calls instead of being deserialised each time.
    pub(super) live: Py<PyAny>,
    /// `cloudpickle` bytes, for remote dispatch only. Distinct from
    /// `live`: a `NodeCatalog` holds live filters and never the pickle,
    /// so these bytes exist nowhere else and a worker cannot rebuild the
    /// filter without them.
    pub(super) pickled: Vec<u8>,
    /// Third-party distributions the worker must install to run this node.
    pub(super) requirements: Vec<String>,
    /// The filter module's full source (imports + classes + helpers), for
    /// agent introspection and editing.
    pub(super) source: String,
    pub(super) trainable: bool,
}

/// One node's implementation, whichever kind it is.
///
/// This was five parallel `HashMap<String, _>` — `pickled_filters`,
/// `filter_sources`, `filter_trainable`, `live_filters`, `live_steps` —
/// written together in one function and never deleted from. They agreed
/// only because nothing in the API removes a node: "a filter is in
/// exactly four of these, a step in the fifth" was an invariant written
/// nowhere and checked by nothing. One record per node makes it a type.
pub(super) enum NodeRecord {
    Filter(FilterRecord),
    /// A step's live `Agent`/`Judge`. A `Step` is immutable once built, so
    /// a study that samples a new prompt or model writes here, and the
    /// catalog is rebuilt from it before every compile and every run.
    Step(Py<PyAny>),
}

/// The node id → implementation map.
#[derive(Default)]
pub(super) struct Registry {
    nodes: HashMap<String, NodeRecord>,
}

impl Registry {
    pub(super) fn insert_filter(&mut self, node_id: &str, record: FilterRecord) {
        self.nodes
            .insert(node_id.to_string(), NodeRecord::Filter(record));
    }

    pub(super) fn insert_step(&mut self, node_id: &str, live: Py<PyAny>) {
        self.nodes
            .insert(node_id.to_string(), NodeRecord::Step(live));
    }

    pub(super) fn filter(&self, node_id: &str) -> Option<&FilterRecord> {
        match self.nodes.get(node_id) {
            Some(NodeRecord::Filter(f)) => Some(f),
            _ => None,
        }
    }

    pub(super) fn filters(&self) -> impl Iterator<Item = (&String, &FilterRecord)> {
        self.nodes.iter().filter_map(|(id, record)| match record {
            NodeRecord::Filter(f) => Some((id, f)),
            NodeRecord::Step(_) => None,
        })
    }

    pub(super) fn steps(&self) -> impl Iterator<Item = (&String, &Py<PyAny>)> {
        self.nodes.iter().filter_map(|(id, record)| match record {
            NodeRecord::Step(live) => Some((id, live)),
            NodeRecord::Filter(_) => None,
        })
    }

    pub(super) fn has_steps(&self) -> bool {
        self.steps().next().is_some()
    }
}

// ── Registration ──

/// Register what a node *does*, without saying what shape it has in the
/// graph. A branch node runs a classifier and routes; a plain node runs
/// the same classifier and stops. The behaviour registration is
/// identical, so it lives here and the two callers differ only in the
/// [`Node`] they add.
///
/// Returns what the caller needs to build the graph node itself.
pub(super) fn register_behaviour(
    g: &mut PyGraph,
    py: Python<'_>,
    node_id: &str,
    obj: &Bound<'_, PyAny>,
) -> PyResult<Behaviour> {
    if let Ok(spec) = to_step_spec(py, obj) {
        // Tools travel with the graph, not with the node: one agent may
        // declare a tool and another list the same one, and both should
        // reach the same implementation.
        for tool in spec.tools() {
            g.tools.insert(tool.tool_name().to_string(), tool.clone());
        }
        let kind = spec.kind().to_string();
        g.library.register_step_arc(node_id, spec.step());
        g.nodes.insert_step(node_id, obj.clone().unbind());
        return Ok(Behaviour::Step(kind));
    }

    let bridge = PyFilterBridge::new(py, obj)?;
    let name = bridge.name.clone();
    g.nodes.insert_filter(
        node_id,
        FilterRecord {
            live: obj.clone().unbind(),
            pickled: bridge.pickled_bytes.clone(),
            requirements: bridge.requirements.clone(),
            source: bridge.source.clone(),
            trainable: bridge.trainable,
        },
    );
    g.library.register(node_id.to_string(), Box::new(bridge));
    Ok(Behaviour::Filter(name))
}

/// Make `sub`'s node implementations runnable by this graph's steps.
pub(super) fn register_graph(
    g: &mut PyGraph,
    py: Python<'_>,
    sub: PyRef<'_, PyGraph>,
) -> PyResult<()> {
    let sub_catalog = rebuild_catalog(&sub, py)?;
    g.library.merge_from(&sub_catalog).map_err(soma_err_to_py)?;
    for (node_id, obj) in sub.nodes.steps() {
        g.nodes.insert_step(node_id, obj.clone_ref(py));
    }
    for (name, tool) in &sub.tools {
        g.tools.insert(name.clone(), tool.clone());
    }
    Ok(())
}

/// Register a step that can be *spawned* but is not a node in the graph.
pub(super) fn register_step(
    g: &mut PyGraph,
    py: Python<'_>,
    step_id: &str,
    obj: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let spec = to_step_spec(py, obj)?;
    for tool in spec.tools() {
        g.tools.insert(tool.tool_name().to_string(), tool.clone());
    }
    g.library.register_step_arc(step_id, spec.step());
    g.nodes.insert_step(step_id, obj.clone().unbind());
    Ok(step_id.to_string())
}

// ── The catalog the runtime reads ──

/// The catalog as it stands *now* — filters and steps together.
///
/// A `Step` is immutable once built, so a study that samples a new prompt
/// or model has no way to change one in place — it writes to the live
/// `Agent` instead, and the steps are rebuilt from those here, before
/// every compile and every run. Cheap: rebuilding a step is reading a
/// handful of fields off a Python object.
///
/// Every entry point passes this one value, which is what stops
/// `compile()` from type-checking a different graph than `run()` does.
pub(super) fn rebuild_catalog(g: &PyGraph, py: Python<'_>) -> PyResult<NodeCatalog> {
    if !g.nodes.has_steps() {
        return Ok(g.library.clone());
    }
    let mut catalog = g.library.clone();
    for (node_id, obj) in g.nodes.steps() {
        catalog.register_step_arc(node_id, to_step_spec(py, obj.bind(py))?.step());
    }
    Ok(catalog)
}

/// The graph's filters, serialized for the wire.
///
/// Three call sites built this vector inline; the strategy path needs it
/// for a reason the others do not — see `register_filters_on`.
pub(super) fn serialized_filters(g: &PyGraph) -> Vec<somatize_worker::protocol::SerializedFilter> {
    g.graph
        .nodes
        .iter()
        .filter_map(|node| {
            let record = g.nodes.filter(&node.id)?;
            Some(somatize_worker::protocol::SerializedFilter {
                node_id: node.id.clone(),
                pickled_filter: record.pickled.clone(),
                state: g.library.get_state(&node.id).map(|arc| (*arc).clone()),
                requirements: record.requirements.clone(),
                trainable: record.trainable,
                config_hash: g.library.get(&node.id).map(|f| f.config_hash()),
            })
        })
        .collect()
}

// ── Queries ──

/// Does anything in this graph need fitting before it can run?
pub(super) fn has_trainable_filters(g: &PyGraph) -> bool {
    g.nodes.filters().any(|(_, f)| f.trainable)
}

/// Does any live filter declare itself differentiable?
///
/// `_differentiable` is a **declaration**, set by subclassing
/// `DifferentiableFilter`, and it sits beside `_kind`, `_cacheable`,
/// `_deterministic` and `_cache_version` — the class attributes this
/// package already uses to let a filter say what it is.
///
/// This used to sniff for a `build_module` method. That made the name of a
/// method load-bearing for the whole graph: any filter that happened to
/// define `build_module` for its own reasons switched the engine for every
/// node beside it, silently, and the failure showed up as a cache miss or
/// a lost seed rather than as an error.
pub(super) fn has_differentiable_filters(g: &PyGraph, py: Python<'_>) -> bool {
    g.nodes.filters().any(|(_, f)| {
        f.live
            .bind(py)
            .getattr("_differentiable")
            .and_then(|v| v.is_truthy())
            .unwrap_or(false)
    })
}

/// The live `Agent`/`Judge` behind each step node, as `(node_id, obj)`.
pub(super) fn steps(g: &PyGraph, py: Python<'_>) -> Vec<(String, PyObject)> {
    let mut items: Vec<(String, PyObject)> = g
        .nodes
        .steps()
        .map(|(id, obj)| (id.clone(), obj.clone_ref(py)))
        .collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    items
}

/// The full module source code for a filter node, or `None`.
pub(super) fn filter_source(g: &PyGraph, node_id: &str) -> Option<String> {
    g.nodes.filter(node_id).map(|f| f.source.clone())
}

/// Third-party distributions the worker must install to run `node_id`.
pub(super) fn filter_requirements(g: &PyGraph, node_id: &str) -> Option<Vec<String>> {
    g.nodes.filter(node_id).map(|f| f.requirements.clone())
}

/// Every filter's source, as `{node_id: source_code}`.
pub(super) fn filter_sources_dict(g: &PyGraph, py: Python<'_>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (node_id, record) in g.nodes.filters() {
        dict.set_item(node_id, &record.source)?;
    }
    Ok(dict.into_any().unbind())
}

/// The live Python filter instance registered under `node_id`.
pub(super) fn filter(g: &PyGraph, py: Python<'_>, node_id: &str) -> Option<PyObject> {
    g.nodes.filter(node_id).map(|f| f.live.clone_ref(py))
}

/// Node ids with live Python filter instances, in topological order.
///
/// Falls back to graph order — which is insertion order — if the
/// topological sort fails, as it can while the graph is still being
/// built. The fallback used to return the map's keys, so the training
/// walk that consumes this list chained filters in whatever order the
/// hash landed in.
pub(super) fn filter_ids(g: &PyGraph) -> Vec<String> {
    match g.graph.topological_sort() {
        Ok(sorted) => sorted
            .into_iter()
            .filter(|id| g.nodes.filter(id).is_some())
            .map(|id| id.to_string())
            .collect(),
        Err(_) => g
            .graph
            .nodes
            .iter()
            .map(|n| n.id.clone())
            .filter(|id| g.nodes.filter(id).is_some())
            .collect(),
    }
}

/// Live Python filter instances as `[(node_id, filter), ...]` in
/// topological order — a list, not a dict, so callers chaining forwards
/// thread their inputs correctly.
pub(super) fn filters(g: &PyGraph, py: Python<'_>) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for node_id in filter_ids(g) {
        if let Some(record) = g.nodes.filter(&node_id) {
            list.append((node_id, record.live.clone_ref(py)))?;
        }
    }
    Ok(list.into_any().unbind())
}

// ── Learned state ──

/// Store a Python state value for a filter node.
pub(super) fn set_node_state(
    g: &mut PyGraph,
    py: Python<'_>,
    node_id: String,
    state: Bound<'_, PyAny>,
) -> PyResult<()> {
    let value = py_to_value(py, &state)?;
    g.library
        .try_set_state(node_id, value)
        .map_err(soma_err_to_py)
}

/// The stored state value for a filter node, or `None`.
pub(super) fn get_node_state(
    g: &PyGraph,
    py: Python<'_>,
    node_id: &str,
) -> PyResult<Option<PyObject>> {
    match g.library.get_state(node_id) {
        Some(arc) => Ok(Some(value_to_py(py, arc.as_ref())?)),
        None => Ok(None),
    }
}
