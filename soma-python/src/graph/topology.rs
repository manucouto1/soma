//! What is wired to what.
//!
//! The nodes and edges themselves, and the two kinds a study may vary:
//! an optional data edge it can cut, and the control edges a loop or a
//! branch is made of. The user-facing documentation for each of these
//! lives on the `#[pymethods]` signature in [`super`] — that is what
//! `help(Graph.branch)` prints — so what is written here is the reason
//! the code is shaped the way it is.

use super::{PyGraph, registry};
use crate::prelude::*;

// ── Building blocks ──

/// A node id not yet taken, suffixing `_2`, `_3`, … as needed.
pub(super) fn free_id(g: &PyGraph, wanted: &str) -> String {
    if g.graph.node(wanted).is_none() {
        return wanted.to_string();
    }
    let mut i = 2;
    loop {
        let candidate = format!("{wanted}_{i}");
        if g.graph.node(&candidate).is_none() {
            return candidate;
        }
        i += 1;
    }
}

/// Resolve one arm of a branch or one entry of a loop body: either the id
/// of a node already in the graph, or a filter/agent to add as one.
fn resolve_member(
    g: &mut PyGraph,
    py: Python<'_>,
    fallback_id: &str,
    obj: &Bound<'_, PyAny>,
) -> PyResult<String> {
    if let Ok(existing) = obj.extract::<String>() {
        if g.graph.node(&existing).is_none() {
            return Err(PyValueError::new_err(format!(
                "`{existing}` names no node in this graph. Pass a filter or an \
                 agent to create one, or an id already added with node()"
            )));
        }
        return Ok(existing);
    }

    let id = free_id(g, fallback_id);
    let node = registry::register_behaviour(g, py, &id, obj)?.node(&id);
    g.graph.add_node(node);
    Ok(id)
}

/// Add a labelled control edge — the wire the compiler reads to decide
/// which nodes a loop or branch owns.
fn control_edge(g: &mut PyGraph, source: &str, target: &str, label: Option<&str>) {
    let id = format!("e_{}", g.graph.edges.len());
    let mut edge = Edge::control(id, source, target);
    if let Some(label) = label {
        edge = edge.with_label(label);
    }
    g.graph.add_edge(edge);
}

// ── Nodes ──

/// Add a filter node, from `node(obj)` or `node(id, obj)`.
pub(super) fn node(
    g: &mut PyGraph,
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
            (to_snake_case(&class_name), filter_obj.to_owned())
        }
        2 => {
            let id = args.get_item(0)?.extract::<String>()?;
            (id, args.get_item(1)?.to_owned())
        }
        n => {
            return Err(PyValueError::new_err(format!(
                "node() takes 1 or 2 positional arguments, got {n}"
            )));
        }
    };

    // An Agent or a Judge is a node too — it just runs a turn loop instead
    // of a function. `register_behaviour` dispatches, so there is one way
    // to add a node rather than a second method whose name would collide
    // with the optimiser's `step()`.
    let actual_id = free_id(g, &node_id);
    let mut node = registry::register_behaviour(g, py, &actual_id, &filter_obj)?.node(&actual_id);
    if let Some(t) = target {
        node = node.with_target(t);
    }
    g.graph.add_node(node);

    Ok(actual_id)
}

/// Add a node that runs a condition and executes only the arm it names.
///
/// The arms are declared, so the compiler can reject one that no edge
/// reaches and one that no arm declares — the silent-drop failure the
/// multi-agent literature files under inter-agent misalignment.
pub(super) fn branch(
    g: &mut PyGraph,
    py: Python<'_>,
    node_id: String,
    condition: &Bound<'_, PyAny>,
    arms: &Bound<'_, PyDict>,
    target: Option<String>,
) -> PyResult<String> {
    if arms.is_empty() {
        return Err(PyValueError::new_err(
            "branch() needs at least one arm; a router with nowhere to \
             route is just a node",
        ));
    }

    let actual_id = free_id(g, &node_id);
    // The branch node *is* the condition: the executor runs it and reads
    // the arm label from its output.
    registry::register_behaviour(g, py, &actual_id, condition)?;

    let labels: Vec<String> = arms
        .keys()
        .iter()
        .map(|k| k.extract::<String>())
        .collect::<PyResult<_>>()?;

    let mut node = Node::branch_over(&actual_id, labels);
    if let Some(t) = target {
        node = node.with_target(t);
    }
    g.graph.add_node(node);

    for (key, value) in arms.iter() {
        let label = key.extract::<String>()?;
        let arm_id = resolve_member(g, py, &label, &value)?;
        control_edge(g, &actual_id, &arm_id, Some(&label));
    }

    Ok(actual_id)
}

/// Add a node that repeats a body until it signals completion.
pub(super) fn loop_(
    g: &mut PyGraph,
    py: Python<'_>,
    node_id: String,
    body: &Bound<'_, PyAny>,
    until: Option<&Bound<'_, PyAny>>,
    max_iterations: Option<usize>,
) -> PyResult<String> {
    use somatize_core::graph::control::LoopCondition;

    // One entry or several: a list is the general case, a bare value the
    // one people write.
    let entries: Vec<Bound<'_, PyAny>> = match body.try_iter() {
        Ok(iter) if !body.is_instance_of::<pyo3::types::PyString>() => {
            iter.collect::<PyResult<_>>()?
        }
        _ => vec![body.clone()],
    };
    if entries.is_empty() {
        return Err(PyValueError::new_err("loop() needs a body"));
    }

    let actual_id = free_id(g, &node_id);
    let until = match until {
        None => LoopCondition::BodyTerminal,
        // `False` is the only bool that means anything here: "run the
        // whole count". `True` would have to mean "stop immediately",
        // which nobody writes on purpose.
        Some(u) if u.is_instance_of::<pyo3::types::PyBool>() => {
            if u.extract::<bool>()? {
                return Err(PyValueError::new_err(
                    "until=True says the loop stops before it runs. Pass a node \
                     id to read the signal from, or False to run the full count",
                ));
            }
            LoopCondition::Exhaust
        }
        Some(u) => {
            let cond = u.extract::<String>()?;
            if g.graph.node(&cond).is_none() {
                return Err(PyValueError::new_err(format!(
                    "`{cond}` names no node in this graph, so it cannot be the \
                     loop's stop condition"
                )));
            }
            LoopCondition::WhenSignaled(cond)
        }
    };

    g.graph
        .add_node(Node::loop_until(&actual_id, max_iterations, until));

    for (i, entry) in entries.iter().enumerate() {
        let fallback = format!("{actual_id}_body_{i}");
        let entry_id = resolve_member(g, py, &fallback, entry)?;
        control_edge(g, &actual_id, &entry_id, None);
    }

    Ok(actual_id)
}

// ── Edges ──

/// Connect two nodes with a data edge.
pub(super) fn edge(g: &mut PyGraph, source: String, target: String) {
    let id = format!("e_{}", g.graph.edges.len());
    g.graph.add_edge(Edge::data(id, source, target));
}

/// Every data and control edge, as `(source, target)` in insertion order.
pub(super) fn edges(g: &PyGraph) -> Vec<(String, String)> {
    g.graph
        .edges
        .iter()
        .map(|e| (e.source.clone(), e.target.clone()))
        .collect()
}

/// Declare that `source` may hand control to `target` — what `Goto` needs.
pub(super) fn handoff(g: &mut PyGraph, source: &str, target: &str) {
    control_edge(g, source, target, None);
}

/// Make a data edge part of the search space: a study may keep it or cut
/// it. Control edges are not eligible — they are what makes a loop a loop,
/// not a design choice.
pub(super) fn optional(g: &mut PyGraph, source: String, target: String) -> PyResult<()> {
    let found = g
        .graph
        .edges
        .iter()
        .find(|e| e.source == source && e.target == target);

    match found {
        None => Err(PyValueError::new_err(format!(
            "there is no edge `{source}` → `{target}` to make optional"
        ))),
        Some(e) if e.kind != somatize_core::graph::EdgeKind::Data => {
            Err(PyValueError::new_err(format!(
                "`{source}` → `{target}` is a control edge; cutting it would \
                 change what the loop or branch owns, not just what flows"
            )))
        }
        Some(_) => {
            let pair = (source, target);
            if !g.optional_edges.contains(&pair) {
                g.optional_edges.push(pair);
            }
            Ok(())
        }
    }
}

/// Keep or cut one of the optional edges.
///
/// A cut edge is set aside whole, so restoring it restores its id, kind
/// and label — a trial that cuts an edge must leave the graph identical to
/// the one the next trial starts from.
pub(super) fn set_edge(
    g: &mut PyGraph,
    source: String,
    target: String,
    enabled: bool,
) -> PyResult<()> {
    let pair = (source, target);
    if !g.optional_edges.contains(&pair) {
        return Err(PyValueError::new_err(format!(
            "`{}` → `{}` was never declared optional; call optional() first",
            pair.0, pair.1
        )));
    }

    if enabled {
        if let Some((at, edge)) = g.cut_edges.remove(&pair) {
            // Back where it was, not on the end. Appending would leave a
            // graph that is semantically the same and renders, hashes and
            // fingerprints differently — so two trials of the same topology
            // would not compare equal.
            g.graph.edges.insert(at.min(g.graph.edges.len()), edge);
        }
    } else if !g.cut_edges.contains_key(&pair)
        && let Some(i) = g
            .graph
            .edges
            .iter()
            .position(|e| e.source == pair.0 && e.target == pair.1)
    {
        let edge = g.graph.edges.remove(i);
        g.cut_edges.insert(pair, (i, edge));
    }
    Ok(())
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
