//! The decided shape of an execution.
//!
//! A [`Graph`] says which nodes exist; a `Plan` says **how they are walked**,
//! and that is a separate decision: the same structure can run in sequence, all
//! at once, or spread across machines.
//!
//! It is an enum and not a trait of executors on purpose. The ways of executing
//! are a **closed** set that we decide, so the compiler keeps track: the day a
//! variant arrives, the engine's `match` stops compiling and someone has to
//! decide what to do, instead of falling into a wildcard arm.
//!
//! Every step carries **where its input comes from** written on it. That is what
//! makes a plan self-contained — executing it never looks at the graph again —
//! and it is why fans, in both directions, need no special variant.
//!
//! # How the shape is recovered
//!
//! [`compile`] does not flatten the graph, it **decomposes** it. The DSL's `>>`
//! and `|` are serial and parallel composition, so the tree you wrote is in
//! there — and it has to be recovered from the graph, because the same graph
//! built with `node()`/`edge()` in a loop must give the same plan (decision 6 of
//! CU5). The expression tree is the **oracle**, not the source.
//!
//! | case | yields |
//! |---|---|
//! | no nodes | [`Plan::Empty`] |
//! | one node | [`Plan::Execute`] |
//! | the subgraph splits into components | [`Plan::Wave`], one branch per component |
//! | there is a **series cut** | [`Plan::Sequence`] of the two sides |
//! | no cut | flat sequence: it is not series-parallel |
//!
//! A **series cut** `(A, B)` is what a `>>` produces: the crossing edges run
//! from **all** the sinks of `A` to **all** the sources of `B`, and from nowhere
//! else. Checking that in full is what keeps a multi-node branch from being
//! split down the middle. Only the **prefixes of a topological order** need
//! testing, and that is provable: in a serial composition every node of `A`
//! precedes every node of `B` in *any* topological order.
//!
//! There are DAGs without a tree — a theorem, not a gap here. The minimal
//! forbidden pattern is the "N": `a→c`, `a→d`, `b→d`. See Valdes, Tarjan and
//! Lawler, *The recognition of series parallel digraphs*, SIAM J. Comput. 11(2),
//! 1982. And there is a fortunate boundary: **the image of the DSL is exactly
//! the series-parallel graphs**, so the N can only be built with
//! `node()`/`edge()`, and those fall to the last case.
//!
//! # Distribution, which is a second step
//!
//! [`compile`] does not see the [`Placement`]; [`distribute`] does, and wraps
//! the slices running elsewhere in [`Plan::Remote`]. Two steps and not one
//! because the two halves of "where" do not weigh the same: a
//! [`Device`](crate::Device) is **inert** for the traversal, so placing cannot
//! alter what comes out of `compile`, and that stays an invariant. A [`Host`] is
//! not: crossing a wire is a step of another nature, and how often it is crossed
//! depends on how you group.

use crate::{Catalog, Graph, Host, NodeId, Placement};
use std::fmt;

/// How a graph is walked.
///
/// No `#[non_exhaustive]`: whoever executes has to decide for each variant.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Plan {
    /// Nothing to do.
    Empty,
    /// Advance one node until it finishes.
    Execute {
        /// Which one.
        node: NodeId,
        /// Where its input comes from. Empty = the graph's input.
        from: Vec<NodeId>,
    },
    /// One after another, in topological order. Each reads what it needs from
    /// what has already been produced.
    Sequence(Vec<Plan>),
    /// Branches launched **at the same time**, one per connected component —
    /// so they are disjoint, which is what the dropped `Plan::Parallel` was
    /// not. Each is a whole plan, so a branch runs start to finish on one
    /// thread.
    Wave(Vec<Plan>),
    /// This whole slice executes elsewhere. A complete plan and not a step, so
    /// a chain of five nodes on the same host is sent once.
    Remote {
        /// Where.
        host: Host,
        /// What runs there.
        inner: Box<Plan>,
    },
}

/// Decides how this graph is walked.
///
/// The catalog is only consulted to check that every node has an
/// implementation: since CU6 the shape no longer depends on **what** each one is.
pub fn compile(graph: &Graph, catalog: &Catalog) -> Result<Plan, CompileError> {
    if graph.is_empty() {
        return Ok(Plan::Empty);
    }

    let order = graph.topological_sort();
    for node in &order {
        if catalog.get(node).is_none() {
            return Err(CompileError::NoImplementation((*node).clone()));
        }
    }

    Ok(decompose(graph, &order))
}

/// Wraps the slices that run on another host in [`Plan::Remote`].
///
/// It groups **as much as it can**, descending only where a slice is spread
/// across several places. Idempotent, and a plan with no hosts comes out of
/// here unchanged.
pub fn distribute(plan: &Plan, placement: &Placement) -> Plan {
    if placement.is_local() {
        return plan.clone();
    }
    wrap(plan, placement)
}

/// Where everything inside a plan runs.
enum Where {
    /// There are no nodes, so it runs nowhere.
    Nothing,
    /// All in the same place: a host, or — with `None` — here.
    All(Option<Host>),
    /// In more than one place, so it has to be descended into and split.
    Mixed,
}

fn wrap(plan: &Plan, placement: &Placement) -> Plan {
    if matches!(plan, Plan::Remote { .. }) {
        return plan.clone();
    }
    match uniform(plan, placement) {
        Where::All(Some(host)) => Plan::Remote {
            host,
            inner: Box::new(plan.clone()),
        },
        Where::All(None) | Where::Nothing => plan.clone(),
        Where::Mixed => match plan {
            Plan::Sequence(plans) => Plan::Sequence(runs(plans, placement)),
            // Branches are wrapped one by one, without regrouping two of the
            // same host: that would change their declaration order, which is
            // observable. All on one host was already wrapped whole above.
            Plan::Wave(branches) => {
                Plan::Wave(branches.iter().map(|p| wrap(p, placement)).collect())
            }
            Plan::Empty | Plan::Execute { .. } | Plan::Remote { .. } => plan.clone(),
        },
    }
}

/// The steps of a sequence, merging **consecutive** runs bound for the same
/// host — consecutive only, because the order is the topological one.
fn runs(plans: &[Plan], placement: &Placement) -> Vec<Plan> {
    let mut out: Vec<Plan> = Vec::new();
    let mut run: Vec<Plan> = Vec::new();
    let mut destination: Option<Host> = None;

    for plan in plans {
        let here = match plan {
            Plan::Remote { .. } => None,
            _ => match uniform(plan, placement) {
                Where::All(Some(host)) => Some(host),
                Where::All(None) | Where::Nothing | Where::Mixed => None,
            },
        };
        if here != destination {
            close(&mut out, &mut run, destination.take());
            destination = here;
        }
        match destination {
            Some(_) => run.push(plan.clone()),
            None => out.push(wrap(plan, placement)),
        }
    }
    close(&mut out, &mut run, destination);
    out
}

/// Closes the open run, if any, as a single trip.
fn close(out: &mut Vec<Plan>, run: &mut Vec<Plan>, destination: Option<Host>) {
    let Some(host) = destination else { return };
    // A run of one is not wrapped in a sequence of one: the shape cannot depend
    // on how you arrived at it.
    let inner = match run.len() {
        1 => run.remove(0),
        _ => Plan::Sequence(std::mem::take(run)),
    };
    out.push(Plan::Remote {
        host,
        inner: Box::new(inner),
    });
}

/// Whether the whole plan lands in the same place.
fn uniform(plan: &Plan, placement: &Placement) -> Where {
    let mut places = Vec::new();
    hosts_in(plan, placement, &mut places);
    match places.split_first() {
        None => Where::Nothing,
        Some((first, rest)) if rest.iter().all(|host| host == first) => Where::All(first.clone()),
        Some(_) => Where::Mixed,
    }
}

/// The host of each node in the plan, with repeats. `None` means "here". A
/// [`Plan::Remote`] counts as its host and is not descended into.
fn hosts_in(plan: &Plan, placement: &Placement, out: &mut Vec<Option<Host>>) {
    match plan {
        Plan::Empty => {}
        Plan::Execute { node, .. } => out.push(placement.host_of(node).cloned()),
        Plan::Sequence(plans) | Plan::Wave(plans) => {
            for plan in plans {
                hosts_in(plan, placement, out);
            }
        }
        Plan::Remote { host, .. } => out.push(Some(host.clone())),
    }
}

/// The shape of a subset of nodes, in topological order.
///
/// The subset is always **closed under paths** — a topological prefix, its
/// complement, or a connected component — which is why reachability inside the
/// subgraph coincides with reachability in the whole graph.
fn decompose<'g>(graph: &'g Graph, nodes: &[&'g NodeId]) -> Plan {
    match nodes {
        [] => Plan::Empty,
        [only] => step(graph, only),
        _ => {
            let parts = components(graph, nodes);
            if parts.len() > 1 {
                return Plan::Wave(parts.iter().map(|part| decompose(graph, part)).collect());
            }

            let Some(cut) = series_cut(graph, nodes) else {
                // No cut, no tree: walked in sequence, as before waves existed.
                return Plan::Sequence(nodes.iter().map(|node| step(graph, node)).collect());
            };

            // Flattening the recursion on the right leaves `Sequence` with its
            // steps in a row rather than nested.
            let mut steps = vec![decompose(graph, &nodes[..cut])];
            match decompose(graph, &nodes[cut..]) {
                Plan::Sequence(rest) => steps.extend(rest),
                other => steps.push(other),
            }
            Plan::Sequence(steps)
        }
    }
}

/// A lone step, with where its input comes from — the whole graph's
/// predecessors, not the subset's.
fn step(graph: &Graph, node: &NodeId) -> Plan {
    Plan::Execute {
        node: node.clone(),
        from: graph.predecessors(node).into_iter().cloned().collect(),
    }
}

/// The connected components — ignoring direction — of the subgraph, each
/// keeping the input's topological order and ordered by their first node.
fn components<'g>(graph: &'g Graph, nodes: &[&'g NodeId]) -> Vec<Vec<&'g NodeId>> {
    let mut unassigned: Vec<bool> = vec![true; nodes.len()];
    let mut out = Vec::new();

    for start in 0..nodes.len() {
        if !unassigned[start] {
            continue;
        }
        unassigned[start] = false;
        let mut group = vec![start];
        let mut frontier = vec![start];

        while let Some(i) = frontier.pop() {
            for j in 0..nodes.len() {
                if unassigned[j] && adjacent(graph, nodes[i], nodes[j]) {
                    unassigned[j] = false;
                    group.push(j);
                    frontier.push(j);
                }
            }
        }

        group.sort_unstable();
        out.push(group.into_iter().map(|i| nodes[i]).collect());
    }
    out
}

/// Whether there is an edge between the two, in either direction.
fn adjacent(graph: &Graph, a: &NodeId, b: &NodeId) -> bool {
    graph.successors(a).contains(&b) || graph.successors(b).contains(&a)
}

/// Where the sequence splits: the smallest series cut, if there is one.
fn series_cut(graph: &Graph, nodes: &[&NodeId]) -> Option<usize> {
    (1..nodes.len()).find(|cut| is_series_cut(graph, &nodes[..*cut], &nodes[*cut..]))
}

/// Whether `before >> after` is exactly what lies between the two: nothing
/// crosses outside the ends, and every sink reaches every source.
fn is_series_cut(graph: &Graph, before: &[&NodeId], after: &[&NodeId]) -> bool {
    let sinks: Vec<&NodeId> = before
        .iter()
        .copied()
        .filter(|node| !graph.successors(node).iter().any(|s| before.contains(s)))
        .collect();
    let sources: Vec<&NodeId> = after
        .iter()
        .copied()
        .filter(|node| !graph.predecessors(node).iter().any(|p| after.contains(p)))
        .collect();

    let crosses_outside_the_ends = before.iter().any(|node| {
        graph
            .successors(node)
            .iter()
            .any(|succ| after.contains(succ) && !(sinks.contains(node) && sources.contains(succ)))
    });
    if crosses_outside_the_ends {
        return false;
    }

    sinks.iter().all(|sink| {
        let onward = graph.successors(sink);
        sources.iter().all(|source| onward.contains(source))
    })
}

/// Why it was not possible to decide how to walk the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// The node is in the graph but nobody registered what it does.
    NoImplementation(NodeId),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoImplementation(id) => {
                write!(f, "node `{id}` has no registered implementation")
            }
        }
    }
}

impl std::error::Error for CompileError {}
