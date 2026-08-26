//! The decided shape of an execution.
//!
//! A [`Graph`] says which nodes exist; a `Plan` says **how they are walked**.
//! An enum and not a trait of executors: the ways of executing are a closed set,
//! so the day a variant arrives the engine's `match` stops compiling and
//! somebody has to decide, instead of falling into a wildcard arm.
//!
//! Every step carries **where its input comes from**, which is what makes a plan
//! self-contained — executing never looks at the graph again — and why fans in
//! both directions need no special variant.
//!
//! [`compile`] does not flatten the graph, it **decomposes** it, recovering the
//! tree from the graph and never from the expression: the same graph built with
//! `node()`/`edge()` in a loop has to give the same plan.
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
//! from **all** the sinks of `A` to **all** the sources of `B` and from nowhere
//! else. Only the prefixes of a topological order need testing, since in a
//! serial composition every node of `A` precedes every node of `B` in any
//! topological order.
//!
//! There are DAGs without a tree — a theorem, not a gap here. The minimal
//! forbidden pattern is the "N": `a→c`, `a→d`, `b→d`. See Valdes, Tarjan and
//! Lawler, *The recognition of series parallel digraphs*, SIAM J. Comput. 11(2),
//! 1982. The image of the DSL is **exactly** the series-parallel graphs, so the
//! N is only reachable through `node()`/`edge()` and falls to the last case.
//!
//! [`compile`] does not see the [`Placement`]; [`distribute`] does, and wraps
//! what runs elsewhere in [`Plan::Remote`]. Two steps because a
//! [`Device`](crate::Device) is inert for the traversal and a [`Host`] is not.

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
    /// Branches launched **at the same time**, one per connected component, so
    /// they are disjoint. Each is a whole plan, so a branch runs start to
    /// finish on one thread.
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

/// Decides how this graph is walked. The catalog is only consulted to check
/// that every node has an implementation: the shape does not depend on what
/// each one is.
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

/// One step of a plan: a node, and where its input comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step<'p> {
    /// Which node.
    pub node: &'p NodeId,
    /// Which nodes it reads. Empty means the graph's input.
    pub from: &'p [NodeId],
}

/// What decides where one part of a plan runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Destination<'p> {
    /// A node, whose host — if it has one — the [`Placement`] knows.
    Node(&'p NodeId),
    /// A slice that already says where it goes.
    Away(&'p Host),
}

impl Plan {
    /// Every step, in declaration order, **wherever it runs**: a
    /// [`Remote`](Plan::Remote) is entered, because what a plan does does not
    /// depend on where.
    pub fn steps(&self) -> impl Iterator<Item = Step<'_>> {
        Steps { left: vec![self] }
    }

    /// What decides where each part of this plan runs, in declaration order.
    /// Differs from [`steps`](Self::steps) in one line: a
    /// [`Remote`](Plan::Remote) is **not** entered, which is what makes
    /// [`distribute`] idempotent.
    pub fn destinations(&self) -> impl Iterator<Item = Destination<'_>> {
        Destinations { left: vec![self] }
    }
}

/// The stack of [`Plan::steps`]. Children are pushed in reverse so that popping
/// gives them back in the order they were declared, which is observable.
struct Steps<'p> {
    left: Vec<&'p Plan>,
}

impl<'p> Iterator for Steps<'p> {
    type Item = Step<'p>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(plan) = self.left.pop() {
            match plan {
                Plan::Empty => {}
                Plan::Execute { node, from } => return Some(Step { node, from }),
                Plan::Sequence(plans) | Plan::Wave(plans) => self.left.extend(plans.iter().rev()),
                Plan::Remote { inner, .. } => self.left.push(inner),
            }
        }
        None
    }
}

/// The stack of [`Plan::destinations`]. The same walk, stopping where the other
/// descends.
struct Destinations<'p> {
    left: Vec<&'p Plan>,
}

impl<'p> Iterator for Destinations<'p> {
    type Item = Destination<'p>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(plan) = self.left.pop() {
            match plan {
                Plan::Empty => {}
                Plan::Execute { node, .. } => return Some(Destination::Node(node)),
                Plan::Sequence(plans) | Plan::Wave(plans) => self.left.extend(plans.iter().rev()),
                Plan::Remote { host, .. } => return Some(Destination::Away(host)),
            }
        }
        None
    }
}

/// Wraps the slices that run on another host in [`Plan::Remote`], grouping as
/// much as it can and descending only where a slice is spread across places.
/// Idempotent; a plan with no hosts comes out unchanged.
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
            // One by one, without regrouping two of the same host: that would
            // change their declaration order, which is observable.
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

/// Whether the whole plan lands in the same place. `None` means "here".
fn uniform(plan: &Plan, placement: &Placement) -> Where {
    let places: Vec<Option<Host>> = plan
        .destinations()
        .map(|destination| match destination {
            Destination::Node(node) => placement.host_of(node).cloned(),
            Destination::Away(host) => Some(host.clone()),
        })
        .collect();
    match places.split_first() {
        None => Where::Nothing,
        Some((first, rest)) if rest.iter().all(|host| host == first) => Where::All(first.clone()),
        Some(_) => Where::Mixed,
    }
}

/// The shape of a subset of nodes, in topological order. The subset is always
/// closed under paths, which is why reachability inside the subgraph coincides
/// with reachability in the whole graph.
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
