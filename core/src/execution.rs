//! The engine: walking a [`Plan`] and executing what it says.
//!
//! It lives in the core, not in the bindings, because walking *is* domain
//! logic. Python only supplies the implementations.
//!
//! The engine never looks at the graph: every plan step carries where its input
//! comes from, and here it is only looked up in what was already produced.
//!
//! Every node is advanced the same way — ask, serve whatever it asks for, ask
//! again — so a node that finishes on the first turn needs no separate path.
//!
//! When **one** thing reaches a node it receives that thing; when several do, a
//! [`Value::Map`] keyed by whoever produced each. Fan-in is neither a plan
//! variant nor a kind of node, it is the shape an input with several origins
//! takes; aggregating them is the receiving node's job, i.e. library.

use crate::{
    Cargo, Catalog, Ctx, Device, Driver, DriverError, Host, NodeError, NodeId, Outcome, Placement,
    Plan, Transition, Transport, TransportError, Value,
};
use std::collections::HashMap;
use std::fmt;

/// How many times a node is asked before it is given up for hung. A node that
/// does not finish is a bug in the node, not a legitimate wait.
const MAX_TURNS: usize = 64;

/// Executes plans.
///
/// A type and not a bare function because executing needs context: today the
/// store, the driver, the placement and the transports.
pub struct Executor<'a> {
    catalog: &'a Catalog,
    driver: Option<&'a dyn Driver>,
    placement: Option<&'a Placement>,
    /// Which host it knows how to reach, and by what route. A list because
    /// there are two or three of them.
    transports: Vec<(Host, &'a dyn Transport)>,
}

impl<'a> Executor<'a> {
    /// An executor with no driver: a plan whose steps ask for something fails
    /// with [`RunError::NoDriver`].
    pub fn new(catalog: &'a Catalog) -> Self {
        Self {
            catalog,
            driver: None,
            placement: None,
            transports: Vec::new(),
        }
    }

    /// The same executor, with whoever will serve what the steps ask for.
    pub fn with_driver(mut self, driver: &'a dyn Driver) -> Self {
        self.driver = Some(driver);
        self
    }

    /// The same executor, knowing where each node runs. Without this every
    /// `ctx.device` is `None`, which means "wherever it lands".
    pub fn placed(mut self, placement: &'a Placement) -> Self {
        self.placement = Some(placement);
        self
    }

    /// The same executor, knowing how to reach a host. Called once per host; a
    /// name nobody resolves is [`RunError::NoTransport`], not a slice executed
    /// here just in case.
    pub fn reaching(mut self, host: impl Into<Host>, transport: &'a dyn Transport) -> Self {
        self.transports.push((host.into(), transport));
        self
    }

    /// Executes the plan and returns what it produced. The first failure stops
    /// the execution.
    pub fn run(&self, plan: &Plan, input: Value) -> Result<Value, RunError> {
        let mut produced: HashMap<NodeId, Value> = HashMap::new();
        let last = self.walk(plan, &input, &mut produced)?;

        // A graph's output is that of its leaves: one leaf gives that value,
        // several a map keyed by each — the same shape as an input with several
        // origins, so a diamond comes back round.
        let leaves = terminals(plan);
        Ok(match leaves.as_slice() {
            [] | [_] => last,
            many => Value::map(
                many.iter()
                    .map(|id| {
                        let value = produced
                            .get(id)
                            .cloned()
                            .expect("the walk executed every step of the plan");
                        (id.to_string(), value)
                    })
                    .collect::<Vec<_>>(),
            ),
        })
    }

    /// Executes a slice that already knows what came before: what a worker does
    /// on receiving one.
    ///
    /// `known` is fed in as if this very run had produced it, so the steps read
    /// it through their `from`. What comes back does **not** include it, and is
    /// ordered by id because this crosses a process boundary.
    pub fn resume(
        &self,
        plan: &Plan,
        input: Value,
        known: Vec<(NodeId, Value)>,
    ) -> Result<Outcome, RunError> {
        let mut produced: HashMap<NodeId, Value> = known.into_iter().collect();
        let brought: Vec<NodeId> = produced.keys().cloned().collect();

        let last = self.walk(plan, &input, &mut produced)?;

        produced.retain(|id, _| !brought.contains(id));
        let mut mine: Vec<(NodeId, Value)> = produced.into_iter().collect();
        mine.sort_by(|(a, _), (b, _)| a.cmp(b));
        Ok(Outcome {
            last,
            produced: mine,
        })
    }

    /// Executes a plan, noting what each node produces, and returns the output
    /// of its last step.
    fn walk(
        &self,
        plan: &Plan,
        graph_input: &Value,
        produced: &mut HashMap<NodeId, Value>,
    ) -> Result<Value, RunError> {
        match plan {
            Plan::Empty => Ok(graph_input.clone()),
            Plan::Execute { node, from } => {
                let input = gather(from, graph_input, produced);
                let output = self.advance(node, input)?;
                produced.insert(node.clone(), output.clone());
                Ok(output)
            }
            Plan::Sequence(plans) => {
                let mut last = graph_input.clone();
                for plan in plans {
                    last = self.walk(plan, graph_input, produced)?;
                }
                Ok(last)
            }
            Plan::Wave(branches) => self.at_once(branches, graph_input, produced),
            Plan::Remote { host, inner } => self.elsewhere(host, inner, graph_input, produced),
        }
    }

    /// Launches a wave's branches at once and merges what they produced.
    ///
    /// Each branch starts with a **copy** of what was produced so far and
    /// returns only its own; being connected components, what each adds is
    /// disjoint and merging cannot clobber anything. Copying is cheap — a
    /// `Value` clones by `Arc` — and in exchange there is not a single lock.
    fn at_once(
        &self,
        branches: &[Plan],
        graph_input: &Value,
        produced: &mut HashMap<NodeId, Value>,
    ) -> Result<Value, RunError> {
        let earlier: &HashMap<NodeId, Value> = produced;
        let outcomes = std::thread::scope(|scope| {
            let running: Vec<_> = branches
                .iter()
                .map(|branch| {
                    scope.spawn(move || {
                        let mut mine = earlier.clone();
                        let last = self.walk(branch, graph_input, &mut mine)?;
                        mine.retain(|id, _| !earlier.contains_key(id));
                        Ok::<_, RunError>((last, mine))
                    })
                })
                .collect();
            running
                .into_iter()
                .map(|handle| match handle.join() {
                    Ok(outcome) => outcome,
                    // Not swallowed: `scope` has already waited on the others.
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect::<Vec<_>>()
        });

        for outcome in outcomes {
            // The first to fail **in declaration order**, not in time.
            let (_, mine) = outcome?;
            produced.extend(mine);
        }

        // A wave has no single output: its branches end in several places.
        Ok(Value::Null)
    }

    /// Sends a slice elsewhere and merges whatever comes back. It is given only
    /// what it reads and does not produce, because the wire is the expensive
    /// part.
    fn elsewhere(
        &self,
        host: &Host,
        inner: &Plan,
        graph_input: &Value,
        produced: &mut HashMap<NodeId, Value>,
    ) -> Result<Value, RunError> {
        let transport = self
            .transports
            .iter()
            .find(|(known, _)| known == host)
            .map(|(_, transport)| *transport)
            .ok_or_else(|| RunError::NoTransport(host.clone()))?;

        let known: Vec<(NodeId, Value)> = needs(inner)
            .into_iter()
            .filter_map(|id| produced.get(&id).map(|value| (id, value.clone())))
            .collect();

        let nowhere = Placement::new();
        let cargo = Cargo {
            input: graph_input,
            known: &known,
            placement: self.placement.unwrap_or(&nowhere),
        };
        let outcome = transport
            .dispatch(inner, &cargo)
            .map_err(|source| RunError::Transport {
                host: host.clone(),
                source,
            })?;

        produced.extend(outcome.produced);
        Ok(outcome.last)
    }

    /// Ask, serve whatever it asks for, ask again. Until it finishes.
    fn advance(&self, node: &NodeId, input: Value) -> Result<Value, RunError> {
        let implementation = self.implementation(node)?;
        let device = self.device(node);
        let mut results: Vec<Value> = Vec::new();

        for turn in 0..MAX_TURNS {
            let ctx = Ctx {
                turn,
                results: &results,
                device,
            };
            let transition =
                implementation
                    .forward(&input, &ctx)
                    .map_err(|source| RunError::Node {
                        node: node.clone(),
                        source,
                    })?;
            match transition {
                Transition::Done(output) => return Ok(output),
                Transition::Await(requests) => {
                    let driver = self
                        .driver
                        .ok_or_else(|| RunError::NoDriver(node.clone()))?;
                    results = driver
                        .perform(&requests)
                        .map_err(|source| RunError::Driver {
                            node: node.clone(),
                            source,
                        })?;
                }
            }
        }
        Err(RunError::TurnLimit {
            node: node.clone(),
            turns: MAX_TURNS,
        })
    }

    /// Where this node was said to run. Without a placement, nowhere.
    fn device(&self, node: &NodeId) -> Option<&'a Device> {
        self.placement.and_then(|placement| placement.of(node))
    }

    /// What the catalog has registered for this node.
    fn implementation(&self, node: &NodeId) -> Result<&std::sync::Arc<dyn crate::Node>, RunError> {
        self.catalog
            .get(node)
            .ok_or_else(|| RunError::NoImplementation(node.clone()))
    }
}

/// Why the execution could not be finished.
///
/// The structural things were already ruled out in [`compile`](crate::compile).
/// What is here are failures of the implementations, or of a plan that does not
/// match its catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// The plan names a node this catalog does not know.
    NoImplementation(NodeId),
    /// The node failed.
    Node {
        /// Where it happened.
        node: NodeId,
        /// What it said.
        source: NodeError,
    },
    /// The node asked for something and there is nobody to serve it.
    NoDriver(NodeId),
    /// The driver could not serve what the node asked for.
    Driver {
        /// Which node the request came from.
        node: NodeId,
        /// What the driver said.
        source: DriverError,
    },
    /// The plan sends a slice to a host nobody knows how to reach.
    NoTransport(Host),
    /// The transport could not carry the slice, or what ran there failed.
    Transport {
        /// Which host it was bound for.
        host: Host,
        /// What the transport said.
        source: TransportError,
    },
    /// The node kept asking for turns without ever finishing.
    TurnLimit {
        /// Which one.
        node: NodeId,
        /// How many turns it was given.
        turns: usize,
    },
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoImplementation(id) => {
                write!(f, "node `{id}` has no registered implementation")
            }
            Self::Node { node, source } => write!(f, "node `{node}` failed: {source}"),
            Self::NoDriver(node) => write!(
                f,
                "`{node}` asked for something and this executor has no driver to serve it"
            ),
            Self::Driver { node, source } => {
                write!(f, "serving what `{node}` asked for: {source}")
            }
            Self::NoTransport(host) => write!(
                f,
                "there is a slice placed on `{host}` and this executor cannot reach it"
            ),
            Self::Transport { host, source } => write!(f, "carrying a slice to `{host}`: {source}"),
            Self::TurnLimit { node, turns } => write!(
                f,
                "`{node}` spent all {turns} turns without finishing; it probably cannot stop"
            ),
        }
    }
}

impl std::error::Error for RunError {}

/// What a node receives: nothing → the graph's input, one thing → that thing,
/// several → a map keyed by whoever produced each, in edge declaration order.
fn gather(from: &[NodeId], graph_input: &Value, produced: &HashMap<NodeId, Value>) -> Value {
    let recall = |id: &NodeId| {
        produced
            .get(id)
            .cloned()
            .expect("topological order already executed the predecessors")
    };
    match from {
        [] => graph_input.clone(),
        [single] => recall(single),
        many => Value::map(
            many.iter()
                .map(|id| (id.to_string(), recall(id)))
                .collect::<Vec<_>>(),
        ),
    }
}

/// What this plan reads and does not produce: what has to travel with it.
fn needs(plan: &Plan) -> Vec<NodeId> {
    let mut produced = Vec::new();
    let mut consumed = Vec::new();
    collect(plan, &mut produced, &mut consumed);

    let mut out: Vec<NodeId> = Vec::new();
    for id in consumed {
        if !produced.contains(&id) && !out.contains(&id) {
            out.push(id);
        }
    }
    out
}

/// The plan's nodes whose output no other node reads: the leaves.
fn terminals(plan: &Plan) -> Vec<NodeId> {
    let mut produced = Vec::new();
    let mut consumed = Vec::new();
    collect(plan, &mut produced, &mut consumed);
    produced
        .into_iter()
        .filter(|id| !consumed.contains(id))
        .collect()
}

/// What each step produces and what it reads. Neither when nor where matters
/// here, so waves, sequences and remotes walk the same.
fn collect(plan: &Plan, produced: &mut Vec<NodeId>, consumed: &mut Vec<NodeId>) {
    match plan {
        Plan::Empty => {}
        Plan::Execute { node, from } => {
            produced.push(node.clone());
            consumed.extend(from.iter().cloned());
        }
        Plan::Sequence(plans) | Plan::Wave(plans) => {
            for plan in plans {
                collect(plan, produced, consumed);
            }
        }
        Plan::Remote { inner, .. } => collect(inner, produced, consumed),
    }
}
