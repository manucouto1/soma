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
//!
//! # The keys travel beside what was produced, not inside it
//!
//! A cached node has to be named before it runs, and its name is built out of
//! the names of what it reads — so a [`Key`] has to reach it along the same
//! edge its input does. labchain hangs that hash **inside** the value, because
//! it has no engine: the pipeline is the user's code and the only place left to
//! put one is the data. Here the walk already carries `produced` everywhere, so
//! the keys go in a **table beside it** — same lifetime, same copy into a
//! wave's branches, same retention in [`Executor::resume`] — and neither
//! [`Value`] grows a wrapper every `forward` would have to unwrap, nor does the
//! [`Node`](crate::Node) contract change.
//!
//! Nothing of this happens without both [`Executor::remembering`] and
//! [`Executor::keeping`]: with either missing the table stays empty and the
//! engine is what it was. They are two calls and not one because they are two
//! kinds of thing — what is remembered is **declared** and travels, a keeper is
//! **injected** and does not.

use crate::{
    Cargo, Catalog, Ctx, Device, Driver, DriverError, Host, Keeper, Kept, Key, Memory, NodeError,
    NodeId, Outcome, Placement, Plan, Transition, Transport, TransportError, Value,
};
use std::collections::HashMap;
use std::fmt;

/// How many times a node is asked before it is given up for hung. A node that
/// does not finish is a bug in the node, not a legitimate wait.
const MAX_TURNS: usize = 64;

/// What the engine writes beside a value it keeps. Private because the engine
/// is the only one that writes it and the only one that reads it back: to a
/// [`Keeper`] the metadata is text it hands over untouched.
const NODE: &str = "node";
const FINGERPRINT: &str = "fingerprint";

/// Executes plans.
///
/// A type and not a bare function because executing needs context: today the
/// store, the driver, the placement and the transports.
pub struct Executor<'a> {
    catalog: &'a Catalog,
    driver: Option<&'a dyn Driver>,
    placement: Option<&'a Placement>,
    /// What is remembered about each node. **Declared**, like the placement: it
    /// belongs to whoever wrote the graph, and it travels.
    memory: Option<&'a Memory>,
    /// Who hashes and where what is named ends up. **Injected**, like the driver
    /// and the transports: it belongs to whoever runs, and it does not travel.
    keeper: Option<&'a dyn Keeper>,
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
            memory: None,
            keeper: None,
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

    /// The same executor, knowing what is remembered about each node.
    ///
    /// **Declared and not injected**, which is why it is its own call and not
    /// half of [`keeping`](Self::keeping): what is remembered belongs to the
    /// graph, so it **travels** with a slice that goes to another host — and a
    /// worker that keeps things is of no use if nobody told it what any of the
    /// nodes are. Whoever coordinates may perfectly well keep nothing itself.
    ///
    /// Whether what it declares can honestly be kept is
    /// [`cacheable`](crate::cacheable)'s question, and it is asked by whoever
    /// has the graph: the engine never looks at one.
    pub fn remembering(mut self, memory: &'a Memory) -> Self {
        self.memory = Some(memory);
        self
    }

    /// The same executor, with somewhere to keep what it names.
    ///
    /// **Injected**, like the driver and the transports: it is whoever runs who
    /// says where things are kept, and it does not travel. Without it — or
    /// without [`remembering`](Self::remembering) — the keys are not even
    /// computed, which is why a graph that declares nothing pays nothing.
    pub fn keeping(mut self, keeper: &'a dyn Keeper) -> Self {
        self.keeper = Some(keeper);
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
        let mut keys: HashMap<NodeId, Key> = HashMap::new();
        let last = self.walk(plan, &input, &mut produced, &mut keys)?;

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
    /// it through their `from`, and `named` likewise: a slice that arrives
    /// without the keys of what it reads can name nothing it produces, which is
    /// a cache that stops at the process boundary rather than an error.
    ///
    /// What comes back does **not** include either of them, and both are ordered
    /// by id because this crosses a process boundary.
    pub fn resume(
        &self,
        plan: &Plan,
        input: Value,
        known: Vec<(NodeId, Value)>,
        named: Vec<(NodeId, Key)>,
    ) -> Result<Outcome, RunError> {
        let mut produced: HashMap<NodeId, Value> = known.into_iter().collect();
        let mut keys: HashMap<NodeId, Key> = named.into_iter().collect();
        let brought: Vec<NodeId> = produced.keys().cloned().collect();
        let named: Vec<NodeId> = keys.keys().cloned().collect();

        let last = self.walk(plan, &input, &mut produced, &mut keys)?;

        produced.retain(|id, _| !brought.contains(id));
        keys.retain(|id, _| !named.contains(id));
        Ok(Outcome {
            last,
            produced: sorted(produced),
            keys: sorted(keys),
        })
    }

    /// Executes a plan, noting what each node produces, and returns the output
    /// of its last step.
    fn walk(
        &self,
        plan: &Plan,
        graph_input: &Value,
        produced: &mut HashMap<NodeId, Value>,
        keys: &mut HashMap<NodeId, Key>,
    ) -> Result<Value, RunError> {
        match plan {
            Plan::Empty => Ok(graph_input.clone()),
            Plan::Execute { node, from } => {
                let key = self.key_for(node, from, graph_input, keys);
                if let Some(key) = &key {
                    keys.insert(node.clone(), key.clone());
                }
                // A hit is the whole point: the node is not advanced, and its
                // input is not even assembled.
                let output = match self.recalled(node, key.as_ref()) {
                    Some(kept) => kept,
                    None => {
                        let input = gather(from, graph_input, produced);
                        let output = self.advance(node, input)?;
                        self.keep(node, key.as_ref(), &output);
                        output
                    }
                };
                produced.insert(node.clone(), output.clone());
                Ok(output)
            }
            Plan::Sequence(plans) => {
                let mut last = graph_input.clone();
                for plan in plans {
                    last = self.walk(plan, graph_input, produced, keys)?;
                }
                Ok(last)
            }
            Plan::Wave(branches) => self.at_once(branches, graph_input, produced, keys),
            Plan::Remote { host, inner } => {
                self.elsewhere(host, inner, graph_input, produced, keys)
            }
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
        keys: &mut HashMap<NodeId, Key>,
    ) -> Result<Value, RunError> {
        let earlier: &HashMap<NodeId, Value> = produced;
        let named: &HashMap<NodeId, Key> = keys;
        let outcomes = std::thread::scope(|scope| {
            let running: Vec<_> = branches
                .iter()
                .map(|branch| {
                    scope.spawn(move || {
                        let mut mine = earlier.clone();
                        let mut mine_keys = named.clone();
                        let last = self.walk(branch, graph_input, &mut mine, &mut mine_keys)?;
                        mine.retain(|id, _| !earlier.contains_key(id));
                        mine_keys.retain(|id, _| !named.contains_key(id));
                        Ok::<_, RunError>((last, mine, mine_keys))
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
            let (_, mine, mine_keys) = outcome?;
            produced.extend(mine);
            keys.extend(mine_keys);
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
        keys: &mut HashMap<NodeId, Key>,
    ) -> Result<Value, RunError> {
        let transport = self
            .transports
            .iter()
            .find(|(known, _)| known == host)
            .map(|(_, transport)| *transport)
            .ok_or_else(|| RunError::NoTransport(host.clone()))?;

        let reads = needs(inner);
        let known: Vec<(NodeId, Value)> = reads
            .iter()
            .filter_map(|id| produced.get(id).map(|value| (id.clone(), value.clone())))
            .collect();
        // The keys of the same set: what it reads is what it has to be able to
        // name, and what it produces it names from those.
        let named: Vec<(NodeId, Key)> = reads
            .iter()
            .filter_map(|id| keys.get(id).map(|key| (id.clone(), key.clone())))
            .collect();

        let nowhere = Placement::new();
        let nothing = Memory::new();
        let cargo = Cargo {
            input: graph_input,
            known: &known,
            keys: &named,
            placement: self.placement.unwrap_or(&nowhere),
            // It travels whether or not there is a keeper here: what is
            // remembered is the graph's, and the one that keeps things may be
            // the other side.
            memory: self.memory.unwrap_or(&nothing),
        };
        let outcome = transport
            .dispatch(inner, &cargo)
            .map_err(|source| RunError::Transport {
                host: host.clone(),
                source,
            })?;

        produced.extend(outcome.produced);
        keys.extend(outcome.keys);
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

    /// The name this node's output will have, **before** it has one.
    ///
    /// One of the two seams of the cache, and it has a name of its own because
    /// it is where the observer of gradients will hang: it is the one place
    /// that sees every edge with what crosses it already decided.
    ///
    /// `None` — no name — whenever anything the recipe is made of is missing:
    /// no keeper, nothing said about what implements this node, or a
    /// predecessor that has no key either. It is not a failure. It means this
    /// output cannot be kept and neither can anything below it, and the run
    /// goes on exactly as it did before any of this existed.
    fn key_for(
        &self,
        node: &NodeId,
        from: &[NodeId],
        graph_input: &Value,
        keys: &HashMap<NodeId, Key>,
    ) -> Option<Key> {
        let (keeper, memory) = (self.keeper?, self.memory?);
        let identity = memory.identity_of(node)?;
        // A root reads the graph's input, and the input is the one thing in the
        // whole chain hashed **by its content** — from here down it is hashes
        // of hashes, which is what makes a key knowable before anything runs.
        let above: Vec<Key> = match from {
            [] => vec![keeper.key_of(graph_input)?],
            many => many
                .iter()
                .map(|id| keys.get(id).cloned())
                .collect::<Option<Vec<_>>>()?,
        };

        // The state of a node that is not frozen is empty, and it does not
        // matter: nothing unfrozen is kept, nor is anything under it.
        let mut parts = vec![
            identity,
            memory.state_of(node).unwrap_or(""),
            memory.salt_of(node).unwrap_or(""),
        ];
        parts.extend(above.iter().map(Key::as_str));
        Some(keeper.combine(&parts))
    }

    /// What is kept under this node's name, if it is kept at all.
    ///
    /// A keeper that cannot answer is **not** the end of the run: the value is
    /// recomputed and the trouble is said out loud, the same as a worker whose
    /// store cannot be reached asks the client instead of refusing. A cache is
    /// an optimization, and an optimization that can kill a run at hour three
    /// is not one.
    fn recalled(&self, node: &NodeId, key: Option<&Key>) -> Option<Value> {
        let (keeper, memory) = (self.keeper?, self.memory?);
        let key = key?;
        if !memory.is_cached(node) {
            return None;
        }
        let kept = match keeper.recall(&[key]) {
            Ok(answers) => answers.into_iter().next().flatten()?,
            Err(why) => {
                eprintln!("what `{node}` produced could not be looked up: {why}");
                return None;
            }
        };

        // The fingerprint is not in the key on purpose — a cosmetic refactor
        // would invalidate half the store in silence — so this is where the two
        // are put side by side. It is said, and what was kept is used: whoever
        // wants the other behaviour asks for it.
        if let (Some(declared), Some(written)) = (memory.fingerprint_of(node), fingerprint(&kept))
            && declared != written
        {
            eprintln!(
                "`{node}` was kept by code fingerprinted `{written}` and this graph declares \
                 `{declared}`: using what is kept, since the fingerprint is not part of the key"
            );
        }
        Some(kept.value)
    }

    /// Keeps what this node produced, if it was said to be worth keeping.
    ///
    /// The other seam. A node with no `.cached()` still got a key above and
    /// still passed it on — it just does not land anywhere, which is what makes
    /// declaring the cache node by node cost nothing to the chain.
    fn keep(&self, node: &NodeId, key: Option<&Key>, output: &Value) {
        let (Some(keeper), Some(memory), Some(key)) = (self.keeper, self.memory, key) else {
            return;
        };
        if !memory.is_cached(node) {
            return;
        }
        let mut meta = vec![(NODE, node.as_str())];
        if let Some(written) = memory.fingerprint_of(node) {
            meta.push((FINGERPRINT, written));
        }
        if let Err(why) = keeper.keep(key, output, &meta) {
            eprintln!("what `{node}` produced could not be kept: {why}");
        }
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

/// A table in the order a wire wants it: by id, so two runs of the same thing
/// answer with the same bytes.
fn sorted<T>(table: HashMap<NodeId, T>) -> Vec<(NodeId, T)> {
    let mut out: Vec<(NodeId, T)> = table.into_iter().collect();
    out.sort_by(|(a, _), (b, _)| a.cmp(b));
    out
}

/// What was written beside a kept value about the code that produced it.
fn fingerprint(kept: &Kept) -> Option<&str> {
    kept.meta
        .iter()
        .find(|(what, _)| what == FINGERPRINT)
        .map(|(_, written)| written.as_str())
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
