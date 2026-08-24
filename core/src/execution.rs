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
    Cargo, Catalog, Ctx, Device, Fact, Host, Keeper, Kept, Key, Keys, Memory, NodeError, NodeId,
    Outcome, Placement, Plan, Transport, TransportError, Value, Watcher,
};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Instant;

/// What the engine writes beside a value it keeps. Private because the engine
/// is the only one that writes it and the only one that reads it back: to a
/// [`Keeper`] the metadata is text it hands over untouched.
const NODE: &str = "node";
const FINGERPRINT: &str = "fingerprint";

/// Executes plans.
///
/// A type and not a bare function because executing needs context: today the
/// store, the placement and the transports.
pub struct Executor<'a> {
    catalog: &'a Catalog,
    placement: Option<&'a Placement>,
    /// What is remembered about each node. **Declared**, like the placement: it
    /// belongs to whoever wrote the graph, and it travels.
    memory: Option<&'a Memory>,
    /// Who hashes and where what is named ends up. **Injected**, like the
    /// transports: it belongs to whoever runs, and it does not travel.
    keeper: Option<&'a dyn Keeper>,
    /// Which host it knows how to reach, and by what route. A list because
    /// there are two or three of them.
    transports: Vec<(Host, &'a dyn Transport)>,
    /// Who is told what happened. **Injected**, like the keeper, and for the
    /// same reason: where a fact ends up belongs to whoever runs, not to the
    /// graph, so it does not travel.
    watcher: Option<&'a dyn Watcher>,
    /// Where this walk's timeline starts, so that every fact can say **when**
    /// and not only how long.
    ///
    /// Set once per [`run`](Self::run) or [`resume`](Self::resume) and read all
    /// the way down, which is why it is a field rather than an argument: it is
    /// one number for the whole walk and threading it through six signatures
    /// would say it was six.
    since: Option<Instant>,
}

impl<'a> Executor<'a> {
    /// An executor over this catalog, with nothing else said yet.
    pub fn new(catalog: &'a Catalog) -> Self {
        Self {
            catalog,
            placement: None,
            memory: None,
            keeper: None,
            transports: Vec::new(),
            watcher: None,
            since: None,
        }
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
    /// **Injected**, like the transports: it is whoever runs who
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

    /// The same executor, telling this one what it sees.
    ///
    /// **Injected and does not travel**, like the keeper: a slice sent to
    /// another host is executed by an engine over there with a watcher of its
    /// own, and what it sees comes back attributed rather than emitted here.
    ///
    /// Without it nothing is measured and no [`Fact`] is built — the emit sites
    /// take a closure, so a run nobody watches pays a branch and not an
    /// allocation.
    pub fn watching(mut self, watcher: &'a dyn Watcher) -> Self {
        self.watcher = Some(watcher);
        self
    }

    /// Hands over one fact, if anybody is listening.
    ///
    /// The closure is the whole point: `node.clone()` and a formatted message
    /// are not paid for by a run that nobody is watching, which is most of them.
    fn saw(&self, fact: impl FnOnce() -> Fact) {
        if let Some(watcher) = self.watcher {
            watcher.saw(&fact());
        }
    }

    /// Executes the plan and returns what it produced. The first failure stops
    /// the execution.
    ///
    /// This is where a run **ends**, and the only place that says so: a
    /// [`Fact::Finished`] or a [`Fact::Broke`] closes the record either way.
    /// [`resume`](Self::resume) deliberately says nothing of the sort — a slice
    /// executed for somebody else is not a `forward`.
    pub fn run(&self, plan: &Plan, input: Value) -> Result<Value, RunError> {
        let began = Instant::now();
        let walking = self.since(began);
        let answer = walking.running(plan, input);
        match &answer {
            Ok(_) => walking.saw(|| Fact::Finished {
                took: began.elapsed(),
            }),
            Err(why) => walking.saw(|| Fact::Broke {
                why: why.to_string(),
            }),
        }
        answer
    }

    /// The same executor with a timeline of its own, for one walk.
    ///
    /// A copy and not a mutation, because [`run`](Self::run) takes `&self`: an
    /// executor is shared, and a run is not.
    fn since(&self, began: Instant) -> Self {
        Self {
            catalog: self.catalog,
            placement: self.placement,
            memory: self.memory,
            keeper: self.keeper,
            transports: self.transports.clone(),
            watcher: self.watcher,
            since: Some(began),
        }
    }

    /// How long this walk has been going, or zero if nobody started a clock.
    fn so_far(&self) -> std::time::Duration {
        self.since.map(|began| began.elapsed()).unwrap_or_default()
    }

    /// The walk itself, so that the two terminal facts wrap one thing and not
    /// every `return` in it.
    fn running(&self, plan: &Plan, input: Value) -> Result<Value, RunError> {
        let mut produced: HashMap<NodeId, Value> = HashMap::new();
        // The names first, and then what will not be needed because of them.
        let (mut keys, unneeded) = self.foreseen(plan, &input);
        let last = self.walk(plan, &input, &mut produced, &mut keys, &unneeded)?;

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
        named: Vec<(NodeId, Keys)>,
    ) -> Result<Outcome, RunError> {
        // A slice counts from its own start: what it says about `when` is a
        // fact about the slice, and whoever draws it adds the offset of the
        // `Left` it arrived under.
        let walking = self.since(Instant::now());
        let mut produced: HashMap<NodeId, Value> = known.into_iter().collect();
        let mut keys: HashMap<NodeId, Keys> = named.into_iter().collect();
        let brought: Vec<NodeId> = produced.keys().cloned().collect();
        let named: Vec<NodeId> = keys.keys().cloned().collect();

        // A slice is not pruned: what it produces goes back to whoever sent it,
        // and only that side knows which of it is read. The client already did
        // the working out, and a node it did not want is not in the plan.
        let last = walking.walk(plan, &input, &mut produced, &mut keys, &HashSet::new())?;

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
        keys: &mut HashMap<NodeId, Keys>,
        unneeded: &HashSet<NodeId>,
    ) -> Result<Value, RunError> {
        match plan {
            Plan::Empty => Ok(graph_input.clone()),
            // Nothing reads what this one makes that is not already kept, so
            // there is nothing for it to be for. Its name is already in `keys`
            // — worked out without it — so whoever hit downstream still hits.
            Plan::Execute { node, .. } if unneeded.contains(node) => {
                self.saw(|| Fact::Spared { node: node.clone() });
                Ok(Value::Null)
            }
            Plan::Execute { node, from } if self.maps(node) => {
                self.over_items(node, from, graph_input, produced, keys)
            }
            Plan::Execute { node, from } => {
                // Worked out already, if anybody worked it out: naming the root
                // is the one place a value is hashed by content, and doing it
                // twice would make asking early cost exactly what it saves.
                let key = match keys.get(node) {
                    Some(Keys::One(named)) => Some(named.clone()),
                    _ => self.key_for(node, from, graph_input, keys),
                };
                if let Some(key) = &key {
                    keys.insert(node.clone(), Keys::One(key.clone()));
                }
                // A hit is the whole point: the node is not advanced, and its
                // input is not even assembled.
                let output = match self.recalled(node, key.as_ref()) {
                    Some(kept) => kept,
                    // What was kept when the store was asked, and gone when it
                    // was read. Rare, and it has to be said rather than turned
                    // into a puzzle: what feeds this was skipped **because** the
                    // answer was there, so there is nothing left to run.
                    None if from.iter().any(|id| unneeded.contains(id)) => {
                        return Err(RunError::Vanished { node: node.clone() });
                    }
                    None => {
                        let input = gather(node, from, graph_input, produced)?;
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
                    last = self.walk(plan, graph_input, produced, keys, unneeded)?;
                }
                Ok(last)
            }
            Plan::Wave(branches) => self.at_once(branches, graph_input, produced, keys, unneeded),
            // A slice nobody needs is a message that is not sent: the whole
            // round trip goes, not just the work at the far end.
            Plan::Remote { inner, .. }
                if inner.steps().all(|step| unneeded.contains(step.node)) =>
            {
                for step in inner.steps() {
                    self.saw(|| Fact::Spared {
                        node: step.node.clone(),
                    });
                }
                Ok(Value::Null)
            }
            Plan::Remote { host, inner } => {
                self.elsewhere(host, inner, graph_input, produced, keys)
            }
        }
    }

    /// Whether this node was declared to map over the items of its input.
    fn maps(&self, node: &NodeId) -> bool {
        self.memory.is_some_and(|memory| memory.is_mapped(node))
    }

    /// One step of a node that maps: the items it is missing, and no more.
    ///
    /// The whole difference from an ordinary step is **where the grain is**. An
    /// ordinary node is named once and either its whole output is there or none
    /// of it is; a mapped one is named once per item, so a list of a thousand
    /// where one is new runs once and reads back nine hundred and ninety-nine.
    ///
    /// The input is assembled first here, and there is no way around it: an
    /// ordinary hit does not even build its input, but the names of these items
    /// are made out of the items, so they have to be in hand.
    fn over_items(
        &self,
        node: &NodeId,
        from: &[NodeId],
        graph_input: &Value,
        produced: &mut HashMap<NodeId, Value>,
        keys: &mut HashMap<NodeId, Keys>,
    ) -> Result<Value, RunError> {
        let input = gather(node, from, graph_input, produced)?;
        let Value::List(items) = &input else {
            return Err(RunError::NotItems {
                node: node.clone(),
                given: input.type_name().to_string(),
            });
        };

        let mine = self.keys_for_items(node, from, items, keys);
        let kept: Vec<Option<Value>> = match &mine {
            Some(mine) => self.recalled_items(node, mine),
            None => vec![None; items.len()],
        };
        let missing: Vec<usize> = (0..items.len()).filter(|i| kept[*i].is_none()).collect();
        self.saw(|| Fact::Items {
            node: node.clone(),
            of: items.len(),
            recalled: items.len() - missing.len(),
        });

        // Nothing missing is the point of all this: the node is not advanced at
        // all, exactly as an ordinary hit does not advance it.
        let mut answers = Vec::new();
        if !missing.is_empty() {
            let asked = Value::list(
                missing
                    .iter()
                    .map(|i| items[*i].clone())
                    .collect::<Vec<_>>(),
            );
            let output = self.advance(node, asked)?;
            let Value::List(back) = &output else {
                return Err(RunError::NotItems {
                    node: node.clone(),
                    given: output.type_name().to_string(),
                });
            };
            if back.len() != missing.len() {
                return Err(RunError::Uncounted {
                    node: node.clone(),
                    asked: missing.len(),
                    answered: back.len(),
                });
            }
            answers = back.to_vec();
        }

        let mut out = Vec::with_capacity(items.len());
        let mut answered = answers.into_iter();
        for (i, was) in kept.into_iter().enumerate() {
            match was {
                Some(value) => out.push(value),
                None => {
                    let value = answered.next().expect("one answer per item asked for");
                    if let Some(mine) = &mine {
                        self.keep(node, Some(&mine[i]), &value);
                    }
                    out.push(value);
                }
            }
        }

        let output = Value::list(out);
        if let Some(mine) = mine {
            keys.insert(node.clone(), Keys::PerItem(mine));
        }
        produced.insert(node.clone(), output.clone());
        Ok(output)
    }

    /// One name per item, or `None` when nothing is being remembered at all.
    ///
    /// **Where the per-item chain starts is where content is hashed.** If what
    /// is above already has a name for each item, these are built out of those
    /// and nothing is hashed; if it does not — a root, or a list that came out
    /// of a node nobody mapped — then each item is hashed by its content, which
    /// is the only thing that makes the same document in another list the same
    /// item. Its **position** would not: that is the whole point.
    fn keys_for_items(
        &self,
        node: &NodeId,
        from: &[NodeId],
        items: &[Value],
        keys: &HashMap<NodeId, Keys>,
    ) -> Option<Vec<Key>> {
        let (keeper, memory) = (self.keeper?, self.memory?);
        let identity = memory.identity_of(node)?;
        let above: Vec<Key> = match from {
            [one] => match keys.get(one) {
                Some(Keys::PerItem(each)) if each.len() == items.len() => each.clone(),
                _ => items
                    .iter()
                    .map(|item| keeper.key_of(item))
                    .collect::<Option<Vec<_>>>()?,
            },
            _ => items
                .iter()
                .map(|item| keeper.key_of(item))
                .collect::<Option<Vec<_>>>()?,
        };
        Some(
            above
                .iter()
                .map(|one| {
                    keeper.combine(&[
                        identity,
                        memory.state_of(node).unwrap_or(""),
                        memory.salt_of(node).unwrap_or(""),
                        one.as_str(),
                    ])
                })
                .collect(),
        )
    }

    /// What is kept for each of these, asked **in one call**.
    ///
    /// Which is why [`Keeper::recall`] has taken a slice since the first day: a
    /// thousand items against a store on the far end of a network is a thousand
    /// round trips unless it is one.
    fn recalled_items(&self, node: &NodeId, mine: &[Key]) -> Vec<Option<Value>> {
        let nothing = vec![None; mine.len()];
        let Some((keeper, memory)) = self.keeper.zip(self.memory) else {
            return nothing;
        };
        if !memory.is_cached(node) {
            return nothing;
        }
        match keeper.recall(&mine.iter().collect::<Vec<_>>()) {
            Ok(answers) => answers
                .into_iter()
                .map(|kept| kept.map(|kept| kept.value))
                .collect(),
            Err(why) => {
                eprintln!("what `{node}` produced could not be looked up: {why}");
                nothing
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
        keys: &mut HashMap<NodeId, Keys>,
        unneeded: &HashSet<NodeId>,
    ) -> Result<Value, RunError> {
        let earlier: &HashMap<NodeId, Value> = produced;
        let named: &HashMap<NodeId, Keys> = keys;
        let outcomes = std::thread::scope(|scope| {
            let running: Vec<_> = branches
                .iter()
                .map(|branch| {
                    scope.spawn(move || {
                        let mut mine = earlier.clone();
                        let mut mine_keys = named.clone();
                        let last =
                            self.walk(branch, graph_input, &mut mine, &mut mine_keys, unneeded)?;
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
        keys: &mut HashMap<NodeId, Keys>,
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
        let named: Vec<(NodeId, Keys)> = reads
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
        // What the far side sees is emitted there and relayed raw; attributing
        // it to a host happens **here**, because here is where the host has a
        // name. So a worker's engine emits exactly what it would emit at home,
        // and nothing that travelled has to be rewritten.
        let attributed = self.watcher.map(|to| Attributed {
            host: host.clone(),
            to,
        });
        let at = self.so_far();
        let began = Instant::now();
        let outcome = transport
            .dispatch(
                inner,
                &cargo,
                attributed.as_ref().map(|one| one as &dyn Watcher),
            )
            .map_err(|source| RunError::Transport {
                host: host.clone(),
                source,
            })?;
        self.saw(|| Fact::Left {
            host: host.clone(),
            began: at,
            took: began.elapsed(),
        });

        produced.extend(outcome.produced);
        keys.extend(outcome.keys);
        Ok(outcome.last)
    }

    /// Run it, and attribute whatever it says to it.
    ///
    /// Whatever the node takes to answer — a retry, a model, three rounds of
    /// something — happens inside it, and the engine neither counts it nor
    /// bounds it. It cannot: it has no way to tell a loop that will not end from
    /// work that is slow, and guessing wrong either way is worse than not
    /// guessing.
    fn advance(&self, node: &NodeId, input: Value) -> Result<Value, RunError> {
        let ctx = Ctx {
            device: self.device(node),
        };
        // Around the `forward` and nothing else: what is measured is the node,
        // not the bookkeeping around it, so the number means the same thing
        // whether or not this node is cached, mapped or on another machine.
        let at = self.so_far();
        let began = Instant::now();
        let answer = self.implementation(node)?.forward(&input, &ctx);
        let took = began.elapsed();
        match answer {
            Ok(output) => {
                self.saw(|| Fact::Ran {
                    node: node.clone(),
                    began: at,
                    took,
                    device: self.device(node).cloned(),
                });
                Ok(output)
            }
            Err(source) => {
                // Said before the error is returned, which is the whole reason
                // this is a fact and not a line in `RunError`: by the time the
                // caller sees the failure the run is over, and whoever is
                // watching wanted to know **which node** while it was happening.
                self.saw(|| Fact::Failed {
                    node: node.clone(),
                    why: source.to_string(),
                });
                Err(RunError::Node {
                    node: node.clone(),
                    source,
                })
            }
        }
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
        keys: &HashMap<NodeId, Keys>,
    ) -> Option<Key> {
        let (keeper, memory) = (self.keeper?, self.memory?);
        let identity = memory.identity_of(node)?;
        let keeper: &dyn Keeper = keeper;
        // A root reads the graph's input, and the input is the one thing in the
        // whole chain hashed **by its content** — from here down it is hashes
        // of hashes, which is what makes a key knowable before anything runs.
        let above: Vec<Key> = match from {
            [] => vec![keeper.key_of(graph_input)?],
            many => many
                .iter()
                .map(|id| keys.get(id).map(|keys| whole(keeper, keys)))
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

    /// The names the whole plan will produce, and the nodes that will not have
    /// to produce them.
    ///
    /// # Why this can be worked out at all
    ///
    /// Because [`key_for`](Self::key_for) says so: *the name this node's output
    /// will have, **before** it has one*. Only the graph's input is hashed by
    /// content; from there down a key is made of keys, so every name in the plan
    /// is knowable with nothing executed. One question to the keeper —
    /// [`present`](Keeper::present), which a store answers by name and without
    /// reading anything — says which of those answers are already there.
    ///
    /// # And then backwards
    ///
    /// From the leaves up: a node is needed if something that will really run
    /// reads it. A node whose answer is already kept **does not need its
    /// inputs** — which is the whole point, and what makes a settled encoder
    /// under a head that hit cost nothing rather than cost a forward whose
    /// result is dropped a microsecond later.
    ///
    /// Two places it deliberately gives up, and both are conservative — they
    /// keep a node rather than skip one:
    ///
    /// - a **mapped** node is named by the content of its items, so its names
    ///   are not knowable before it has its input. It counts as a miss, and
    ///   everything above it stays.
    /// - a node with **no key at all** — nothing said about what implements it,
    ///   or something unnameable upstream — is a miss for the same reason it
    ///   was never cached.
    fn foreseen(
        &self,
        plan: &Plan,
        graph_input: &Value,
    ) -> (HashMap<NodeId, Keys>, HashSet<NodeId>) {
        let nothing = (HashMap::new(), HashSet::new());
        let (Some(keeper), Some(memory)) = (self.keeper, self.memory) else {
            return nothing;
        };

        // Plan order is topological, so a predecessor's name is always in hand.
        let mut named: HashMap<NodeId, Keys> = HashMap::new();
        let mut asked: Vec<(NodeId, Key)> = Vec::new();
        for step in plan.steps() {
            if self.maps(step.node) {
                continue;
            }
            let Some(key) = self.key_for(step.node, step.from, graph_input, &named) else {
                continue;
            };
            if memory.is_cached(step.node) {
                asked.push((step.node.clone(), key.clone()));
            }
            named.insert(step.node.clone(), Keys::One(key));
        }
        if asked.is_empty() {
            return (named, HashSet::new());
        }

        let keys: Vec<&Key> = asked.iter().map(|(_, key)| key).collect();
        let there = match keeper.present(&keys) {
            Ok(there) => there,
            // A keeper that cannot answer is not the end of the run, here as
            // everywhere else: nothing is skipped and everything is computed,
            // which is what happened before any of this existed.
            Err(why) => {
                eprintln!("what is already kept could not be looked up: {why}");
                return (named, HashSet::new());
            }
        };
        let kept: HashSet<&NodeId> = asked
            .iter()
            .zip(&there)
            .filter(|(_, there)| **there)
            .map(|((node, _), _)| node)
            .collect();

        let mut needed: HashSet<NodeId> = HashSet::new();
        let mut asking: Vec<NodeId> = terminals(plan);
        while let Some(node) = asking.pop() {
            if !needed.insert(node.clone()) || kept.contains(&node) {
                continue;
            }
            for step in plan.steps().filter(|step| *step.node == node) {
                asking.extend(step.from.iter().cloned());
            }
        }
        let unneeded = plan
            .steps()
            .map(|step| step.node)
            .filter(|node| !needed.contains(*node))
            .cloned()
            .collect();
        (named, unneeded)
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
        self.saw(|| Fact::Recalled {
            node: node.clone(),
            key: key.clone(),
        });
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
        match keeper.keep(key, output, &meta) {
            Ok(()) => self.saw(|| Fact::Kept {
                node: node.clone(),
                key: key.clone(),
            }),
            Err(why) => eprintln!("what `{node}` produced could not be kept: {why}"),
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

/// A watcher that says where what it is told happened, and passes it on.
///
/// Private, and it belongs to the engine rather than to the transport: a
/// `Transport` relays what came off the wire untouched — it has no business
/// knowing what a fact means — while the name of the host it reached is
/// something only the walk has, because that name is the graph's.
///
/// One wrapper per dispatch, so a slice that carries on to a third machine
/// arrives nested and comes out of [`Fact::flattened`] with the route in order.
struct Attributed<'a> {
    host: Host,
    to: &'a dyn Watcher,
}

impl Watcher for Attributed<'_> {
    fn saw(&self, fact: &Fact) {
        self.to.saw(&Fact::Elsewhere {
            host: self.host.clone(),
            saw: Box::new(fact.clone()),
        });
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
    /// The plan sends a slice to a host nobody knows how to reach.
    NoTransport(Host),
    /// The transport could not carry the slice, or what ran there failed.
    Transport {
        /// Which host it was bound for.
        host: Host,
        /// What the transport said.
        source: TransportError,
    },
    /// A node that maps was handed something that is not a list of items, or
    /// answered with something that is not one.
    NotItems {
        /// Which node.
        node: NodeId,
        /// And what arrived instead.
        given: String,
    },
    /// A node that maps answered with a different number of items than it was
    /// asked for, so nobody knows which answer goes with which item.
    Uncounted {
        /// Which node.
        node: NodeId,
        /// How many items it was handed.
        asked: usize,
        /// And how many came back.
        answered: usize,
    },
    /// What was kept when the store was asked and gone when it was read, after
    /// what feeds this node had already been skipped because of the answer.
    Vanished {
        /// The one with nothing left to run.
        node: NodeId,
    },
    /// A step reads what another produced, and what it produced never came back
    /// from wherever it ran.
    Lost {
        /// The one that cannot be assembled.
        node: NodeId,
        /// What it was reading.
        from: NodeId,
    },
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoImplementation(id) => {
                write!(f, "node `{id}` has no registered implementation")
            }
            Self::Node { node, source } => write!(f, "node `{node}` failed: {source}"),
            Self::NotItems { node, given } => write!(
                f,
                "`{node}` maps over the items of its input, so what reaches it and \
                 what it answers with are lists; a `{given}` is one thing and has \
                 no items. Either it does not map, or whoever feeds it should be \
                 handing it a list"
            ),
            Self::Uncounted {
                node,
                asked,
                answered,
            } => write!(
                f,
                "`{node}` was handed {asked} items and answered with {answered}: a \
                 node that maps gives back one for each, in order, or nobody can \
                 tell which answer belongs to which item"
            ),
            Self::NoTransport(host) => write!(
                f,
                "there is a slice placed on `{host}` and this executor cannot reach it"
            ),
            Self::Transport { host, source } => write!(f, "carrying a slice to `{host}`: {source}"),
            Self::Vanished { node } => write!(
                f,
                "what was kept for `{node}` was there when the store was asked and gone when \
                 it was read, and what feeds it was not run because of that answer. Nothing \
                 was lost — run it again"
            ),
            Self::Lost { node, from } => write!(
                f,
                "`{node}` reads what `{from}` produced, and that stayed where it ran: \
                 only what can leave a process comes back from one"
            ),
        }
    }
}

impl std::error::Error for RunError {}

/// What a node receives: nothing → the graph's input, one thing → that thing,
/// several → a map keyed by whoever produced each, in edge declaration order.
fn gather(
    node: &NodeId,
    from: &[NodeId],
    graph_input: &Value,
    produced: &HashMap<NodeId, Value>,
) -> Result<Value, RunError> {
    // Topological order already ran the predecessors, so what is missing here is
    // not a bug in the walk: it is a value that stayed in the process that made
    // it, because it could not leave one.
    let recall = |id: &NodeId| {
        produced.get(id).cloned().ok_or_else(|| RunError::Lost {
            node: node.clone(),
            from: id.clone(),
        })
    };
    match from {
        [] => Ok(graph_input.clone()),
        [single] => recall(single),
        many => Ok(Value::map(
            many.iter()
                .map(|id| Ok((id.to_string(), recall(id)?)))
                .collect::<Result<Vec<_>, RunError>>()?,
        )),
    }
}

/// One name for what a node produced, whether it has one or a thousand.
///
/// A list of item keys becomes a single key by being combined, and it is
/// [`Keeper::combine`] that decides how rather than anything here: the core does
/// not know what a hash is. What matters is that it is **deterministic and
/// depends on all of them**, so that whatever reads the whole list is named
/// after the whole list.
fn whole(keeper: &dyn Keeper, keys: &Keys) -> Key {
    match keys {
        Keys::One(key) => key.clone(),
        Keys::PerItem(each) => keeper.combine(&each.iter().map(Key::as_str).collect::<Vec<_>>()),
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
    let produced: Vec<&NodeId> = plan.steps().map(|step| step.node).collect();
    let mut out: Vec<NodeId> = Vec::new();
    for id in plan.steps().flat_map(|step| step.from) {
        if !produced.contains(&id) && !out.contains(id) {
            out.push(id.clone());
        }
    }
    out
}

/// The plan's nodes whose output no other node reads: the leaves.
fn terminals(plan: &Plan) -> Vec<NodeId> {
    let consumed: Vec<&NodeId> = plan.steps().flat_map(|step| step.from).collect();
    plan.steps()
        .map(|step| step.node)
        .filter(|node| !consumed.contains(node))
        .cloned()
        .collect()
}
