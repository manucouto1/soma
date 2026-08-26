//! The engine: walking a [`Plan`] and executing what it says.
//!
//! Walking is domain logic, so it lives here and not in the bindings; Python
//! only supplies the implementations. The engine never looks at the graph —
//! every plan step carries where its input comes from.
//!
//! When one thing reaches a node it receives that thing; when several do, a
//! [`Value::Map`] keyed by whoever produced each. Aggregating them is the
//! receiving node's job.
//!
//! Keys travel in a table **beside** `produced`, never inside a [`Value`], so
//! the [`Node`](crate::Node) contract does not change. Nothing is named without
//! both [`Executor::remembering`] (declared, travels) and [`Executor::keeping`]
//! (injected, does not).

use crate::{
    Cargo, Catalog, Ctx, Device, Fact, Host, Keeper, Kept, Key, Keys, Memory, NodeError, NodeId,
    Outcome, Placement, Plan, Transport, TransportError, Value, Watcher,
};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Instant;

/// What the engine writes beside a value it keeps. Public because a store
/// outlives the process that wrote to it and readers need these strings.
pub const NODE: &str = "node";

/// Which version of the code produced it, written beside the value rather than
/// mixed into the name.
pub const FINGERPRINT: &str = "fingerprint";

/// What the graph was fed, by the name its content has. Only a
/// [`Keeper`](crate::Keeper) can hash a [`Value`], and a key does not run
/// backwards, so it is written now or never. Set on [`run`](Executor::run) and
/// never on [`resume`](Executor::resume): a slice is not handed the graph's
/// input.
pub const INPUT: &str = "input";

/// The words the engine writes itself, so a layer that refuses a stamp can ask
/// which ones are taken.
pub const OURS: [&str; 3] = [NODE, FINGERPRINT, INPUT];

/// Executes plans. A type and not a bare function because executing needs
/// context: the store, the placement and the transports.
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
    /// Who is told what happened. Injected, so it does not travel.
    watcher: Option<&'a dyn Watcher>,
    /// Where this walk's timeline starts, so a fact can say **when** and not
    /// only how long. One number for the whole walk, hence a field.
    since: Option<Instant>,
    /// What else to write beside everything this run keeps. **Injected**, like
    /// the keeper, and opaque: see [`stamping`](Self::stamping).
    stamp: Vec<(String, String)>,
    /// The name of what the graph was fed, worked out once per
    /// [`run`](Self::run) rather than at every kept node.
    input: Option<Key>,
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
            stamp: Vec::new(),
            input: None,
        }
    }

    /// The same executor, knowing where each node runs. Without this every
    /// `ctx.device` is `None`, which means "wherever it lands".
    pub fn placed(mut self, placement: &'a Placement) -> Self {
        self.placement = Some(placement);
        self
    }

    /// The same executor, knowing what is remembered about each node. Declared,
    /// so it **travels** with a slice — its own call and not half of
    /// [`keeping`](Self::keeping), which does not.
    pub fn remembering(mut self, memory: &'a Memory) -> Self {
        self.memory = Some(memory);
        self
    }

    /// The same executor, with somewhere to keep what it names. Injected.
    /// Without it, or without [`remembering`](Self::remembering), no key is
    /// even computed.
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

    /// The same executor, telling this one what it sees. Injected, so a slice
    /// sent away is watched over there and comes back attributed.
    pub fn watching(mut self, watcher: &'a dyn Watcher) -> Self {
        self.watcher = Some(watcher);
        self
    }

    /// The same executor, writing this beside everything it keeps.
    ///
    /// Opaque text the core passes through untouched: an environment, a commit,
    /// a run are facts about the world outside a graph. Injected, so a slice
    /// sent away is stamped by the engine over there.
    pub fn stamping(mut self, stamp: impl IntoIterator<Item = (String, String)>) -> Self {
        self.stamp = stamp.into_iter().collect();
        self
    }

    /// Hands over one fact, if anybody is listening. The closure is why a run
    /// nobody watches pays a branch and not an allocation.
    fn saw(&self, fact: impl FnOnce() -> Fact) {
        if let Some(watcher) = self.watcher {
            watcher.saw(&fact());
        }
    }

    /// Executes the plan and returns what it produced; the first failure stops
    /// it. The only place a run is said to end — [`resume`](Self::resume) says
    /// nothing of the sort, because a slice is not a `forward`.
    pub fn run(&self, plan: &Plan, input: Value) -> Result<Value, RunError> {
        let began = Instant::now();
        // `None` when there is no keeper, or when the input cannot leave this
        // process — the same absence that leaves everything under it nameless.
        let named = self.keeper.and_then(|keeper| keeper.key_of(&input));
        let walking = self.since(began).fed(named);
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

    /// The same executor with a timeline of its own, for one walk. A copy
    /// because [`run`](Self::run) takes `&self`: an executor is shared.
    fn since(&self, began: Instant) -> Self {
        Self {
            catalog: self.catalog,
            placement: self.placement,
            memory: self.memory,
            keeper: self.keeper,
            transports: self.transports.clone(),
            watcher: self.watcher,
            since: Some(began),
            stamp: self.stamp.clone(),
            // Not carried over: `run` works it out and sets it, and `resume`
            // deliberately leaves it empty. A slice's input is not a graph's.
            input: None,
        }
    }

    /// The same walk, knowing what the graph was fed.
    fn fed(mut self, input: Option<Key>) -> Self {
        self.input = input;
        self
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
        // several a map keyed by each, so a diamond comes back round.
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
    /// on receiving one. `known` and `named` are fed in as if this run had
    /// produced them; neither comes back, and both are ordered by id because
    /// this crosses a process boundary.
    pub fn resume(
        &self,
        plan: &Plan,
        input: Value,
        known: Vec<(NodeId, Value)>,
        named: Vec<(NodeId, Keys)>,
    ) -> Result<Outcome, RunError> {
        // A slice counts from its own start: an offset into a slice is a fact
        // about the slice, and two wall clocks would not have composed.
        let walking = self.since(Instant::now());
        let mut produced: HashMap<NodeId, Value> = known.into_iter().collect();
        let mut keys: HashMap<NodeId, Keys> = named.into_iter().collect();
        let brought: Vec<NodeId> = produced.keys().cloned().collect();
        let named: Vec<NodeId> = keys.keys().cloned().collect();

        // A slice is not pruned: only the sender knows which of what it
        // produces is read, and it already left out what it did not want.
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
            // Nothing reads what this makes that is not already kept. Its name
            // is in `keys` regardless, so whoever hit downstream still hits.
            Plan::Execute { node, .. } if unneeded.contains(node) => {
                self.saw(|| Fact::Spared { node: node.clone() });
                Ok(Value::Null)
            }
            Plan::Execute { node, from } if self.maps(node) => {
                self.over_items(node, from, graph_input, produced, keys)
            }
            Plan::Execute { node, from } => {
                // Naming the root is the one place a value is hashed by
                // content; twice would cost exactly what asking early saves.
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
                    // Kept when the store was asked, gone when it was read.
                    // What feeds this was skipped *because* the answer was
                    // there, so there is nothing left to run.
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

    /// One step of a node that maps: the items it is missing, and no more. Its
    /// input is assembled first and there is no way around it — the names of
    /// these items are made out of the items.
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

    /// One name per item, or `None` when nothing is being remembered.
    ///
    /// If what is above already names each item these are built out of those;
    /// if not, each item is hashed by **its own content** — its position would
    /// not make the same document in another list the same item.
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

    /// What is kept for each of these, asked in one call: a thousand items
    /// against a remote store is a thousand round trips unless it is one.
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

    /// Launches a wave's branches at once and merges what they produced. Each
    /// gets a copy of `produced` and returns only its own; being connected
    /// components they are disjoint, so merging clobbers nothing and there is
    /// no lock.
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

    /// Sends a slice elsewhere and merges whatever comes back, given only what
    /// it reads and does not produce.
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
            // Travels whether or not there is a keeper here: what is
            // remembered is the graph's, and the far side may be the keeper.
            memory: self.memory.unwrap_or(&nothing),
        };
        // The far side emits exactly what it would emit at home; attributing
        // it happens here, because here is where the host has a name.
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

    /// Run it, and attribute whatever it says to it. Whatever the node takes to
    /// answer happens inside it: the engine cannot tell a loop that will not end
    /// from work that is slow, so it neither counts nor bounds.
    fn advance(&self, node: &NodeId, input: Value) -> Result<Value, RunError> {
        let ctx = Ctx {
            device: self.device(node),
        };
        // Around the `forward` and nothing else, so the number means the same
        // whether or not the node is cached, mapped or on another machine.
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
                // Said before the error is returned: by the time the caller
                // sees it the run is over, and a watcher wanted the node now.
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
    /// `None` whenever anything the recipe is made of is missing. Not a
    /// failure: it means neither this output nor anything below it can be kept.
    /// One of the two seams of the cache — the one that sees every edge.
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
        // A root reads the graph's input, the one thing hashed by content; from
        // here down it is hashes of hashes, which is what makes a key foreseen.
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
            memory.declaration_of(node).unwrap_or(""),
            memory.state_of(node).unwrap_or(""),
            memory.salt_of(node).unwrap_or(""),
        ];
        parts.extend(above.iter().map(Key::as_str));
        Some(keeper.combine(&parts))
    }

    /// The names the whole plan will produce, and the nodes that will not have
    /// to produce them.
    ///
    /// Names first, since a key needs nothing to have run;
    /// then [`present`](Keeper::present) says which answers are already there;
    /// then backwards from the leaves, because a node whose answer is kept does
    /// not need its inputs. Gives up towards **keeping** a node in two places: a
    /// mapped node, named by its items' content, and a node with no key.
    ///
    /// Public because the answer is worth having without the run: two versions
    /// of a graph name a node differently exactly when its recipe changed.
    /// Names nothing without a keeper and a memory.
    pub fn foreseen(
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
            // A keeper that cannot answer is not the end of the run: nothing
            // is skipped and everything is computed.
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

    /// What is kept under this node's name, if it is kept at all. A keeper that
    /// cannot answer recomputes and says so — an optimization that can kill a
    /// run at hour three is not one.
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

        // The fingerprint is deliberately not in the key, so this is where the
        // two are put side by side. It is said, and what was kept is used.
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

    /// Keeps what this node produced, if it was said to be worth keeping. A
    /// node with no `.cached()` still got a key and still passed it on.
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
        if let Some(fed) = &self.input {
            meta.push((INPUT, fed.as_str()));
        }
        // What the engine knows is not shadowed by a caller who picked one of
        // its words. Dropped and not put last: the obvious way to read a list
        // of pairs takes the last. Silent here because the layer a person types
        // in already refused it out loud.
        meta.extend(
            self.stamp
                .iter()
                .filter(|(what, _)| !OURS.contains(&what.as_str()))
                .map(|(what, said)| (what.as_str(), said.as_str())),
        );
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

/// A watcher that says where what it is told happened, and passes it on. One
/// wrapper per dispatch, so a slice that carried on to a third machine comes out
/// of [`Fact::flattened`] with its route in order.
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

/// Why the execution could not be finished. The structural things were ruled
/// out in [`compile`](crate::compile); these are the implementations' failures.
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
    // Topological order already ran the predecessors, so what is missing stayed
    // in the process that made it, unable to leave one.
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
/// [`Keeper::combine`] decides how; what matters is that it is deterministic and
/// depends on all of them.
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
