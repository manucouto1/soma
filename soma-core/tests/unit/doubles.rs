//! Fake nodes shared by the other modules. None declares what kind it is:
//! what tells them apart is what they answer.

use somatize_core::{
    Cargo, Catalog, Ctx, Device, Fact, Keeper, KeeperError, Kept, Key, Keys, Memory, Node,
    NodeError, NodeId, Outcome, Placement, Plan, Transport, TransportError, Value, Watcher,
};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

/// Adds a constant to a number. Always finishes.
pub struct Add(pub f64);

impl Node for Add {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        match input {
            Value::Number(x) => Ok(Value::number(x + self.0)),
            other => Err(NodeError::new(format!(
                "Add needs a number, it was given {}",
                other.type_name()
            ))),
        }
    }
}

/// Always fails.
/// Produces something that only exists in the process it ran in.
pub struct Opaquely;

impl Node for Opaquely {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Ok(Value::opaque(7u32))
    }
}

/// Takes whatever it is handed and answers a number: the step that reads one of
/// those where it was made.
pub struct Anything;

impl Node for Anything {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Ok(Value::number(1.0))
    }
}

pub struct Fail;

impl Node for Fail {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Err(NodeError::new("I broke"))
    }
}

/// The mean of whatever arrives along its edges. An aggregator is exactly this:
/// a node that reads a map. There is no new type behind it.
pub struct Mean;

impl Node for Mean {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        let Some(values) = input.values() else {
            return Err(NodeError::new(format!(
                "Mean needs several inputs, it got {}",
                input.type_name()
            )));
        };
        let numbers: Vec<f64> = values
            .iter()
            .map(|v| match v {
                Value::Number(x) => Ok(*x),
                other => Err(NodeError::new(format!(
                    "Mean only averages numbers, one was {}",
                    other.type_name()
                ))),
            })
            .collect::<Result<_, _>>()?;
        Ok(Value::number(
            numbers.iter().sum::<f64>() / numbers.len() as f64,
        ))
    }
}

/// Returns its input without asking for anything.
pub struct Immediate;

impl Node for Immediate {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Ok(input.clone())
    }
}

/// Where the execution went: who, in what order, and on what thread.
#[derive(Default)]
pub struct Journal(Mutex<Vec<(String, ThreadId)>>);

impl Journal {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub(crate) fn note(&self, who: &str) {
        self.0
            .lock()
            .expect("nobody poisons this mutex")
            .push((who.to_string(), std::thread::current().id()));
    }

    /// The nodes in the order they executed.
    pub fn order(&self) -> Vec<String> {
        self.0
            .lock()
            .expect("nobody poisons this mutex")
            .iter()
            .map(|(who, _)| who.clone())
            .collect()
    }

    /// Which thread a node ran on.
    pub fn thread_of(&self, who: &str) -> ThreadId {
        self.0
            .lock()
            .expect("nobody poisons this mutex")
            .iter()
            .find(|(name, _)| name == who)
            .map(|(_, thread)| *thread)
            .unwrap_or_else(|| panic!("`{who}` never got to execute"))
    }
}

/// Notes itself in the journal and returns its input untouched.
pub struct Witness(pub &'static str, pub Arc<Journal>);

impl Node for Witness {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        self.1.note(self.0);
        Ok(input.clone())
    }
}

/// A place where several branches agree to meet.
#[derive(Default)]
pub struct MeetingPoint {
    arrived: Mutex<usize>,
    notice: Condvar,
}

impl MeetingPoint {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

/// Does not finish until `how_many` have arrived — how two branches are shown
/// to run at the same time without sleeping a millisecond. The deadline turns a
/// hang into a named error.
pub struct Rendezvous {
    pub point: Arc<MeetingPoint>,
    pub how_many: usize,
    /// With a message, fails with it **after** arriving at the meeting. Used to
    /// make two branches genuinely fail at the same time.
    pub fails: Option<&'static str>,
}

impl Node for Rendezvous {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        let deadline = Duration::from_secs(5);
        let mut arrived = self.point.arrived.lock().expect("nobody poisons it");
        *arrived += 1;
        if *arrived >= self.how_many {
            self.point.notice.notify_all();
        } else {
            let (guard, waited) = self
                .point
                .notice
                .wait_timeout_while(arrived, deadline, |n| *n < self.how_many)
                .expect("nobody poisons it");
            arrived = guard;
            if waited.timed_out() {
                return Err(NodeError::new(format!(
                    "only {} of {} arrived at the meeting: the branches did not run at the same time",
                    *arrived, self.how_many
                )));
            }
        }
        drop(arrived);

        match self.fails {
            Some(message) => Err(NodeError::new(message)),
            None => Ok(input.clone()),
        }
    }
}

/// Blows up. To check that a panic in one branch is not swallowed.
pub struct Panics;

impl Node for Panics {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        panic!("I blew up")
    }
}

/// Where each node was told to run.
#[derive(Default)]
pub struct Ledger(Mutex<Vec<(String, Option<Device>)>>);

impl Ledger {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The device that reached it in the `Ctx`, or `None` if none did. Panics
    /// if the node never got to execute, which is a different thing.
    pub fn of(&self, who: &str) -> Option<Device> {
        self.0
            .lock()
            .expect("nobody poisons this mutex")
            .iter()
            .find(|(name, _)| name == who)
            .map(|(_, where_)| where_.clone())
            .unwrap_or_else(|| panic!("`{who}` never got to execute"))
    }
}

/// Notes where it was told to run and returns its input untouched: everything a
/// core node can do with a device is see it.
pub struct Ubiquitous(pub &'static str, pub Arc<Ledger>);

impl Node for Ubiquitous {
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        self.1
            .0
            .lock()
            .expect("nobody poisons this mutex")
            .push((self.0.to_string(), ctx.device.cloned()));
        Ok(input.clone())
    }
}

/// Executes the slice **right here**, noting what it was sent. What this
/// exercises is the core's seam — what gets sent, what comes back, where it is
/// merged — so a double that never leaves its seat will do.
pub struct Mirror {
    catalog: Arc<Catalog>,
    /// What reached it on each trip, in order.
    pub trips: Mutex<Vec<Trip>>,
}

/// What was sent to a transport in one go.
#[derive(Debug, Clone, PartialEq)]
pub struct Trip {
    pub plan: Plan,
    pub input: Value,
    pub known: Vec<(NodeId, Value)>,
    pub keys: Vec<(NodeId, Keys)>,
    pub placement: Placement,
    pub memory: Memory,
}

impl Mirror {
    pub fn new(catalog: Catalog) -> Self {
        Self {
            catalog: Arc::new(catalog),
            trips: Mutex::new(Vec::new()),
        }
    }

    /// The trips that were made, so they can be inspected.
    pub fn trips(&self) -> Vec<Trip> {
        self.trips.lock().expect("nobody poisons it").clone()
    }
}

impl Transport for Mirror {
    fn dispatch(
        &self,
        plan: &Plan,
        cargo: &Cargo<'_>,
        seen: Option<&dyn Watcher>,
    ) -> Result<Outcome, TransportError> {
        self.trips.lock().expect("nobody poisons it").push(Trip {
            plan: plan.clone(),
            input: cargo.input.clone(),
            known: cargo.known.to_vec(),
            keys: cargo.keys.to_vec(),
            placement: cargo.placement.clone(),
            memory: cargo.memory.clone(),
        });

        // What a real worker does: the engine over there is told and what it
        // says is handed straight back, or the far half of the live view would
        // be untestable without a process.
        let mut over_there = somatize_core::Executor::new(&self.catalog).placed(cargo.placement);
        if let Some(seen) = seen {
            over_there = over_there.watching(seen);
        }
        over_there
            .resume(
                plan,
                cargo.input.clone(),
                cargo.known.to_vec(),
                cargo.keys.to_vec(),
            )
            // A real one cannot carry back what only exists over there, and
            // a double that answers more than a wire can hides the case.
            .map(Outcome::travelling)
            .map_err(|e| TransportError::new(e.to_string()))
    }
}

/// Reaches nowhere, and says so.
pub struct Cable(pub &'static str);

impl Transport for Cable {
    fn dispatch(
        &self,
        _plan: &Plan,
        _cargo: &Cargo<'_>,
        _seen: Option<&dyn Watcher>,
    ) -> Result<Outcome, TransportError> {
        Err(TransportError::new(self.0))
    }
}

/// Maps over the items it is handed and writes down which ones it was asked for.
///
/// The double the item cache needs: what has to be observable is not what it
/// answers but **which items it was made to look at**, because the whole claim
/// is that the ones already kept are not looked at again.
pub struct EachOne(pub Arc<Journal>);

impl Node for EachOne {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        let Value::List(items) = input else {
            return Err(NodeError::new("EachOne needs a list"));
        };
        for item in items.iter() {
            self.0.note(&format!("{item:?}"));
        }
        Ok(Value::list(
            items
                .iter()
                .map(|item| match item {
                    Value::Number(x) => Value::number(x * 10.0),
                    other => other.clone(),
                })
                .collect::<Vec<_>>(),
        ))
    }
}

/// One that answers with the wrong number of items, which nobody can make sense
/// of.
pub struct Miscounts;

impl Node for Miscounts {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Ok(Value::list(vec![Value::number(1.0)]))
    }
}

/// Hashes by writing the recipe down, and keeps what it is given in a map. No
/// `sha256` on purpose: a key that spells its recipe out gives two different
/// keys for two recipes **and** a failed assertion you can read.
#[derive(Default)]
pub struct Notebook {
    kept: Mutex<Vec<(Key, Kept)>>,
}

impl Notebook {
    pub fn new() -> Self {
        Self::default()
    }

    /// What each thing was kept under, in the order it was kept.
    pub fn names(&self) -> Vec<Key> {
        self.entries().into_iter().map(|(key, _)| key).collect()
    }

    /// What is kept under that name, if anything.
    pub fn under(&self, key: &Key) -> Option<Value> {
        self.entries()
            .into_iter()
            .find(|(under, _)| under == key)
            .map(|(_, kept)| kept.value)
    }

    /// What was said beside the one kept under that name.
    pub fn said_of(&self, key: &Key) -> Vec<(String, String)> {
        self.entries()
            .into_iter()
            .find(|(under, _)| under == key)
            .map(|(_, kept)| kept.meta)
            .unwrap_or_default()
    }

    fn entries(&self) -> Vec<(Key, Kept)> {
        self.kept.lock().expect("nobody poisons this mutex").clone()
    }
}

impl Keeper for Notebook {
    fn key_of(&self, value: &Value) -> Option<Key> {
        value.travels().then(|| Key::new(format!("{value:?}")))
    }

    fn combine(&self, parts: &[&str]) -> Key {
        // Length-prefixed: run together, `["ab", "c"]` and `["a", "bc"]` would
        // be one string, and two recipes under one name is the failure a cache
        // must not have.
        Key::new(
            parts
                .iter()
                .map(|part| format!("{}:{part}", part.len()))
                .collect::<Vec<_>>()
                .join("+"),
        )
    }

    fn recall(&self, keys: &[&Key]) -> Result<Vec<Option<Kept>>, KeeperError> {
        let kept = self.entries();
        Ok(keys
            .iter()
            .map(|wanted| {
                kept.iter()
                    .find(|(under, _)| &under == wanted)
                    .map(|(_, kept)| kept.clone())
            })
            .collect())
    }

    fn keep(&self, key: &Key, value: &Value, meta: &[(&str, &str)]) -> Result<(), KeeperError> {
        let kept = Kept {
            value: value.clone(),
            meta: meta
                .iter()
                .map(|(what, said)| (what.to_string(), said.to_string()))
                .collect(),
        };
        let mut inside = self.kept.lock().expect("nobody poisons this mutex");
        inside.retain(|(under, _)| under != key);
        inside.push((key.clone(), kept));
        Ok(())
    }
}

/// Keeps every fact it is told, in order. What is checked is not what the engine
/// returned but what it said while it was working.
#[derive(Default)]
pub struct Told(Mutex<Vec<Fact>>);

impl Told {
    pub fn new() -> Self {
        Self::default()
    }

    /// Everything it was told, in the order it arrived — which for a wave is not
    /// the order things happened in, and no test here may assume otherwise.
    pub fn all(&self) -> Vec<Fact> {
        self.0.lock().expect("nobody poisons this mutex").clone()
    }

    /// The names of what it was told, which is what most of these assert on.
    pub fn kinds(&self) -> Vec<String> {
        // Owned: a `Fact::Said` carries a kind that came off a wire, so
        // `flattened` borrows from the fact and there is nothing static here.
        self.all()
            .iter()
            .map(|fact| fact.flattened().0.to_string())
            .collect()
    }

    /// The nodes it was told ran, in order.
    pub fn ran(&self) -> Vec<String> {
        self.all()
            .iter()
            .filter_map(|fact| match fact {
                Fact::Ran { node, .. } => Some(node.to_string()),
                _ => None,
            })
            .collect()
    }
}

impl Watcher for Told {
    fn saw(&self, fact: &Fact) {
        self.0
            .lock()
            .expect("nobody poisons this mutex")
            .push(fact.clone());
    }
}
