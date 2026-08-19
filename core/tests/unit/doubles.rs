//! Fake nodes and drivers, shared by the other modules.
//!
//! Note that none of them declares what "kind" it is: what tells them apart is
//! the `Transition` variant they return.

use soma_next_core::{
    Cargo, Catalog, Ctx, Device, Driver, DriverError, Node, NodeError, NodeId, Outcome, Placement,
    Plan, Transition, Transport, TransportError, Value,
};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

/// Adds a constant to a number. Always finishes.
pub struct Add(pub f64);

impl Node for Add {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        match input {
            Value::Number(x) => Ok(Transition::Done(Value::number(x + self.0))),
            other => Err(NodeError::new(format!(
                "Add needs a number, it was given {}",
                other.type_name()
            ))),
        }
    }
}

/// Always fails.
pub struct Fail;

impl Node for Fail {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Err(NodeError::new("I broke"))
    }
}

/// The mean of whatever arrives along its edges. An aggregator is exactly this:
/// a node that reads a map. There is no new type behind it.
pub struct Mean;

impl Node for Mean {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
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
        Ok(Transition::Done(Value::number(
            numbers.iter().sum::<f64>() / numbers.len() as f64,
        )))
    }
}

/// Returns its input without asking for anything.
pub struct Immediate;

impl Node for Immediate {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Ok(Transition::Done(input.clone()))
    }
}

/// Asks for things on turn 0 and returns whatever it is told.
pub struct Ask(pub Vec<Value>);

impl Node for Ask {
    fn forward(&self, _input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        if ctx.turn == 0 {
            return Ok(Transition::Await(self.0.clone()));
        }
        Ok(Transition::Done(
            ctx.results.first().cloned().unwrap_or(Value::Null),
        ))
    }
}

/// Does not know how to stop.
pub struct Insatiable;

impl Node for Insatiable {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Ok(Transition::Await(vec![Value::Null]))
    }
}

/// Answers every request with its text in upper case.
pub struct Shout;

impl Driver for Shout {
    fn perform(&self, requests: &[Value]) -> Result<Vec<Value>, DriverError> {
        requests
            .iter()
            .map(|r| match r {
                Value::Text(t) => Ok(Value::text(t.to_uppercase())),
                other => Err(DriverError::new(format!(
                    "I only know how to shout text, I was given {}",
                    other.type_name()
                ))),
            })
            .collect()
    }
}

/// Answers anything, to exercise the turn limit.
pub struct AlwaysNull;

impl Driver for AlwaysNull {
    fn perform(&self, requests: &[Value]) -> Result<Vec<Value>, DriverError> {
        Ok(vec![Value::Null; requests.len()])
    }
}

// ── What it takes to look inside a wave ──

/// Where the execution went: who, in what order, and on what thread.
#[derive(Default)]
pub struct Journal(Mutex<Vec<(String, ThreadId)>>);

impl Journal {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn note(&self, who: &str) {
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
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        self.1.note(self.0);
        Ok(Transition::Done(input.clone()))
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

/// Does not finish until `how_many` have arrived.
///
/// It is how to check that two branches run **at the same time** without
/// sleeping for a single millisecond: were they executed one after the other,
/// the first would wait forever. The deadline turns that hang into a named
/// error rather than a test that never returns.
pub struct Rendezvous {
    pub point: Arc<MeetingPoint>,
    pub how_many: usize,
    /// With a message, fails with it **after** arriving at the meeting. Used to
    /// make two branches genuinely fail at the same time.
    pub fails: Option<&'static str>,
}

impl Node for Rendezvous {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
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
            None => Ok(Transition::Done(input.clone())),
        }
    }
}

/// Blows up. To check that a panic in one branch is not swallowed.
pub struct Panics;

impl Node for Panics {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        panic!("I blew up")
    }
}

/// Asks for one thing and returns the answer, but only once `how_many` have
/// arrived at the meeting: checks that two branches can keep the driver busy at
/// the same time.
pub struct RendezvousDriver {
    pub point: Arc<MeetingPoint>,
    pub how_many: usize,
}

impl Driver for RendezvousDriver {
    fn perform(&self, requests: &[Value]) -> Result<Vec<Value>, DriverError> {
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
                return Err(DriverError::new(
                    "the driver never got to be serving two branches at the same time",
                ));
            }
        }
        drop(arrived);
        Ok(vec![Value::text("served"); requests.len()])
    }
}

// ── What it takes to look inside a placement ──

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

/// Notes where it was told to run and returns its input untouched.
///
/// It is everything a core node can do with a device: see it. Moving something
/// to a GPU is not this layer's business.
pub struct Ubiquitous(pub &'static str, pub Arc<Ledger>);

impl Node for Ubiquitous {
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        self.1
            .0
            .lock()
            .expect("nobody poisons this mutex")
            .push((self.0.to_string(), ctx.device.cloned()));
        Ok(Transition::Done(input.clone()))
    }
}

// ── Fake transports ──

/// Executes the slice **right here**, noting what it was sent.
///
/// There is no process, no bytes, no pipes: that belongs to
/// `soma-next-transport`. What this exercises is the core's seam — what gets
/// sent, what comes back, where it is merged — and that is why a double that
/// never leaves its seat will do.
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
    pub placement: Placement,
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
    fn dispatch(&self, plan: &Plan, cargo: &Cargo<'_>) -> Result<Outcome, TransportError> {
        self.trips.lock().expect("nobody poisons it").push(Trip {
            plan: plan.clone(),
            input: cargo.input.clone(),
            known: cargo.known.to_vec(),
            placement: cargo.placement.clone(),
        });

        soma_next_core::Executor::new(&self.catalog)
            .placed(cargo.placement)
            .resume(plan, cargo.input.clone(), cargo.known.to_vec())
            .map_err(|e| TransportError::new(e.to_string()))
    }
}

/// Reaches nowhere, and says so.
pub struct Cable(pub &'static str);

impl Transport for Cable {
    fn dispatch(&self, _plan: &Plan, _cargo: &Cargo<'_>) -> Result<Outcome, TransportError> {
        Err(TransportError::new(self.0))
    }
}
