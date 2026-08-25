//! One machine, as this section sees it.
//!
//! # Two names, and they are not interchangeable
//!
//! A worker calls itself `node3-4127` — its hostname and its pid, because two
//! workers on one box are two workers. A graph calls it `w1`. **The machine
//! does not know that second name**, so the two only ever meet in one place: a
//! reading that came down a wire, where the client attributed the fact and the
//! reading carried the machine's own id beside it.
//!
//! Everything here follows from that. [`Standing`] is a state of the **pair**
//! and not of the machine, which is why a generic *online* / *offline* would be
//! a lie: a machine can be perfectly alive and belong to nobody.
//!
//! # Why the fields are restated rather than a `Machine` embedded
//!
//! A [`Machine`](soma_fabric_wire::Machine) already has a written form and it
//! is `said()` — `(kind, pairs)`, the one place the levels meet. Deriving
//! `Serialize` on it would give it a second one, and a type with two written
//! forms is a type whose two readers eventually disagree. So this is a view of
//! one, built by whoever answers a request, and the JSON below is **this
//! section's** and says so.

use serde::Serialize;
use soma_fabric_wire::Machine;
use soma_next_core::Host;

/// One machine that is writing readings, and what is known about it.
#[derive(Debug, Clone, Serialize)]
pub struct Seen {
    /// What it calls itself. The hostname and the pid.
    pub id: String,
    /// What a graph calls it, when the two have met. `None` is not missing
    /// data: it is a machine nobody has sent work to.
    pub named: Option<Host>,
    /// How the pair stands.
    pub standing: Standing,
    /// When it last wrote, in seconds since the epoch. The store stamps it.
    pub wrote: u64,
    /// How far behind the newest reading in this store it is, in seconds.
    ///
    /// **Behind another writer and not behind this clock**, which is the whole
    /// of it: two machines' clocks disagree by minutes on a cluster, so a panel
    /// that subtracted its own would be confidently wrong about a fleet it
    /// cannot see. The number the screen shows, because *4 s behind* is a fact
    /// and a green dot is a rule somebody applied to it.
    pub silent_for: u64,
    /// The run queue against the number of cores. `None` is **nobody measured
    /// it** and never zero.
    pub busy: Option<f64>,
    /// How many cores it divided by.
    pub cores: Option<usize>,
    /// What fraction of memory is in use.
    pub memory: Option<f64>,
    /// How long the worker process has been up, in seconds.
    pub up_s: u64,
    /// How many slices it has run since it started.
    pub served: u64,
}

/// How a name and a machine stand towards each other.
///
/// Three, and the missing fourth is deliberate: *named with no machine* is not
/// here, because a name with nothing behind it comes from a listing and a
/// listing is the broker's. This crate reads a store, and everything in a store
/// wrote itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Standing {
    /// A name and a machine that found each other, and it is still writing.
    Joined,
    /// Writing with nobody's name on it: capacity that nobody is using. It is
    /// what the idle reporting was written for, and it is the row no fleet view
    /// derived from a run can ever have.
    Loose,
    /// It has stopped writing.
    ///
    /// **It wins over the other two**, and keeps whatever name it had beside
    /// it: what somebody needs to see about `w1` gone quiet is that it is `w1`.
    Quiet,
}

impl Seen {
    /// One machine, from its reading and when the store stamped it.
    ///
    /// `silent_for` is handed in rather than taken from a clock here so that
    /// every row of one answer is measured against the same instant — two rows
    /// a wall clock apart would sort by a difference nobody made.
    pub fn of(
        machine: Machine,
        wrote: u64,
        silent_for: u64,
        named: Option<Host>,
        quiet_after: u64,
    ) -> Self {
        Self {
            standing: match (silent_for > quiet_after, &named) {
                (true, _) => Standing::Quiet,
                (false, Some(_)) => Standing::Joined,
                (false, None) => Standing::Loose,
            },
            id: machine.id,
            named,
            wrote,
            silent_for,
            busy: machine.busy,
            cores: machine.cores,
            memory: machine.memory,
            up_s: machine.up.as_secs(),
            served: machine.served,
        }
    }
}
