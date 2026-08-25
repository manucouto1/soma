//! What a machine says about itself. The one thing no record can derive.
//!
//! Everything else about a worker is already written down by whoever asked it
//! to do something: which slices crossed, how long the round trip took, what
//! ran over there and what it produced. Turn the record on its side and you
//! have a fleet view without anybody keeping a registry.
//!
//! What is **not** in there is the machine — how loaded it is, how much memory
//! is left, how long it has been up. Nobody on the other end can work that out,
//! and it is the half of *see the health of the workers* that a scan cannot
//! answer.
//!
//! # A level of its own, and it stays out of the core
//!
//! A load average is not a fact about a graph. Putting it in
//! [`Fact`](soma_next_core::Fact) as its own variant would be the engine
//! learning what a machine is, which is the mistake that keeps `loss` out of
//! the core too. So the vocabulary lives **here**, where a host is already a
//! thing, and it crosses as `(kind, pairs)` inside
//! [`Fact::Said`](soma_next_core::Fact::Said) — the shape CU20 named as the one
//! place the levels meet.
//!
//! Which costs nothing on the wire. `Answer::Saw` already carries a `Fact`, the
//! client already relays one straight to its watcher, and the engine already
//! wraps whatever arrives in [`Fact::Elsewhere`] — so this **arrives saying
//! which host it came from** without one line attributing it, and no message
//! had to be added to the protocol.
//!
//! # It is read, not judged
//!
//! No thresholds, here or anywhere near here. A machine at 0.9 busy is a
//! machine at 0.9 busy; whether that is bad is somebody's opinion at a bound,
//! and this library keeps those in `health/` where they can be argued with
//! against a record that has already been written.

use soma_next_core::Fact;
use std::time::Duration;

/// What one machine looks like right now.
///
/// A struct and not an enum: these are not alternatives, they are a snapshot,
/// and every one of them is measured at the same instant. `None` is **nobody
/// measured it** and never zero — a kernel that does not keep a load average is
/// not a machine that is idle.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Machine {
    /// How long this worker process has been up.
    ///
    /// Its own monotonic clock and never a wall clock: two machines' wall
    /// clocks disagree by minutes on a cluster as a matter of course, and CU21
    /// already ruled that two of them would not have composed. What the reader
    /// gets is a duration, and *when* is stamped by whoever writes it down.
    pub up: Duration,
    /// The run queue against the number of cores, so two machines of different
    /// sizes can be compared.
    ///
    /// The ratio and not the raw load, because a load of 8 is a busy laptop and
    /// an idle compute node. [`Machine::cores`] is beside it for whoever wants
    /// to undo the division.
    pub busy: Option<f64>,
    /// How many cores it divided by.
    pub cores: Option<usize>,
    /// What fraction of memory is in use.
    ///
    /// Against what the kernel says is **available** rather than what is free:
    /// page cache is not memory anybody is short of, and counting it as used is
    /// how a perfectly healthy machine reads as full.
    pub memory: Option<f64>,
    /// How many slices this worker has run since it started.
    pub served: u64,
    /// What this machine calls **itself**: its hostname and this process.
    ///
    /// Not the name the graph gave it. **A worker does not know that name** —
    /// `w1` is the client's word and it is why the client is the one that
    /// attributes a fact. So a reading written to a store, where there is no
    /// client to attribute it, has to be filed under something the worker can
    /// know on its own, and this is it. Whoever reads joins the two by seeing
    /// the same `id` on a reading that **did** come down a wire.
    pub id: String,
}

impl Machine {
    /// A reading of the machine this is running on.
    ///
    /// Everything comes from `/proc`, which is Linux. Elsewhere the fields are
    /// `None` and say so by being `None` — a worker on a laptop still reports
    /// its uptime and how much it has served, which is the part that never
    /// needed a kernel.
    pub fn here(up: Duration, served: u64) -> Self {
        let cores = std::thread::available_parallelism()
            .map(|one| one.get())
            .ok();
        Self {
            up,
            busy: load().zip(cores).map(|(load, cores)| load / cores as f64),
            cores,
            memory: memory(),
            served,
            id: mine(),
        }
    }

    /// This reading as the fact that crosses, already flat.
    ///
    /// Named `machine`, which is what it will be written down as and what a
    /// reader filters on. A field that was not measured is **absent** rather
    /// than empty: the record has no null and a reader that finds no `busy`
    /// has to be able to tell *this kernel does not say* from *nothing is
    /// running*.
    pub fn said(&self) -> Fact {
        let mut pairs = vec![
            ("up_us".into(), self.up.as_micros().to_string()),
            ("served".into(), self.served.to_string()),
        ];
        if !self.id.is_empty() {
            pairs.push(("id".into(), self.id.clone()));
        }
        for (name, what) in [("busy", self.busy), ("memory", self.memory)] {
            if let Some(one) = what.filter(|one| one.is_finite()) {
                pairs.push((name.into(), format!("{one:.4}")));
            }
        }
        if let Some(cores) = self.cores {
            pairs.push(("cores".into(), cores.to_string()));
        }
        Fact::Said {
            kind: "machine".into(),
            pairs,
        }
    }

    /// A reading, back out of the pairs [`said`](Self::said) wrote.
    ///
    /// Here and not wherever it is read, which is the whole point: two halves
    /// of one format in two crates drift, and the day a field is added the
    /// writer and the reader would each look right and disagree about what a
    /// reading is. Whoever reads a store, a record or a fact off a wire gets
    /// the same answer because there is only one of these.
    ///
    /// **What is not there is `None`**, which is the rule the other half writes
    /// by: absent means nobody measured it and never that something is zero. A
    /// field that *is* there and will not parse is treated the same way, on
    /// purpose — a reading written by a version that says `busy` differently is
    /// still a reading, and refusing all of it would throw away the uptime for
    /// the sake of the load average. It is the same decision this file already
    /// makes about a `/proc` line it does not understand.
    pub fn read(pairs: &[(String, String)]) -> Self {
        Self {
            // Absent is zero here and only here: `up` and `served` are not
            // measurements of a kernel, they are what the process itself
            // counted, and a reading that does not carry them is one this
            // version cannot read rather than a machine that did not say.
            up: Duration::from_micros(parsed(pairs, "up_us").unwrap_or(0)),
            busy: parsed(pairs, "busy"),
            cores: parsed(pairs, "cores"),
            memory: parsed(pairs, "memory"),
            served: parsed(pairs, "served").unwrap_or(0),
            id: beside(pairs, "id").unwrap_or_default().to_string(),
        }
    }
}

/// One field of a reading, if it is there.
fn beside<'p>(pairs: &'p [(String, String)], what: &str) -> Option<&'p str> {
    pairs
        .iter()
        .find(|(name, _)| name == what)
        .map(|(_, said)| said.as_str())
}

/// The same, read as whatever it should be. Anything that will not parse is
/// nobody having said it, which is [`Machine::read`]'s whole rule.
fn parsed<T: std::str::FromStr>(pairs: &[(String, String)], what: &str) -> Option<T> {
    beside(pairs, what)?.parse().ok()
}

/// Where a reading of this machine is filed in a store.
///
/// **One name per machine and rewritten every time**, not one per reading. That
/// is CU18's shape and it buys two things: a store that does not grow while a
/// worker sits there, and liveness for free — the store stamps every write, so
/// a reading that has not moved is a machine that has stopped, and finding that
/// out is a scan with no fetches.
pub fn filed(id: &str) -> String {
    format!("machine/{id}")
}

/// What this machine calls itself: its hostname, and which process on it.
///
/// The pid matters: two workers on one box are two workers, and filing both
/// under the hostname would have the second quietly overwriting the first.
fn mine() -> String {
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|one| one.trim().to_string())
        .filter(|one| !one.is_empty())
        .unwrap_or_else(|| "unknown".into());
    format!("{host}-{}", std::process::id())
}

/// The one-minute run queue, out of `/proc/loadavg`.
fn load() -> Option<f64> {
    std::fs::read_to_string("/proc/loadavg")
        .ok()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// What fraction of memory is in use, out of `/proc/meminfo`.
fn memory() -> Option<f64> {
    let said = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total: Option<f64> = None;
    let mut free: Option<f64> = None;
    for line in said.lines() {
        // A line this does not understand is skipped and not fatal: losing the
        // whole reading because one kernel grew a field is the kind of thing
        // that goes wrong on the machine you cannot log into.
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        let value = rest
            .split_whitespace()
            .next()
            .and_then(|one| one.parse().ok());
        match name {
            "MemTotal" => total = value,
            "MemAvailable" => free = value,
            _ => {}
        }
    }
    let (total, free) = (total?, free?);
    (total > 0.0).then(|| (1.0 - free / total).clamp(0.0, 1.0))
}
