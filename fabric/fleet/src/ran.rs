//! One run, with the record turned the other way up.
//!
//! The record is written run → `forward` → node, and *where* is an attribute of
//! a fact. **What did this machine do** is a question nobody can ask of it in
//! that shape, and it is the one somebody with three hosts came for.
//!
//! # The column that only exists up here
//!
//! `waiting_us` is the round trip **minus** what actually ran over there: the
//! wire, the queue and the codec. It is the number that says whether sending it
//! was worth it, and no per-node view can produce it because neither half of the
//! subtraction belongs to a node — `left` is the client's fact and `ran` is the
//! worker's.
//!
//! Never below zero. A `left` counted on one `forward` and the work it carried
//! counted on another would otherwise read as a machine that finished before it
//! was asked.
//!
//! # And the machine says the half no record can derive
//!
//! How loaded it is. That arrives as a reading wrapped in `Fact::Elsewhere`,
//! under the graph's name, and the **newest wins**: a reading is a snapshot and
//! the question is what the machine is like now, not what it averaged.
//!
//! # What this costs
//!
//! A scan and a fetch per `forward`. The same as the join next door and for the
//! same reason: what is in the scan is the summary, and the hosts are in the
//! blob.

use serde::Serialize;
use soma_next_store::{Bound, Store, StoreError};
use std::collections::{BTreeMap, BTreeSet};

/// Where a `Recorder` writes.
const RUNS: &str = "run/";

/// What the client's own process is called in this view.
///
/// A row like any other, because it is one: things ran here too, and a fleet
/// view that only showed other people's machines would be hiding the machine
/// the graph was declared on.
const HERE: &str = "aquí";

/// One run, as a scan already carries it.
#[derive(Debug, Clone, Serialize)]
pub struct Run {
    /// The id it was given, or the one it made up.
    pub run: String,
    /// How many `forward`s are written down.
    pub forwards: usize,
    /// When the last of them was written.
    pub when: u64,
    /// Whether any of them broke.
    pub broke: bool,
}

/// One run, by where its work happened.
#[derive(Debug, Clone, Serialize)]
pub struct Ran {
    /// Which run.
    pub run: String,
    /// How many `forward`s were read.
    pub forwards: usize,
    /// One row per place, this process included.
    pub did: Vec<Did>,
}

/// What one place did in one run.
#[derive(Debug, Clone, Serialize)]
pub struct Did {
    /// The name the graph gave it, or `aquí`.
    pub host: String,
    /// What the machine calls itself, when a reading came down the wire.
    pub id: Option<String>,
    /// How many slices crossed to it.
    pub slices: u64,
    /// How long the round trips took, all told.
    pub trip_us: u64,
    /// How many nodes ran there, and how long they took.
    pub ran: u64,
    /// How many did not answer.
    pub failed: u64,
    /// What running them cost.
    pub took_us: u64,
    /// The round trip minus the work: the wire, the queue and the codec.
    pub waiting_us: u64,
    /// Which nodes, by name.
    pub nodes: Vec<String>,
    /// The newest reading that came down the wire, where there was one.
    pub busy: Option<f64>,
    /// The same.
    pub memory: Option<f64>,
    /// The same.
    pub cores: Option<usize>,
    /// The same.
    pub served: Option<u64>,
}

/// Every run this store has a record of, newest last.
///
/// One scan and no fetches: what is here is what the record already carries, so
/// choosing which run to look at costs nothing.
pub fn runs(store: &dyn Store) -> Result<Vec<Run>, StoreError> {
    let mut runs: BTreeMap<String, Run> = BTreeMap::new();
    for record in records(store)? {
        let Some(named) = beside(&record, "run") else {
            continue;
        };
        let one = runs.entry(named.to_string()).or_insert_with(|| Run {
            run: named.to_string(),
            forwards: 0,
            when: 0,
            broke: false,
        });
        one.forwards += 1;
        one.when = one.when.max(record.when);
        one.broke |= beside(&record, "state") == Some("broke");
    }
    let mut runs: Vec<Run> = runs.into_values().collect();
    runs.sort_by_key(|one| one.when);
    Ok(runs)
}

/// One run, by where its work happened. `last` bounds the fetches.
pub fn ran(store: &dyn Store, run: &str, last: usize) -> Result<Ran, StoreError> {
    let mut mine: Vec<Bound> = records(store)?
        .into_iter()
        .filter(|record| beside(record, "run") == Some(run))
        .collect();
    // By which `forward` it is and not by when it was written: a record that a
    // loss rewrote is stamped later than the one after it.
    mine.sort_by_key(|record| {
        beside(record, "forward")
            .and_then(|which| which.parse::<usize>().ok())
            .unwrap_or(0)
    });
    let read = mine.len().saturating_sub(last.min(mine.len()));
    let mine = &mine[read..];

    let mut did: BTreeMap<String, Did> = BTreeMap::new();
    let mut nodes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for record in mine {
        let Ok(Some(blob)) = store.get(&record.digest) else {
            continue;
        };
        let Ok(serde_json::Value::Array(facts)) = serde_json::from_slice(&blob) else {
            continue;
        };
        for fact in facts {
            let Some(said) = fact.as_object() else {
                continue;
            };
            let host = said
                .get("host")
                .and_then(|one| one.as_str())
                .unwrap_or(HERE)
                .to_string();
            let kind = said.get("fact").and_then(|one| one.as_str()).unwrap_or("");
            let one = did.entry(host.clone()).or_insert_with(|| Did {
                host: host.clone(),
                id: None,
                slices: 0,
                trip_us: 0,
                ran: 0,
                failed: 0,
                took_us: 0,
                waiting_us: 0,
                nodes: Vec::new(),
                busy: None,
                memory: None,
                cores: None,
                served: None,
            });
            let took = |what: &str| -> u64 {
                said.get(what)
                    .and_then(|one| one.as_str())
                    .and_then(|one| one.parse().ok())
                    .unwrap_or(0)
            };
            match kind {
                // The half no record can derive, and the newest wins: it is a
                // snapshot, and the question is what the machine is like now.
                "machine" => {
                    if let Some(id) = said.get("id").and_then(|one| one.as_str()) {
                        one.id = Some(id.to_string());
                    }
                    one.busy = number(said.get("busy")).or(one.busy);
                    one.memory = number(said.get("memory")).or(one.memory);
                    one.cores = number(said.get("cores"))
                        .map(|one| one as usize)
                        .or(one.cores);
                    one.served = number(said.get("served"))
                        .map(|one| one as u64)
                        .or(one.served);
                }
                // The client's fact about a slice that crossed.
                "left" => {
                    one.slices += 1;
                    one.trip_us += took("took_us");
                }
                _ => {
                    if let Some(node) = said.get("node").and_then(|one| one.as_str()) {
                        nodes.entry(host).or_default().insert(node.to_string());
                    }
                    if kind == "ran" || kind == "failed" {
                        if kind == "ran" {
                            one.ran += 1;
                        } else {
                            one.failed += 1;
                        }
                        one.took_us += took("took_us");
                    }
                }
            }
        }
    }

    for (host, one) in did.iter_mut() {
        one.nodes = nodes.remove(host).unwrap_or_default().into_iter().collect();
        one.waiting_us = one.trip_us.saturating_sub(one.took_us);
    }

    Ok(Ran {
        run: run.to_string(),
        forwards: mine.len(),
        // `aquí` first, and the rest by name: the machine the graph was
        // declared on is where somebody's eye starts.
        did: {
            let mut rows: Vec<Did> = did.into_values().collect();
            rows.sort_by_key(|one| (one.host != HERE, one.host.clone()));
            rows
        },
    })
}

/// Every `forward`'s record in this store.
fn records(store: &dyn Store) -> Result<Vec<Bound>, StoreError> {
    Ok(store
        .bound()?
        .into_iter()
        .filter(|one| one.name.starts_with(RUNS))
        .collect())
}

/// One field of a record's summary.
fn beside<'b>(record: &'b Bound, what: &str) -> Option<&'b str> {
    record
        .meta
        .iter()
        .find(|(name, _)| name == what)
        .map(|(_, said)| said.as_str())
}

/// A number a record wrote as text. Anything that will not parse is nobody
/// having said it, which is the rule the whole format reads by.
fn number(said: Option<&serde_json::Value>) -> Option<f64> {
    said?.as_str()?.parse().ok()
}
