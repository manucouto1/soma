//! The engine's [`Watcher`], filled in by a [`Store`]: what happened, kept.
//!
//! The hole the core left, plugged with the thing the core cannot have —
//! somewhere for the record to live. What arrives here is a stream of facts;
//! what is written is **one record per `forward`**, which is a different grain
//! and deliberately so.
//!
//! # Why the grain is a `forward` and not a node
//!
//! Five nodes trained ten thousand steps are fifty thousand node executions.
//! A record each would be fifty thousand writes and a scan nobody can afford;
//! one record for the whole training run would have no step 500 in it at all.
//! The `forward` is the unit the engine actually has: a [`Plan`] is walked once
//! per one, and [`Fact::Finished`] is emitted by exactly that walk.
//!
//! # The shape, which is CU18's and not a new one
//!
//! ```text
//! run/<id>/<n>
//! ```
//!
//! In the **record**, which a scan already carries:
//!
//! ```text
//! run            = the id this run was given, or the one it was handed
//! forward        = which one, from zero
//! took_us        = how long all of it took
//! state          = ok | broke
//! nodes          = how many ran
//! <kind>.<field> = whatever was asked for with `summarising`
//! ```
//!
//! In the **blob**, for whoever wants the detail: every fact, flattened, in the
//! order it arrived. So "how is it going" costs one scan and no fetches, and
//! only the detail is paid for — the same split, for the same reason, as a
//! trial's record.
//!
//! The last row is what keeps a **training curve** on the cheap side of that
//! line: ten thousand losses read one blob at a time is ten thousand round
//! trips, and it is the one number somebody wants from every step.
//!
//! # Two ways in, and they are not the same question
//!
//! [`saw`](Watcher::saw) is the engine's, and a terminal fact closes a record.
//! [`said`](Recorder::said) is for whatever vocabulary is not the engine's —
//! level 2's loss, which is Python's and arrives **after** the `forward` it
//! belongs to has already ended. That one is written into the record that was
//! last closed, rewriting it, which a store already does: a name is a question
//! and its answer can be refreshed.
//!
//! Guessing which of the two a fact was would have been impossible and there is
//! no need: they come through different doors.
//!
//! # What it does not do
//!
//! It does not judge. Whether 400 ms is slow is CU21's opinion about this, and
//! the invariant that keeps the two apart is that the opinion has to be
//! reachable from what is written here without running anything again.

use crate::{Meta, Store, StoreError};
use somatize_core::{Fact, Watcher};
use std::sync::{Arc, Mutex};

/// One fact as it is written: a name and text-to-text fields.
type Written = (String, Vec<(String, String)>);

/// What is being accumulated for the `forward` in flight.
#[derive(Default)]
struct Pending {
    /// Which `forward` this is, from zero.
    which: usize,
    /// Its facts, in the order they arrived — which for a wave is not the order
    /// they happened in, and nothing here pretends otherwise.
    facts: Vec<Written>,
    /// Whether a terminal fact has already closed it.
    closed: bool,
}

/// Writes down what happened, one record per `forward`.
///
/// It **owns** its store, where a [`Cache`](crate::Cache) borrows one, and the
/// difference is a fact about their lives: a cache is made for one `forward`
/// and a recorder counts them, so it outlives every call it is passed to.
pub struct Recorder {
    store: Arc<dyn Store>,
    run: String,
    /// Which kinds of fact are worth having in the record itself and not only
    /// in the blob. See [`summarising`](Self::summarising).
    summarising: Vec<String>,
    pending: Mutex<Pending>,
}

impl Recorder {
    /// A recorder over this store, under a name of its own.
    ///
    /// The name is made here and can be read back with [`run`](Self::run),
    /// because a `forward` in a notebook has no reason to invent one and still
    /// has to be findable afterwards.
    pub fn over(store: Arc<dyn Store>) -> Self {
        Self::named(store, made_up())
    }

    /// The same, under a name you chose: a training run that wants to be found
    /// again by the name it already has.
    pub fn named(store: Arc<dyn Store>, run: impl Into<String>) -> Self {
        Self {
            store,
            run: run.into(),
            summarising: Vec::new(),
            pending: Mutex::new(Pending::default()),
        }
    }

    /// The same recorder, with these kinds of fact carried **in the record** and
    /// not only in the blob, as `<kind>.<field>`.
    ///
    /// It is the lesson CU18 already paid for. A trial keeps its score beside
    /// its configuration in the record, so a sampler rebuilds a whole history
    /// with **one scan and no fetches**, and only a pruner comparing curves pays
    /// for blobs. The same question is asked here of every training curve ever
    /// drawn: ten thousand steps read one at a time is ten thousand round trips
    /// against a bucket, and the number wanted from each of them is one.
    ///
    /// Which kinds those are is the caller's, exactly as the vocabulary is:
    /// `Recorder::over(store).summarising(["loss"])` is what a training run
    /// wants, and this crate does not learn what a loss is to know it.
    ///
    /// The **last** fact of each kind in a `forward` is the one carried. A
    /// forward has one loss; if it somehow had two, the one that stands is the
    /// one that was said last.
    pub fn summarising(mut self, kinds: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.summarising = kinds.into_iter().map(Into::into).collect();
        self
    }

    /// What this run is called, which is the first half of every name it writes.
    pub fn run(&self) -> &str {
        &self.run
    }

    /// One fact from a vocabulary that is not the engine's.
    ///
    /// It lands in the `forward` in flight, or — if that one has already ended,
    /// which is the normal case for a loss — in the one that ended last, whose
    /// record is rewritten. Nothing is lost and nothing lands in the wrong step.
    ///
    /// The fields are text to text because that is what a record is; the
    /// vocabulary is the caller's, exactly as an [`Artifact`]'s `kind` is.
    ///
    /// [`Artifact`]: https://docs.rs/somatize-fabric-wire
    pub fn said(&self, kind: &str, fields: Vec<(String, String)>) {
        let mut pending = self.pending.lock().expect("nobody poisons this mutex");
        pending.facts.push((kind.to_string(), fields));
        if pending.closed {
            // Already written once. A name is a question and the answer can be
            // refreshed, so this is a rebind of the same record and not a new
            // one — the same thing a trial's record does on every report.
            self.write(&pending);
        }
    }

    /// The name of one `forward`'s record. `run/<id>/<n>`, which is
    /// `<study>/trial/<n>/<attempt>` with a different noun in it: the level
    /// above, and a number.
    fn name(&self, which: usize) -> String {
        format!("run/{}/{which}", self.run)
    }

    /// Writes what is pending. Failing is reported and not returned: there is no
    /// useful answer to "the record could not be written" in the middle of a
    /// run, and stopping one because its log could not be kept would be the
    /// observability layer breaking the thing it observes.
    fn write(&self, pending: &Pending) {
        let blob = match blob(&pending.facts) {
            Ok(blob) => blob,
            Err(why) => return eprintln!("what happened could not be written down: {why}"),
        };
        let written = self.store.put(&blob).and_then(|digest| {
            self.store.bind(
                &self.name(pending.which),
                &digest,
                meta(&self.run, &self.summarising, pending),
            )
        });
        if let Err(why) = written {
            eprintln!("what happened could not be kept: {why}");
        }
    }
}

impl Watcher for Recorder {
    fn saw(&self, fact: &Fact) {
        let (kind, fields) = fact.flattened();
        let mut pending = self.pending.lock().expect("nobody poisons this mutex");
        // A fact after a closed record is the next `forward` beginning. Only the
        // engine's door does this: level 2's belongs to the one that ended.
        if pending.closed {
            let next = pending.which + 1;
            *pending = Pending {
                which: next,
                ..Pending::default()
            };
        }
        pending.facts.push((kind.to_string(), fields));
        if fact.ends_a_run() {
            pending.closed = true;
            self.write(&pending);
        }
    }
}

/// What a scan carries, so that "how is it going" costs no fetches.
///
/// Read back off the facts rather than counted as they arrive: a record that is
/// rewritten has to say the same thing about the same facts however many times
/// it is written, and a counter that only goes up would not.
fn meta(run: &str, summarising: &[String], pending: &Pending) -> Meta {
    let how_many = |kind: &str| pending.facts.iter().filter(|(one, _)| one == kind).count();
    let mut meta = vec![
        ("run".to_string(), run.to_string()),
        ("forward".to_string(), pending.which.to_string()),
        (
            "state".to_string(),
            match how_many("broke") {
                0 => "ok".to_string(),
                _ => "broke".to_string(),
            },
        ),
        ("nodes".to_string(), how_many("ran").to_string()),
    ];
    if let Some(took) = field_of(&pending.facts, "finished", "took_us") {
        meta.push(("took_us".to_string(), took.to_string()));
    }
    for kind in summarising {
        let Some((_, fields)) = pending.facts.iter().rev().find(|(one, _)| one == kind) else {
            continue;
        };
        for (name, what) in fields {
            meta.push((format!("{kind}.{name}"), what.clone()));
        }
    }
    meta
}

/// One field of the last fact of that kind, if it is there.
fn field_of<'f>(facts: &'f [Written], kind: &str, field: &str) -> Option<&'f str> {
    facts
        .iter()
        .rev()
        .find(|(one, _)| one == kind)?
        .1
        .iter()
        .find(|(name, _)| name == field)
        .map(|(_, what)| what.as_str())
}

/// The detail, as JSON: whoever reads a record is another process, often on
/// another machine and sometimes a notebook, and none of them should need this
/// library's version of anything to look at it.
fn blob(facts: &[Written]) -> Result<Vec<u8>, StoreError> {
    let said: Vec<_> = facts
        .iter()
        .map(|(kind, fields)| {
            let mut one = serde_json::Map::new();
            one.insert("fact".to_string(), serde_json::Value::String(kind.clone()));
            for (name, what) in fields {
                one.insert(name.clone(), serde_json::Value::String(what.clone()));
            }
            serde_json::Value::Object(one)
        })
        .collect();
    serde_json::to_vec_pretty(&said)
        .map_err(|e| StoreError::Corrupt(format!("that record cannot be written: {e}")))
}

/// A name for a run nobody named. The pid and a counter, which is enough:
/// two runs in one process are two numbers, and two processes are two pids.
fn made_up() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static RUNS: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}",
        std::process::id(),
        RUNS.fetch_add(1, Ordering::Relaxed)
    )
}
