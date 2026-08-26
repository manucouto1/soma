//! Everything that is out there, once.
//!
//! One request, one answer: a scan of the store for who is writing, a bounded
//! read of the record for what the graphs have been calling them, and the join.
//! Nothing is held between the two — see the crate docs for why that is what
//! makes this the same code in a monolith and behind a URL.

use crate::naming;
use crate::seen::Seen;
use serde::Serialize;
use somatize_fabric_wire::Machine;
use somatize_store::{Bound, Store, StoreError};
use std::time::{SystemTime, UNIX_EPOCH};

/// What is out there, and the rules that were applied to say so.
///
/// The rules travel with the answer on purpose. *Quiet* is not a fact in the
/// store, it is a bound somebody chose, and a screen that showed it without
/// saying which bound would be presenting an opinion as a reading.
#[derive(Debug, Clone, Serialize)]
pub struct Fleet {
    /// The machines, by what they call themselves.
    pub seen: Vec<Seen>,
    /// After how many seconds behind the newest writer a machine was called
    /// quiet.
    pub quiet_after_s: u64,
    /// How many records were read to learn the names.
    pub read_records: usize,
    /// When the newest reading in this store was written, seconds since the
    /// epoch. It is the instant every row is measured against.
    pub newest: u64,
    /// And how long ago that was **by the clock of whoever is reading**.
    ///
    /// The one number here that crosses between clocks, and it is at the level
    /// where it is worth it: the honest hole in measuring writer against writer
    /// is that a fleet where *everything* stopped has no newest write to be
    /// behind, so nothing looks quiet — and that is exactly when somebody is
    /// looking. It can be negative, and saying `-4` is more use than pretending
    /// it is zero; `None` is a store nobody has written a reading into.
    pub since_newest_s: Option<i64>,
}

impl Fleet {
    /// Everything writing into this store.
    ///
    /// How far behind each machine is, is measured against **the newest reading
    /// in this store** and not against the clock of whoever is asking. Those are
    /// two clocks on two machines, and on a cluster they disagree by minutes as
    /// a matter of course — so a panel run from a laptop whose clock drifted
    /// would declare a working fleet dead, or a dead one working.
    ///
    /// It has an honest hole and the answer carries it: a fleet where
    /// **everything** stopped has no newest write to be behind, so nothing looks
    /// quiet. That is what [`Fleet::since_newest_s`] is for.
    ///
    /// `quiet_after` is **told and not derived**: the store keeps one reading
    /// per machine and rewrites it, so there is no history to work a cadence out
    /// of, and a panel that guessed would call a machine dead for reporting
    /// slowly. `read_records` bounds the join — a fetch each; the rest is a scan.
    pub fn read(
        store: &dyn Store,
        quiet_after: u64,
        read_records: usize,
    ) -> Result<Self, StoreError> {
        let named = naming::names(store, read_records)?;
        let readings: Vec<Bound> = store
            .bound()?
            .into_iter()
            .filter(|record| record.name.starts_with(FILED))
            .collect();
        // The instant every row is measured against, and it is one of theirs.
        let newest = readings.iter().map(|one| one.when).max().unwrap_or(0);

        let mut seen: Vec<Seen> = readings
            .into_iter()
            .filter_map(|record| {
                let id = record.name.strip_prefix(FILED)?;
                let machine = Machine::read(&record.meta);
                // The name it is filed under is the one to believe: a reading
                // whose `id` field went missing is still that machine's.
                let machine = Machine {
                    id: id.to_string(),
                    ..machine
                };
                Some(Seen::of(
                    machine,
                    record.when,
                    newest.saturating_sub(record.when),
                    named.get(id).cloned(),
                    quiet_after,
                ))
            })
            .collect();

        // By the name it was filed under, and never by a number that moves. A
        // list that reordered itself as the load changed would move the row
        // somebody is reading, which is the same reason the drawing shows the
        // load and does not sort by it.
        seen.sort_by(|one, other| one.id.cmp(&other.id));

        // The one crossing between clocks, made deliberately and said as such.
        let here = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_secs() as i64)
            .unwrap_or(0);
        Ok(Self {
            since_newest_s: (newest > 0).then(|| here - newest as i64),
            seen,
            quiet_after_s: quiet_after,
            read_records,
            newest,
        })
    }
}

/// The head of the name a reading is filed under.
///
/// The wire's `filed` builds the whole thing and owns the decision; this is the
/// prefix a scan filters on. That the two still agree is a test and not a
/// comment — `tests/unit/fleet.rs` asks `filed` itself.
static FILED: &str = "machine/";
