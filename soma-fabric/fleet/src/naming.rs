//! What the graphs have been calling these machines.
//!
//! The join, and the only expensive thing this crate does.
//!
//! # Where the two names are ever in one place
//!
//! Nowhere a machine writes. A reading filed in a store has no client to
//! attribute it, so it is filed under what the machine calls itself. The pair
//! only appears in **the record of a run**: a reading that came down a wire is
//! wrapped in `Fact::Elsewhere`, which flattens to a `host` field on whatever
//! it wrapped, and the reading it wrapped carries the machine's own `id`. So
//! one object in one blob has both:
//!
//! ```json
//! { "fact": "machine", "host": "w1", "id": "node3-4127", "busy": "0.4213" }
//! ```
//!
//! # And why it is not inferred any other way
//!
//! Because a broker's listing says `w1` is at `node3:7000` and a machine calls
//! itself `node3-4127`, it is tempting to match the hostnames and be right
//! almost always. That is the inference `Path::InProcess` refuses for the same
//! reason — a guess that is usually right is a bug that is occasionally silent
//! — and it would be wrong exactly where it costs most: two workers on one box,
//! which is the case the pid is in the id for.
//!
//! # The price, said out loud
//!
//! Reading which machines are there is a scan and no fetches, because the whole
//! of a reading is in its record. This is **a scan and a fetch per `forward`**,
//! which is what the same join costs in `somatize.record.fleet` and for the
//! same reason: the host lives in the blob. `last` is the bound on it, and the
//! question worth asking of a fleet that is working now is the last few
//! anyway.

use somatize_core::Host;
use somatize_store::{Bound, Store, StoreError};
use std::collections::BTreeMap;

/// Where a `Recorder` writes. Known here because a reader has to look
/// somewhere; owned by whoever writes it.
const RUNS: &str = "run/";

/// What each machine has been called, from the last `how_many` records.
///
/// Newest first, so a machine that was `w1` yesterday and `gpu` this morning
/// comes back as `gpu`. A name is not a fact about a machine — it is a fact
/// about a run — and the newest one is the only one that could still be true.
pub fn names(store: &dyn Store, how_many: usize) -> Result<BTreeMap<String, Host>, StoreError> {
    let mut records: Vec<Bound> = store
        .bound()?
        .into_iter()
        .filter(|one| one.name.starts_with(RUNS))
        .collect();
    // Newest first, which a descending key says by negating rather than by
    // reversing the comparison.
    records.sort_by_key(|one| std::cmp::Reverse(one.when));

    let mut named = BTreeMap::new();
    for record in records.into_iter().take(how_many) {
        let Ok(Some(blob)) = store.get(&record.digest) else {
            // A record whose bytes are gone is one fewer place to look, not a
            // reason to answer nothing: the fleet is still the fleet.
            continue;
        };
        for (id, host) in said_by_a_machine(&blob) {
            // Whoever got here first wins, and the sort put the newest first.
            named.entry(id).or_insert(host);
        }
    }
    Ok(named)
}

/// The `(id, host)` pairs in one record's blob.
///
/// A blob this version cannot read is skipped rather than fatal. It is the
/// decision the wire already makes about a `/proc` line it does not understand,
/// and the reason is the same: the thing that goes wrong on a machine you
/// cannot log into should cost you that thing and not everything.
fn said_by_a_machine(blob: &[u8]) -> Vec<(String, Host)> {
    let Ok(serde_json::Value::Array(facts)) = serde_json::from_slice(blob) else {
        return Vec::new();
    };
    facts
        .iter()
        .filter_map(|fact| {
            let said = fact.as_object()?;
            // Only a reading carries both names. Every other fact from over
            // there has the host and knows nothing about what the machine
            // calls itself, which is the half that cannot be guessed.
            (said.get("fact")?.as_str()? == "machine").then_some(())?;
            let id = said.get("id")?.as_str()?;
            let host = said.get("host")?.as_str()?;
            Some((id.to_string(), Host::new(host)))
        })
        .collect()
}
