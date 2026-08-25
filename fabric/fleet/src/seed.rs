//! A fleet to look at: a store with machines, names and a run already in it.
//!
//! Not a test double and not a demo mode. It writes, through the store's own
//! public types, exactly what a worker and a run would have written — so what
//! the screens draw from it is what they will draw from a cluster, and the day
//! a format moves this stops working rather than going on lying.
//!
//! # Why a fixture can have a past at all
//!
//! A store stamps a name when it is bound, so a reading written now is a
//! machine that is up to date, and there is no way to ask it for a stamp from
//! ten minutes ago. That would make *quiet* impossible to see without waiting
//! for it.
//!
//! What makes it possible is the same decision that made liveness right: how
//! far behind a machine is, is measured **against the newest reading in the
//! store** and never against anybody's clock. So a fixture does not need a past
//! — it needs a *spread*. The stamps here are offsets from one instant, and the
//! fleet they describe reads the same in a minute and in a year.
//!
//! The record is written by hand at the store's documented path rather than
//! through `bind`, because `bind` is the thing that stamps. It is the store's
//! own public [`Bound`] at the store's own documented layout, and
//! `tests/unit/seed.rs` reads every one of them back through
//! [`Store::bound`] — so if that layout ever moves, this fails loudly instead of
//! seeding a store nobody can read.

use crate::listing::{Listed, Listing};
use serde_json::json;
use soma_fabric_wire::{Machine, filed};
use soma_next_store::{Bound, Digest, Meta, Store, StoreError};
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// What was sown, so whoever ran it knows what to go and look at.
pub struct Sown {
    /// How many machines are writing.
    pub machines: usize,
    /// How many names the listing carries.
    pub names: usize,
    /// The run that was written.
    pub run: String,
}

/// Writes a fleet into this directory: five machines, a run across two of them,
/// and — if a path is given — a listing that names four.
///
/// The five are chosen to be every state there is, because a fixture that only
/// shows the ordinary case is a fixture that hides the interesting half:
///
/// | machine | behind the newest | what it shows |
/// |---|---|---|
/// | `node3-4127` | 0 s | joined, and the one everything is measured against |
/// | `node7-991`  | 3 s | joined, idle, and the one that waits more than it runs |
/// | `node9-3312` | 6 s | writing and nobody's — free capacity |
/// | `node9-3319` | 5 s | the second worker on that same box |
/// | `node4-8810` | 40 min | quiet, and it keeps the name it had |
///
/// `laptop-91` is there too, with nothing but an uptime: a kernel that keeps no
/// load average is not a machine that is idle, and the screens have to say so.
pub fn sow(root: &Path, listing: Option<&Path>) -> Result<Sown, StoreError> {
    let store = soma_next_store::Local::at(root)?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or(0);

    let machines = [
        (
            0,
            reading(
                "node3-4127",
                8,
                Some(0.42),
                Some(0.62),
                1_284,
                3 * 86_400 + 4 * 3_600,
            ),
        ),
        (
            3,
            reading("node7-991", 32, Some(0.05), Some(0.18), 96, 11 * 86_400),
        ),
        (
            6,
            reading("node9-3312", 16, Some(0.88), Some(0.74), 0, 2 * 3_600),
        ),
        (
            5,
            reading("node9-3319", 16, Some(0.88), Some(0.74), 3, 2 * 3_600),
        ),
        (
            2_400,
            reading("node4-8810", 8, Some(0.31), Some(0.44), 512, 6 * 3_600),
        ),
        // Everything a `/proc` would have said is absent, and that is the point
        // of it: absent has to read as *nobody measured* and never as zero.
        (
            4,
            Machine {
                up: Duration::from_secs(1_800),
                served: 2,
                id: "laptop-91".into(),
                ..Machine::default()
            },
        ),
    ];
    for (behind, machine) in &machines {
        let said = machine.said();
        let (kind, mut meta) = said.flattened();
        meta.insert(0, ("fact".into(), kind.to_string()));
        stamped(
            root,
            &filed(&machine.id),
            &store.put(&[])?,
            meta,
            now - behind,
        )?;
    }

    let run = "3f8a1c";
    sow_a_run(root, &store, run, now)?;

    let names = match listing {
        None => 0,
        Some(at) => {
            let paper = a_listing();
            let names = paper.listed.len();
            paper.write(at).map_err(|why| {
                StoreError::Io(format!("the listing would not be written: {why}"))
            })?;
            names
        }
    };

    Ok(Sown {
        machines: machines.len(),
        names,
        run: run.to_string(),
    })
}

/// One reading, of a machine that measured itself.
fn reading(
    id: &str,
    cores: usize,
    busy: Option<f64>,
    memory: Option<f64>,
    served: u64,
    up_s: u64,
) -> Machine {
    Machine {
        up: Duration::from_secs(up_s),
        busy,
        cores: Some(cores),
        memory,
        served,
        id: id.into(),
    }
}

/// A run across two machines and this process, three `forward`s of it.
///
/// `gpu-box` is the case worth having in a fixture: twelve slices crossed to it
/// and three quarters of the round trip was waiting. The machine is at 0.05 —
/// it is not busy, it is waiting — and no per-node view can say that, because
/// neither half of the subtraction belongs to a node.
fn sow_a_run(root: &Path, store: &dyn Store, run: &str, now: u64) -> Result<(), StoreError> {
    let forwards = [
        vec![
            json!({ "fact": "ran", "node": "read", "began_us": "0", "took_us": "4000" }),
            json!({ "fact": "left", "host": "w1", "began_us": "4000", "took_us": "184000" }),
            json!({ "fact": "ran", "host": "w1", "node": "classify", "began_us": "0", "took_us": "171000" }),
            json!({ "fact": "machine", "host": "w1", "id": "node3-4127", "busy": "0.4213", "memory": "0.6187", "cores": "8", "up_us": "273600000000", "served": "1284" }),
            json!({ "fact": "left", "host": "gpu-box", "began_us": "188000", "took_us": "402000" }),
            json!({ "fact": "ran", "host": "gpu-box", "node": "embed", "began_us": "0", "took_us": "96000" }),
            json!({ "fact": "machine", "host": "gpu-box", "id": "node7-991", "busy": "0.0512", "memory": "0.1804", "cores": "32", "up_us": "950400000000", "served": "96" }),
            // `w2` was in this run and has since gone quiet. It is the case the
            // screen has to get right: what somebody needs to see about a
            // machine that stopped is **which** machine stopped.
            json!({ "fact": "left", "host": "w2", "began_us": "590000", "took_us": "22000" }),
            json!({ "fact": "ran", "host": "w2", "node": "score", "began_us": "0", "took_us": "19000" }),
            json!({ "fact": "machine", "host": "w2", "id": "node4-8810", "busy": "0.3102", "memory": "0.4401", "cores": "8", "up_us": "21600000000", "served": "512" }),
            json!({ "fact": "finished", "took_us": "616000" }),
        ],
        vec![
            json!({ "fact": "ran", "node": "read", "began_us": "0", "took_us": "3000" }),
            json!({ "fact": "left", "host": "w1", "began_us": "3000", "took_us": "179000" }),
            json!({ "fact": "ran", "host": "w1", "node": "classify", "began_us": "0", "took_us": "168000" }),
            json!({ "fact": "finished", "took_us": "185000" }),
        ],
        vec![
            json!({ "fact": "ran", "node": "read", "began_us": "0", "took_us": "3000" }),
            json!({ "fact": "left", "host": "gpu-box", "began_us": "3000", "took_us": "398000" }),
            json!({ "fact": "ran", "host": "gpu-box", "node": "embed", "began_us": "0", "took_us": "94000" }),
            json!({ "fact": "machine", "host": "gpu-box", "id": "node7-991", "busy": "0.0498", "memory": "0.1811", "cores": "32", "up_us": "950500000000", "served": "97" }),
            json!({ "fact": "finished", "took_us": "402000" }),
        ],
    ];

    for (which, facts) in forwards.iter().enumerate() {
        let blob = serde_json::to_vec_pretty(&facts)
            .map_err(|why| StoreError::Corrupt(format!("those facts will not write: {why}")))?;
        let digest = store.put(&blob)?;
        let took: u64 = facts
            .iter()
            .filter(|fact| fact["fact"] == "finished")
            .filter_map(|fact| fact["took_us"].as_str()?.parse::<u64>().ok())
            .sum();
        let meta = vec![
            ("run".to_string(), run.to_string()),
            ("forward".to_string(), which.to_string()),
            ("state".to_string(), "ok".to_string()),
            (
                "nodes".to_string(),
                facts
                    .iter()
                    .filter(|fact| fact["fact"] == "ran")
                    .count()
                    .to_string(),
            ),
            ("took_us".to_string(), took.to_string()),
        ];
        // Newest last, so a reader taking the last N takes the last N.
        stamped(
            root,
            &format!("run/{run}/{which}"),
            &digest,
            meta,
            now - (forwards.len() - which) as u64,
        )?;
    }
    Ok(())
}

/// A listing with the two cases that are not obvious in it.
fn a_listing() -> Listing {
    Listing {
        listed: vec![
            Listed::at("w1", "node3:7000"),
            // The same address under a second name: one wire, one catalog, and
            // packed once. Getting this wrong is not an extra socket, it is a
            // run that quietly loses its state.
            Listed::at("principal", "node3:7000"),
            Listed::at("gpu-box", "node7:7000"),
            Listed::at("w2", "node4:7000"),
            // A command is not an identity: running it twice gives two of them,
            // so this is never grouped with anything.
            Listed::run("tok", ["python", "-m", "soma_next.worker"]),
        ],
    }
}

/// Binds a name with a stamp of our choosing, at the layout the store
/// documents, through the store's own public record type.
fn stamped(
    root: &Path,
    name: &str,
    digest: &Digest,
    meta: Meta,
    when: u64,
) -> Result<(), StoreError> {
    let bound = Bound {
        name: name.to_string(),
        digest: digest.clone(),
        meta,
        when,
    };
    let (head, rest) = Digest::of(name.as_bytes()).path();
    let at = root.join("names").join(head);
    fs::create_dir_all(&at).map_err(|why| StoreError::Io(why.to_string()))?;
    let bytes = serde_json::to_vec_pretty(&bound)
        .map_err(|why| StoreError::Corrupt(format!("that record cannot be written: {why}")))?;
    fs::write(at.join(rest), bytes).map_err(|why| StoreError::Io(why.to_string()))
}
