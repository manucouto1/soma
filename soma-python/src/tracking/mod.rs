//! Reading a run back.
//!
//! [`run`] is `Run` — the handle `begin_run` returns, which owns the run
//! directory while the run is alive and is the single writer of
//! `graph.json`, `graph.mmd` and `fingerprint.json`.
//!
//! [`readers`] is the other direction: free functions over a finished run
//! directory and over the experiment pool, each returning JSON for the pure
//! Python layer to shape into `RunView`. That there are a dozen of them,
//! each with exactly one caller and each re-parsing `events.jsonl`, is D-63.

pub(crate) mod readers;
pub(crate) mod run;
