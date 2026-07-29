//! File-based tracking backend: run directories under `.soma/runs/`.
//!
//! [`JsonlEventSink`] is the lossless event consumer wired into
//! [`EventBus`](crate::event_bus::EventBus); [`LocalTracker`] owns one
//! run directory (manifest, status, logs). See
//! `docs/src/content/docs/design/tracking.md` for the on-disk layout.

mod jsonl_sink;
mod local_tracker;
mod reader;

pub use jsonl_sink::JsonlEventSink;
pub use local_tracker::{LocalTracker, collect_git_info, load_manifest, load_status};
pub use reader::{
    CacheActivity, HealthFlagRecord, MetricPoint, NodeCacheCounts, NodeSpan, RunInfo, RunReader,
    STALE_HEARTBEAT_SECS, TrialSpan, list_runs,
};
