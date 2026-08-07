//! File-based tracking backend: run directories under `.soma/runs/`.
//!
//! [`JsonlEventSink`] is the lossless event consumer wired into
//! [`EventBus`](crate::tracking::event_bus::EventBus); [`LocalTracker`] owns one
//! run directory (manifest, status, logs). See
//! `docs/src/content/docs/design/tracking.md` for the on-disk layout.

pub mod event_bus;

mod head;
mod jsonl_sink;
mod local_tracker;
mod reader;
mod summary;

pub use head::{
    PARENT_ENV, advance_head, checkout, clear_head, head_path, read_head, resolve_parent,
    resolve_parent_from, run_exists, write_head,
};
pub use jsonl_sink::JsonlEventSink;
pub use local_tracker::{LocalTracker, collect_git_info, load_manifest, load_status};
pub use reader::{
    AgentNodeActivity, AgenticActivity, CacheActivity, EffectSpan, HealthFlagRecord, MetricPoint,
    NodeCacheCounts, NodeSpan, RunInfo, RunReader, STALE_HEARTBEAT_SECS, TrialSpan, list_runs,
};
pub use summary::summarize;
