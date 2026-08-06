//! Append-only JSONL event sink.

use chrono::Utc;
use somatize_core::tracking::event::Event;
use somatize_core::tracking::{EventEnvelope, EventSink};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Writes every event as one JSON line to `events.jsonl`, teeing
/// metric-bearing events (`TrialMetric`, `MetricReported`) into a flat
/// `metrics.jsonl` for cheap time-series reads.
///
/// Writes are buffered and flushed every `flush_every` events (and on
/// [`EventSink::flush`]/drop). I/O errors are logged and swallowed —
/// tracking must never take down a training run.
pub struct JsonlEventSink {
    events: Mutex<BufWriter<File>>,
    metrics: Option<Mutex<BufWriter<File>>>,
    seq: AtomicU64,
    flush_every: u64,
}

impl JsonlEventSink {
    /// Create fresh log files (truncating any existing ones).
    pub fn create(
        events_path: &Path,
        metrics_path: Option<&Path>,
        flush_every: usize,
    ) -> std::io::Result<Self> {
        Self::open(events_path, metrics_path, flush_every, 0, false)
    }

    /// Open existing logs in append mode, continuing at `start_seq`.
    pub fn append(
        events_path: &Path,
        metrics_path: Option<&Path>,
        flush_every: usize,
        start_seq: u64,
    ) -> std::io::Result<Self> {
        Self::open(events_path, metrics_path, flush_every, start_seq, true)
    }

    fn open(
        events_path: &Path,
        metrics_path: Option<&Path>,
        flush_every: usize,
        start_seq: u64,
        append: bool,
    ) -> std::io::Result<Self> {
        let open = |path: &Path| -> std::io::Result<BufWriter<File>> {
            let mut opts = OpenOptions::new();
            opts.create(true).write(true);
            if append {
                opts.append(true);
            } else {
                opts.truncate(true);
            }
            Ok(BufWriter::new(opts.open(path)?))
        };
        Ok(Self {
            events: Mutex::new(open(events_path)?),
            metrics: metrics_path.map(|p| open(p).map(Mutex::new)).transpose()?,
            seq: AtomicU64::new(start_seq),
            flush_every: flush_every.max(1) as u64,
        })
    }

    /// Next sequence number to be assigned (== events recorded so far
    /// when starting from zero).
    pub fn next_seq(&self) -> u64 {
        self.seq.load(Ordering::SeqCst)
    }

    /// Flat metric line for `metrics.jsonl`, if the event carries one.
    fn metric_line(event: &Event) -> Option<serde_json::Value> {
        match event {
            Event::TrialMetric {
                study_id: _,
                trial_id,
                metric,
            } => Some(serde_json::json!({
                "ts": metric.timestamp,
                "name": metric.name,
                "value": metric.value,
                "step": metric.step,
                "trial_id": trial_id,
                "node_id": null,
            })),
            Event::MetricReported {
                run_id: _,
                metric,
                node_id,
                trial_id,
            } => Some(serde_json::json!({
                "ts": metric.timestamp,
                "name": metric.name,
                "value": metric.value,
                "step": metric.step,
                "trial_id": trial_id,
                "node_id": node_id,
            })),
            _ => None,
        }
    }

    fn write_line(writer: &Mutex<BufWriter<File>>, line: &str, do_flush: bool) {
        let mut guard = match writer.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Err(e) = writeln!(guard, "{line}") {
            tracing::warn!("tracking: failed to write event line: {e}");
            return;
        }
        if do_flush && let Err(e) = guard.flush() {
            tracing::warn!("tracking: failed to flush event log: {e}");
        }
    }
}

impl EventSink for JsonlEventSink {
    fn record(&self, event: &Event) {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let envelope = EventEnvelope {
            seq,
            ts: Utc::now(),
            event: event.clone(),
        };
        let do_flush = seq % self.flush_every == self.flush_every - 1;
        match serde_json::to_string(&envelope) {
            Ok(line) => Self::write_line(&self.events, &line, do_flush),
            Err(e) => tracing::warn!("tracking: failed to serialize event: {e}"),
        }
        if let Some(metrics) = &self.metrics
            && let Some(line) = Self::metric_line(event)
        {
            Self::write_line(metrics, &line.to_string(), do_flush);
        }
    }

    fn flush(&self) {
        for writer in std::iter::once(&self.events).chain(self.metrics.iter()) {
            let mut guard = match writer.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            if let Err(e) = guard.flush() {
                tracing::warn!("tracking: failed to flush event log: {e}");
            }
        }
    }
}

impl Drop for JsonlEventSink {
    fn drop(&mut self) {
        self.flush();
    }
}
