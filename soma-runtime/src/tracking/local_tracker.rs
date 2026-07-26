//! Local run-directory tracker.

use super::JsonlEventSink;
use chrono::Utc;
use somatize_core::error::{Result, SomaError};
use somatize_core::study::Study;
use somatize_core::tracking::{
    EventSink, GitInfo, RunKind, RunManifest, RunState, RunStatus, Tracker,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

const EVENTS_FILE: &str = "events.jsonl";
const METRICS_FILE: &str = "metrics.jsonl";
const MANIFEST_FILE: &str = "manifest.json";
const STATUS_FILE: &str = "status.json";
const STUDY_FILE: &str = "study.json";
const FLUSH_EVERY: usize = 20;

/// A run directory under `<root>/runs/<run_id>/`.
///
/// `create` starts a new run (manifest + running status + fresh logs);
/// `open` re-attaches to an existing directory for resume, continuing
/// the event sequence where it left off.
pub struct LocalTracker {
    run_id: String,
    dir: PathBuf,
    sink: Arc<JsonlEventSink>,
}

impl LocalTracker {
    /// Create a new run directory under `root` (e.g. `.soma`) and write
    /// its manifest (environment fields best-effort) and running status.
    pub fn create(root: impl AsRef<Path>, kind: RunKind, name: &str) -> Result<Self> {
        let run_id = new_run_id(kind);
        let dir = root.as_ref().join("runs").join(&run_id);
        fs::create_dir_all(&dir)?;

        let mut manifest = RunManifest::new(&run_id, kind, name);
        manifest.soma_version = Some(env!("CARGO_PKG_VERSION").to_string());
        manifest.hostname = hostname();
        manifest.git = collect_git_info(Path::new("."));
        manifest.argv = std::env::args().collect();
        manifest.entrypoint = manifest.argv.first().cloned();
        manifest.cwd = std::env::current_dir()
            .ok()
            .map(|p| p.display().to_string());
        if kind == RunKind::Study {
            manifest.study_path = Some(STUDY_FILE.to_string());
        }
        atomic_write_json(&dir.join(MANIFEST_FILE), &manifest)?;
        atomic_write_json(&dir.join(STATUS_FILE), &RunStatus::running())?;

        let sink = JsonlEventSink::create(
            &dir.join(EVENTS_FILE),
            Some(&dir.join(METRICS_FILE)),
            FLUSH_EVERY,
        )?;
        Ok(Self {
            run_id,
            dir,
            sink: Arc::new(sink),
        })
    }

    /// Re-attach to an existing run directory (resume). The event
    /// sequence continues after the last recorded line, and the status
    /// flips back to running.
    pub fn open(run_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = run_dir.as_ref().to_path_buf();
        let manifest = load_manifest(&dir)?;
        let start_seq = repair_and_count_lines(&dir.join(EVENTS_FILE))?;
        let sink = JsonlEventSink::append(
            &dir.join(EVENTS_FILE),
            Some(&dir.join(METRICS_FILE)),
            FLUSH_EVERY,
            start_seq,
        )?;
        atomic_write_json(&dir.join(STATUS_FILE), &RunStatus::running())?;
        Ok(Self {
            run_id: manifest.run_id,
            dir,
            sink: Arc::new(sink),
        })
    }
}

impl Tracker for LocalTracker {
    fn run_id(&self) -> &str {
        &self.run_id
    }

    fn run_dir(&self) -> &Path {
        &self.dir
    }

    fn sink(&self) -> Arc<dyn EventSink> {
        self.sink.clone()
    }

    fn save_manifest(&self, manifest: &RunManifest) -> Result<()> {
        atomic_write_json(&self.dir.join(MANIFEST_FILE), manifest)
    }

    fn save_artifact(&self, rel_path: &str, bytes: &[u8]) -> Result<()> {
        let path = self.dir.join(rel_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, bytes)?;
        Ok(())
    }

    fn save_study(&self, study: &Study) -> Result<()> {
        atomic_write_json(&self.dir.join(STUDY_FILE), study)
    }

    fn heartbeat(&self) -> Result<()> {
        let mut status = load_status(&self.dir)?;
        let now = Utc::now();
        status.heartbeat_at = Some(now);
        status.updated_at = now;
        atomic_write_json(&self.dir.join(STATUS_FILE), &status)
    }

    fn finalize(&self, state: RunState) -> Result<()> {
        self.sink.flush();
        let now = Utc::now();
        let status = RunStatus {
            state,
            updated_at: now,
            heartbeat_at: Some(now),
            finished_at: Some(now),
        };
        atomic_write_json(&self.dir.join(STATUS_FILE), &status)
    }
}

/// Read a run's manifest.
pub fn load_manifest(run_dir: &Path) -> Result<RunManifest> {
    let bytes = fs::read(run_dir.join(MANIFEST_FILE))?;
    serde_json::from_slice(&bytes).map_err(|e| SomaError::Serialization(e.to_string()))
}

/// Read a run's status file.
pub fn load_status(run_dir: &Path) -> Result<RunStatus> {
    let bytes = fs::read(run_dir.join(STATUS_FILE))?;
    serde_json::from_slice(&bytes).map_err(|e| SomaError::Serialization(e.to_string()))
}

/// Best-effort git context via subprocess; all-`None` outside a repo.
pub fn collect_git_info(dir: &Path) -> GitInfo {
    let run = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        (!s.is_empty()).then_some(s)
    };
    GitInfo {
        sha: run(&["rev-parse", "HEAD"]),
        branch: run(&["rev-parse", "--abbrev-ref", "HEAD"]),
        dirty: run(&["status", "--porcelain"]).map(|s| !s.is_empty()).or({
            // `git status` succeeded with empty output → clean repo.
            run(&["rev-parse", "HEAD"]).map(|_| false)
        }),
    }
}

fn hostname() -> Option<String> {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|h| !h.is_empty())
        .or_else(|| {
            fs::read_to_string("/etc/hostname")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|h| !h.is_empty())
        })
        .or_else(|| {
            let out = Command::new("hostname").output().ok()?;
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            (!s.is_empty()).then_some(s)
        })
}

/// Sortable, collision-resistant run id: `<prefix>_<utc-compact>_<4hex>`.
fn new_run_id(kind: RunKind) -> String {
    let prefix = match kind {
        RunKind::Study => "study",
        RunKind::Trial => "trial",
        _ => "run",
    };
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "{prefix}_{}_{:04x}",
        Utc::now().format("%Y%m%dT%H%M%S"),
        (nanos & 0xffff) as u16
    )
}

/// Write JSON via tmp file + rename so readers never observe a partial
/// file and a crash mid-write preserves the previous version.
fn atomic_write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let json =
        serde_json::to_vec_pretty(value).map_err(|e| SomaError::Serialization(e.to_string()))?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, &json)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Prepare the events log for appending: a crash mid-write leaves a
/// torn trailing line with no `\n`; appending after it would concatenate
/// the next event onto garbage. Truncate the torn tail (that event was
/// never durably recorded) and return the number of complete lines —
/// the next sequence number.
fn repair_and_count_lines(path: &Path) -> Result<u64> {
    let content = match fs::read(path) {
        Ok(c) => c,
        Err(_) => return Ok(0), // no events yet — fresh log
    };
    let newlines = content.iter().filter(|b| **b == b'\n').count() as u64;
    if content.is_empty() || content.last() == Some(&b'\n') {
        return Ok(newlines);
    }
    let keep = content
        .iter()
        .rposition(|b| *b == b'\n')
        .map(|i| i + 1)
        .unwrap_or(0);
    let file = fs::OpenOptions::new().write(true).open(path)?;
    file.set_len(keep as u64)?;
    Ok(newlines)
}
