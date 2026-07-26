//! File-backed knowledge base: append-only JSONL persistence.
//!
//! Wraps [`MemoryKnowledgeBase`] with a durable log: on open, existing
//! records are loaded from the file (tolerating a torn trailing line
//! from a crash mid-write); every [`record`](KnowledgeBase::record)
//! appends one JSON line and delegates. Queries delegate unchanged.

use crate::knowledge_base::{KnowledgeBase, MemoryKnowledgeBase};
use crate::record::{ChangePoint, ExperimentRecord, ResearchLine};
use somatize_core::error::{Result, SomaError};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// JSONL-backed [`KnowledgeBase`] (one [`ExperimentRecord`] per line).
///
/// Default location: `.soma/experiments.jsonl`.
pub struct FileKnowledgeBase {
    inner: MemoryKnowledgeBase,
    path: PathBuf,
}

impl FileKnowledgeBase {
    /// Open (or create) the knowledge base at `path`, loading every
    /// parseable line. A corrupt trailing line — the signature of a
    /// crash mid-append — is skipped with a warning; a corrupt line in
    /// the middle of the file is an error.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        let mut inner = MemoryKnowledgeBase::new();
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
            for (i, line) in lines.iter().enumerate() {
                match serde_json::from_str::<ExperimentRecord>(line) {
                    Ok(record) => inner.record(record)?,
                    Err(e) if i == lines.len() - 1 => {
                        tracing::warn!(
                            "knowledge base: skipping corrupt trailing line in {}: {e}",
                            path.display()
                        );
                    }
                    Err(e) => {
                        return Err(SomaError::Serialization(format!(
                            "corrupt experiment record at {}:{}: {e}",
                            path.display(),
                            i + 1
                        )));
                    }
                }
            }
        }
        Ok(Self { inner, path })
    }

    /// Path of the backing JSONL file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl KnowledgeBase for FileKnowledgeBase {
    fn record(&mut self, experiment: ExperimentRecord) -> Result<()> {
        let line = serde_json::to_string(&experiment)
            .map_err(|e| SomaError::Serialization(e.to_string()))?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{line}")?;
        self.inner.record(experiment)
    }

    fn get(&self, id: &str) -> Result<Option<&ExperimentRecord>> {
        self.inner.get(id)
    }

    fn search(&self, query: &str, max_results: usize) -> Result<Vec<&ExperimentRecord>> {
        self.inner.search(query, max_results)
    }

    fn experiments_in_line(&self, line: &str) -> Result<Vec<&ExperimentRecord>> {
        self.inner.experiments_in_line(line)
    }

    fn research_lines(&self) -> Result<Vec<ResearchLine>> {
        self.inner.research_lines()
    }

    fn promising_lines(&self, metric: &str) -> Result<Vec<ResearchLine>> {
        self.inner.promising_lines(metric)
    }

    fn trajectory(&self, line: &str, metric: &str) -> Result<Vec<(String, f64)>> {
        self.inner.trajectory(line, metric)
    }

    fn change_points(&self, line: &str, metric: &str, threshold: f64) -> Result<Vec<ChangePoint>> {
        self.inner.change_points(line, metric, threshold)
    }

    fn children(&self, experiment_id: &str) -> Result<Vec<&ExperimentRecord>> {
        self.inner.children(experiment_id)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn record(id: &str, f1: f64) -> ExperimentRecord {
        ExperimentRecord::new(id, format!("experiment {id}"))
            .with_research_line("mos")
            .with_metrics(HashMap::from([("f1".to_string(), f1)]))
    }

    #[test]
    fn open_record_reopen_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");

        {
            let mut kb = FileKnowledgeBase::open(&path).unwrap();
            kb.record(record("e1", 0.8)).unwrap();
            kb.record(record("e2", 0.9)).unwrap();
            assert_eq!(kb.len(), 2);
        }

        let kb = FileKnowledgeBase::open(&path).unwrap();
        assert_eq!(kb.len(), 2);
        assert!(kb.get("e1").unwrap().is_some());
        assert_eq!(kb.experiments_in_line("mos").unwrap().len(), 2);
    }

    #[test]
    fn corrupt_trailing_line_is_tolerated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        {
            let mut kb = FileKnowledgeBase::open(&path).unwrap();
            kb.record(record("e1", 0.8)).unwrap();
        }
        // Simulate a crash mid-append.
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        write!(file, "{{\"id\": \"e2\", \"name\": tru").unwrap();
        drop(file);

        let kb = FileKnowledgeBase::open(&path).unwrap();
        assert_eq!(kb.len(), 1);
    }

    #[test]
    fn corrupt_middle_line_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        {
            let mut kb = FileKnowledgeBase::open(&path).unwrap();
            kb.record(record("e1", 0.8)).unwrap();
        }
        let content = fs::read_to_string(&path).unwrap();
        fs::write(&path, format!("not json\n{content}")).unwrap();
        assert!(FileKnowledgeBase::open(&path).is_err());
    }

    #[test]
    fn creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".soma").join("experiments.jsonl");
        let mut kb = FileKnowledgeBase::open(&path).unwrap();
        kb.record(record("e1", 0.5)).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn empty_file_yields_empty_kb() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        fs::write(&path, "").unwrap();
        let kb = FileKnowledgeBase::open(&path).unwrap();
        assert!(kb.is_empty());
        assert_eq!(kb.path(), path.as_path());
    }

    #[test]
    fn blank_lines_between_records_are_tolerated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        {
            let mut kb = FileKnowledgeBase::open(&path).unwrap();
            kb.record(record("e1", 0.8)).unwrap();
            kb.record(record("e2", 0.9)).unwrap();
        }
        let content = fs::read_to_string(&path).unwrap();
        let padded = content.replace('\n', "\n\n");
        fs::write(&path, format!("\n{padded}")).unwrap();

        let kb = FileKnowledgeBase::open(&path).unwrap();
        assert_eq!(kb.len(), 2);
    }

    /// CONTRACT (pinned): a file whose ONLY line is corrupt is
    /// classified as a torn tail — it opens as an empty KB with a
    /// warning rather than erroring. Loud failure requires at least
    /// one valid record before the corruption.
    #[test]
    fn single_fully_corrupt_line_opens_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        fs::write(&path, "{definitely not json\n").unwrap();
        let kb = FileKnowledgeBase::open(&path).unwrap();
        assert!(kb.is_empty());
    }

    #[test]
    fn unicode_content_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        {
            let mut kb = FileKnowledgeBase::open(&path).unwrap();
            let rec = ExperimentRecord::new("π-experimento", "atención emoción 🧠")
                .with_research_line("línea-ñ")
                .with_notes("multi\nline\nnotes stay one JSONL line");
            kb.record(rec).unwrap();
        }
        let kb = FileKnowledgeBase::open(&path).unwrap();
        assert_eq!(kb.len(), 1);
        let rec = kb.get("π-experimento").unwrap().unwrap();
        assert_eq!(rec.name, "atención emoción 🧠");
        assert_eq!(rec.research_line.as_deref(), Some("línea-ñ"));
        // Embedded newlines are JSON-escaped: still one record per line.
        let lines = fs::read_to_string(&path).unwrap().lines().count();
        assert_eq!(lines, 1);
    }

    #[test]
    fn two_handles_share_the_append_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        let mut a = FileKnowledgeBase::open(&path).unwrap();
        let mut b = FileKnowledgeBase::open(&path).unwrap();

        a.record(record("from_a", 0.1)).unwrap();
        b.record(record("from_b", 0.2)).unwrap();

        // In-memory views don't see each other's writes…
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        // …but the log has both, and a fresh open sees both.
        let c = FileKnowledgeBase::open(&path).unwrap();
        assert_eq!(c.len(), 2);
        assert!(c.get("from_a").unwrap().is_some());
        assert!(c.get("from_b").unwrap().is_some());
    }

    #[cfg(unix)]
    #[test]
    fn record_io_failure_leaves_memory_untouched() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        let mut kb = FileKnowledgeBase::open(&path).unwrap();
        kb.record(record("e1", 0.5)).unwrap();

        // Make the file unwritable: the append fails BEFORE the
        // in-memory mutation, so the KB stays consistent with disk.
        fs::set_permissions(&path, fs::Permissions::from_mode(0o444)).unwrap();
        assert!(kb.record(record("e2", 0.9)).is_err());
        assert_eq!(kb.len(), 1);
        assert!(kb.get("e2").unwrap().is_none());
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    }

    #[test]
    fn queries_delegate_over_rehydrated_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        {
            let mut kb = FileKnowledgeBase::open(&path).unwrap();
            kb.record(record("e1", 0.6).with_parent("e0")).unwrap();
            kb.record(record("e0", 0.5)).unwrap();
            kb.record(record("e2", 0.8).with_parent("e0")).unwrap();
        }
        // Everything below runs over records loaded from disk.
        let kb = FileKnowledgeBase::open(&path).unwrap();
        assert_eq!(kb.experiments_in_line("mos").unwrap().len(), 3);
        assert_eq!(kb.children("e0").unwrap().len(), 2);
        assert!(!kb.search("experiment", 10).unwrap().is_empty());
        let lines = kb.research_lines().unwrap();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].name, "mos");
        let trajectory = kb.trajectory("mos", "f1").unwrap();
        assert_eq!(trajectory.len(), 3);
        assert!(kb.promising_lines("f1").is_ok());
        assert!(kb.change_points("mos", "f1", 0.5).is_ok());
    }
}
