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
}
