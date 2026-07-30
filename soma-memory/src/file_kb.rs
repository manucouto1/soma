//! File-backed knowledge base: append-only JSONL persistence.
//!
//! Wraps [`MemoryKnowledgeBase`] with a durable log: on open, existing
//! records are loaded from the file (tolerating a torn trailing line
//! from a crash mid-write); every [`record`](KnowledgeBase::record)
//! appends one JSON line and delegates. Queries delegate unchanged.
//!
//! Because the log is strictly append-only, [`refresh`] can pick up
//! another process's writes by reading from a byte offset instead of
//! re-parsing the file: a long-lived reader (the MCP server) sees runs
//! finishing in another terminal without reopening anything.
//!
//! [`refresh`]: KnowledgeBase::refresh

use crate::knowledge_base::{KnowledgeBase, MemoryKnowledgeBase};
use crate::record::ExperimentRecord;
use somatize_core::error::{Result, SomaError};
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// JSONL-backed [`KnowledgeBase`] (one [`ExperimentRecord`] per line).
///
/// Default location: `.soma/experiments.jsonl`.
pub struct FileKnowledgeBase {
    inner: MemoryKnowledgeBase,
    path: PathBuf,
    /// Bytes of the log already folded into `inner`. Only ever advanced
    /// past a complete, newline-terminated region.
    offset: u64,
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

        let mut kb = Self {
            inner: MemoryKnowledgeBase::new(),
            path,
            offset: 0,
        };
        kb.load_from_offset(true)?;
        Ok(kb)
    }

    /// Path of the backing JSONL file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read everything after `self.offset` and fold it in.
    ///
    /// `strict` distinguishes the two callers: on `open`, a corrupt
    /// line that is not the last one is a real error worth surfacing;
    /// on `refresh`, the tail is expected to be racing a writer, so an
    /// unparseable final line is simply left for next time.
    fn load_from_offset(&mut self, strict: bool) -> Result<usize> {
        if !self.path.exists() {
            return Ok(0);
        }
        let mut file = fs::File::open(&self.path)?;
        let size = file.metadata()?.len();
        if size < self.offset {
            // Truncated or replaced underneath us (a `kb reindex` in
            // another process): start over rather than read garbage.
            self.inner = MemoryKnowledgeBase::new();
            self.offset = 0;
        }
        if size == self.offset {
            return Ok(0);
        }
        file.seek(SeekFrom::Start(self.offset))?;
        let mut chunk = String::new();
        file.read_to_string(&mut chunk)?;

        // Only consume up to the last newline: a partially written
        // final line stays unread, and its bytes are re-read next time.
        let complete = match chunk.rfind('\n') {
            Some(i) => &chunk[..=i],
            None => "",
        };
        let leftover = &chunk[complete.len()..];
        if !leftover.trim().is_empty() {
            tracing::warn!(
                "knowledge base: {} has an unterminated trailing line; deferring it",
                self.path.display()
            );
        }

        let lines: Vec<&str> = complete.lines().filter(|l| !l.trim().is_empty()).collect();
        let mut loaded = 0;
        for (i, line) in lines.iter().enumerate() {
            match serde_json::from_str::<ExperimentRecord>(line) {
                Ok(record) => {
                    self.inner.record(record)?;
                    loaded += 1;
                }
                Err(e) if !strict || i == lines.len() - 1 => {
                    tracing::warn!(
                        "knowledge base: skipping corrupt line in {}: {e}",
                        self.path.display()
                    );
                }
                Err(e) => {
                    return Err(SomaError::Serialization(format!(
                        "corrupt experiment record at {}:{}: {e}",
                        self.path.display(),
                        i + 1
                    )));
                }
            }
        }
        self.offset += complete.len() as u64;
        Ok(loaded)
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
        drop(file);
        // Fold the log back in rather than pushing `experiment`
        // straight into memory. The log is then the single source of
        // what this handle holds: our own line and anything another
        // process appended since we last looked both land exactly
        // once, and the offset can never drift.
        self.load_from_offset(false)?;
        Ok(())
    }

    fn all(&self) -> Result<Vec<ExperimentRecord>> {
        self.inner.all()
    }

    fn get(&self, id: &str) -> Result<Option<ExperimentRecord>> {
        self.inner.get(id)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn refresh(&mut self) -> Result<usize> {
        self.load_from_offset(false)
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
        // `a` has not looked at the log since, so it still sees one…
        assert_eq!(a.len(), 1);

        // …while `b` reads the log to append, so it lands with both.
        b.record(record("from_b", 0.2)).unwrap();
        assert_eq!(b.len(), 2);

        // A stale handle catches up on demand, exactly once.
        assert_eq!(a.refresh().unwrap(), 1);
        assert_eq!(a.len(), 2);
        assert_eq!(a.refresh().unwrap(), 0);
        assert_eq!(a.len(), 2);

        let c = FileKnowledgeBase::open(&path).unwrap();
        assert_eq!(c.len(), 2);
        assert!(c.get("from_a").unwrap().is_some());
        assert!(c.get("from_b").unwrap().is_some());
    }

    #[test]
    fn refresh_sees_a_line_appended_by_another_process() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        let mut reader = FileKnowledgeBase::open(&path).unwrap();
        assert!(reader.is_empty());

        // Another process finishes a run mid-session.
        let mut writer = FileKnowledgeBase::open(&path).unwrap();
        writer.record(record("appeared", 0.9)).unwrap();

        assert!(reader.get("appeared").unwrap().is_none(), "not yet visible");
        assert_eq!(reader.refresh().unwrap(), 1);
        assert!(reader.get("appeared").unwrap().is_some());
    }

    #[test]
    fn refresh_defers_a_half_written_line_until_it_is_complete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        let mut reader = FileKnowledgeBase::open(&path).unwrap();

        let complete = serde_json::to_string(&record("done", 0.5)).unwrap();
        let partial = serde_json::to_string(&record("torn", 0.6)).unwrap();
        let torn_prefix = &partial[..partial.len() / 2];
        fs::write(&path, format!("{complete}\n{torn_prefix}")).unwrap();

        assert_eq!(reader.refresh().unwrap(), 1);
        assert!(reader.get("done").unwrap().is_some());
        assert!(reader.get("torn").unwrap().is_none());

        // The writer finishes its line; the deferred bytes are re-read.
        fs::write(&path, format!("{complete}\n{partial}\n")).unwrap();
        assert_eq!(reader.refresh().unwrap(), 1);
        assert!(reader.get("torn").unwrap().is_some());
        assert_eq!(reader.len(), 2);
    }

    #[test]
    fn refresh_recovers_from_the_log_being_rewritten_shorter() {
        // `soma kb reindex` replaces the journal wholesale. A reader
        // holding a byte offset past the new end must not read garbage.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiments.jsonl");
        let mut reader = FileKnowledgeBase::open(&path).unwrap();
        {
            let mut writer = FileKnowledgeBase::open(&path).unwrap();
            for id in ["e1", "e2", "e3"] {
                writer.record(record(id, 0.5)).unwrap();
            }
        }
        reader.refresh().unwrap();
        assert_eq!(reader.len(), 3);

        let single = serde_json::to_string(&record("only", 0.1)).unwrap();
        fs::write(&path, format!("{single}\n")).unwrap();
        reader.refresh().unwrap();
        assert_eq!(reader.len(), 1);
        assert!(reader.get("only").unwrap().is_some());
        assert!(reader.get("e1").unwrap().is_none());
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
