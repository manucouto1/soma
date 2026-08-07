//! `.soma/HEAD` — which run the next one descends from.
//!
//! An experiment pool is only as useful as its edges, and an edge needs
//! a parent. Soma resolves one in four steps, most explicit first:
//!
//! 1. the `parent=` argument the caller passed,
//! 2. `$SOMA_PARENT_RUN` (for schedulers and CI that fan work out),
//! 3. `.soma/HEAD`, advanced automatically by the last **successful**
//!    run in this root,
//! 4. nothing — this run starts a new line.
//!
//! HEAD advances only on success: a crashed attempt must not become the
//! parent of everything that follows. To branch off an older run,
//! rewind with `soma.checkout(run_id)`.
//!
//! **Inferring the parent from timestamps is deliberately not done.**
//! "The run before this one" is not the same claim as "the run this one
//! was derived from", and a single false edge poisons every delta
//! computed downstream of it. An absent parent is recoverable; a wrong
//! one is not.

use somatize_core::error::{Result, SomaError};
use std::fs;
use std::path::{Path, PathBuf};

/// Environment override for the parent run id.
pub const PARENT_ENV: &str = "SOMA_PARENT_RUN";

/// Path of the HEAD file for a tracking root (`.soma/HEAD`).
pub fn head_path(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join("HEAD")
}

/// The run id in `.soma/HEAD`, if any. An unreadable, empty or
/// whitespace-only HEAD reads as absent — never as an error, because a
/// broken pointer must not stop a run from starting.
pub fn read_head(root: impl AsRef<Path>) -> Option<String> {
    let text = fs::read_to_string(head_path(root)).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Point HEAD at `run_id`, atomically, so a crash mid-write leaves the
/// previous pointer intact rather than a truncated one.
pub fn write_head(root: impl AsRef<Path>, run_id: &str) -> Result<()> {
    let root = root.as_ref();
    fs::create_dir_all(root)?;
    crate::fsutil::write_atomic(&head_path(root), format!("{run_id}\n").as_bytes())
}

/// Detach HEAD: the next run starts a new line. Absent HEAD is fine.
pub fn clear_head(root: impl AsRef<Path>) -> Result<()> {
    match fs::remove_file(head_path(root)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(SomaError::Io(e)),
    }
}

/// Whether `<root>/runs/<run_id>/manifest.json` exists.
pub fn run_exists(root: impl AsRef<Path>, run_id: &str) -> bool {
    root.as_ref()
        .join("runs")
        .join(run_id)
        .join("manifest.json")
        .exists()
}

/// Point HEAD at an existing run so the next run branches from it.
///
/// Errors when the run is unknown to this root: silently accepting a
/// typo would attach the next experiment to a parent that does not
/// exist, which is exactly the false edge this module refuses to make.
pub fn checkout(root: impl AsRef<Path>, run_id: &str) -> Result<()> {
    let root = root.as_ref();
    if !run_exists(root, run_id) {
        return Err(SomaError::Other(format!(
            "no run '{run_id}' under {}/runs — checkout needs a run that exists",
            root.display()
        )));
    }
    write_head(root, run_id)
}

/// Resolve the parent run for a run about to start.
///
/// See the module docs for the precedence. Reads the environment and
/// the filesystem; [`resolve_parent_from`] is the pure core.
pub fn resolve_parent(root: impl AsRef<Path>, explicit: Option<&str>) -> Option<String> {
    let root = root.as_ref();
    let env = std::env::var(PARENT_ENV).ok();
    resolve_parent_from(explicit, env.as_deref(), || read_head(root))
}

/// The precedence rule, with its inputs injected — the testable core.
///
/// `head` is a closure so the file is only read when the earlier, more
/// explicit sources came up empty.
pub fn resolve_parent_from(
    explicit: Option<&str>,
    env: Option<&str>,
    head: impl FnOnce() -> Option<String>,
) -> Option<String> {
    let non_empty = |s: &str| {
        let s = s.trim();
        (!s.is_empty()).then(|| s.to_string())
    };
    explicit
        .and_then(non_empty)
        .or_else(|| env.and_then(non_empty))
        .or_else(head)
}

/// Advance HEAD after a run finished successfully.
///
/// Best-effort: lineage bookkeeping must never fail a training run that
/// already produced its results. Returns whether HEAD moved.
pub fn advance_head(root: impl AsRef<Path>, run_id: &str) -> bool {
    write_head(root, run_id).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn head_roundtrips_and_tolerates_absence() {
        let root = TempDir::new().unwrap();
        assert_eq!(read_head(root.path()), None);

        write_head(root.path(), "run_a").unwrap();
        assert_eq!(read_head(root.path()).as_deref(), Some("run_a"));

        // The trailing newline the writer adds is not part of the id.
        let raw = fs::read_to_string(head_path(root.path())).unwrap();
        assert_eq!(raw, "run_a\n");

        write_head(root.path(), "run_b").unwrap();
        assert_eq!(read_head(root.path()).as_deref(), Some("run_b"));

        clear_head(root.path()).unwrap();
        assert_eq!(read_head(root.path()), None);
        // Clearing twice is not an error.
        clear_head(root.path()).unwrap();
    }

    #[test]
    fn a_blank_head_reads_as_no_parent() {
        let root = TempDir::new().unwrap();
        fs::write(head_path(root.path()), "   \n").unwrap();
        assert_eq!(read_head(root.path()), None);
    }

    #[test]
    fn write_head_creates_the_root() {
        let root = TempDir::new().unwrap();
        let nested = root.path().join("deep").join(".soma");
        write_head(&nested, "run_x").unwrap();
        assert_eq!(read_head(&nested).as_deref(), Some("run_x"));
    }

    #[test]
    fn precedence_is_explicit_then_env_then_head() {
        let head = || Some("from_head".to_string());
        assert_eq!(
            resolve_parent_from(Some("explicit"), Some("env"), head).as_deref(),
            Some("explicit")
        );
        assert_eq!(
            resolve_parent_from(None, Some("env"), head).as_deref(),
            Some("env")
        );
        assert_eq!(
            resolve_parent_from(None, None, head).as_deref(),
            Some("from_head")
        );
        assert_eq!(resolve_parent_from(None, None, || None), None);
        // Blank strings are absence, not a parent named "".
        assert_eq!(resolve_parent_from(Some("  "), Some(""), || None), None);
    }

    #[test]
    fn head_is_not_read_when_a_parent_is_already_known() {
        let mut read = false;
        let head = || {
            read = true;
            Some("from_head".to_string())
        };
        assert_eq!(
            resolve_parent_from(Some("explicit"), None, head).as_deref(),
            Some("explicit")
        );
        assert!(!read, "the filesystem is only touched as a last resort");
    }

    #[test]
    fn checkout_refuses_a_run_that_does_not_exist() {
        let root = TempDir::new().unwrap();
        let err = checkout(root.path(), "typo_run").unwrap_err();
        assert!(err.to_string().contains("no run 'typo_run'"), "{err}");
        assert_eq!(read_head(root.path()), None, "HEAD must not move");

        // A run with a manifest is a run.
        let run_dir = root.path().join("runs").join("run_real");
        fs::create_dir_all(&run_dir).unwrap();
        fs::write(run_dir.join("manifest.json"), "{}").unwrap();
        assert!(run_exists(root.path(), "run_real"));
        checkout(root.path(), "run_real").unwrap();
        assert_eq!(read_head(root.path()).as_deref(), Some("run_real"));
    }

    #[test]
    fn advancing_head_reports_whether_it_moved() {
        let root = TempDir::new().unwrap();
        assert!(advance_head(root.path(), "run_1"));
        assert_eq!(read_head(root.path()).as_deref(), Some("run_1"));
    }
}
