//! Writing a file so that a crash cannot leave half of one.

use somatize_core::error::{Result, SomaError};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes concurrent writers *within* one process; the pid
/// distinguishes them across processes.
static WRITE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Write `data` to `path` atomically: a temp file in the same directory,
/// fsync, rename.
///
/// A crash mid-write leaves only an orphan temp file — never read back,
/// because every reader goes through the exact final path — so the rename
/// is the commit point. Renames are idempotent, which makes concurrent
/// writers of the same key safe on POSIX filesystems.
///
/// Both halves are load-bearing, and two of the four copies this replaces
/// were missing them:
///
/// - the **unique temp name** (pid + sequence): with a fixed one, two
///   writers to the same directory interleave into each other's temp file
///   and one of them renames a half-written mixture into place;
/// - the **`sync_all`**: without it the rename can reach the disk before
///   the bytes do, so a crash leaves a file that is present, has the right
///   name, and is truncated — and a truncated JSON document parses cleanly
///   right up to the tear.
pub fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| SomaError::Other(format!("`{}` has no parent directory", path.display())))?;
    fs::create_dir_all(parent)?;

    let seq = WRITE_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp-{}-{seq}", std::process::id()));
    {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(data)?;
        file.sync_all()?;
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

/// [`write_atomic`] over pretty-printed JSON.
pub fn write_json_atomic<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let json =
        serde_json::to_vec_pretty(value).map_err(|e| SomaError::Serialization(e.to_string()))?;
    write_atomic(path, &json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_write_lands_whole() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("thing.json");
        write_atomic(&path, b"{\"a\":1}").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "{\"a\":1}");
    }

    #[test]
    fn a_second_write_replaces_the_first() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("thing");
        write_atomic(&path, b"first").unwrap();
        write_atomic(&path, b"second").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "second");
    }

    #[test]
    fn nothing_but_the_target_is_left_behind() {
        // The temp name is unique per write, so a run that leaves one
        // behind is a bug in the rename path, not in the naming.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..4 {
            write_atomic(&dir.path().join("thing"), format!("{i}").as_bytes()).unwrap();
        }
        let names: Vec<String> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["thing".to_string()]);
    }

    /// Concurrent writers to one path must each land whole. A fixed temp
    /// name is what this catches: the two would share a file, and the
    /// loser of the rename race would publish a mixture.
    #[test]
    fn concurrent_writers_never_publish_a_mixture() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("contended");
        let a = vec![b'a'; 64 * 1024];
        let b = vec![b'b'; 64 * 1024];

        std::thread::scope(|s| {
            for payload in [&a, &b] {
                s.spawn(|| {
                    for _ in 0..20 {
                        write_atomic(&path, payload).unwrap();
                    }
                });
            }
        });

        let written = fs::read(&path).unwrap();
        assert!(
            written == a || written == b,
            "a reader saw a mixture of two writes, {} bytes",
            written.len()
        );
    }
}
