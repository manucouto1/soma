//! Reading and writing a [`Study`] to disk.
//!
//! `soma-core` describes what a study *is*; putting it on a filesystem is
//! I/O, and a contract crate does not do I/O. Bring [`StudyIo`] into scope
//! and the calls read exactly as they did before the split.
//!
//! ```ignore
//! use somatize_runtime::study_io::StudyIo;
//!
//! study.save(dir.join("study.json"))?;
//! let study = Study::load(dir.join("study.json"))?;
//! ```

use somatize_core::error::{Result, SomaError};
use somatize_core::optimizer::study::Study;
use std::path::Path;

/// Persist a [`Study`] as JSON.
pub trait StudyIo: Sized {
    /// Serialize to pretty JSON at `path`.
    fn save(&self, path: impl AsRef<Path>) -> Result<()>;

    /// Load a study previously written by [`save`](StudyIo::save) — or by
    /// a tracker's `study.json`, which is the same format.
    fn load(path: impl AsRef<Path>) -> Result<Self>;
}

impl StudyIo for Study {
    fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let json =
            serde_json::to_vec_pretty(self).map_err(|e| SomaError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn load(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| SomaError::Serialization(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::optimizer::search::SearchSpace;
    use somatize_core::optimizer::study::{Direction, Objective, SearchStrategy, Study};

    fn study() -> Study {
        Study::new(
            "persisted",
            SearchSpace::new(),
            SearchStrategy::Grid { points_per_dim: 3 },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        )
    }

    #[test]
    fn a_study_survives_a_round_trip_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("study.json");

        let mut original = study();
        original.tags = vec!["mos".into()];
        original.planned_trials = Some(6);
        original.git_sha = Some("abc123".into());
        original.save(&path).unwrap();

        let back = Study::load(&path).unwrap();
        assert_eq!(back.name, "persisted");
        assert_eq!(back.tags, vec!["mos"]);
        assert_eq!(back.planned_trials, Some(6));
        assert_eq!(back.git_sha.as_deref(), Some("abc123"));
        assert!(back.created_at.is_some());
    }

    /// A missing file and a corrupt one are different failures, and a
    /// caller that wants to distinguish "no study yet" from "the study on
    /// disk is broken" needs them to stay different.
    #[test]
    fn load_errors_are_typed() {
        let missing = Study::load("/nonexistent/dir/study.json");
        assert!(matches!(missing, Err(SomaError::Io(_))), "{missing:?}");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("study.json");
        std::fs::write(&path, "{not json").unwrap();
        let corrupt = Study::load(&path);
        assert!(
            matches!(corrupt, Err(SomaError::Serialization(_))),
            "{corrupt:?}"
        );
    }
}
