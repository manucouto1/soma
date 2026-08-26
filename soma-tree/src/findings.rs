//! What an edit did, read rather than decided.
//!
//! The model lives in `somatize.foreseen` and not here — its vocabulary, its
//! propagation and its 24 tests, one implementation rather than two that
//! drift. This is a typed reader of what it answers.
//!
//! One axis is still this side's, and it is not a second model: `identity` is
//! a class name, so `Embed(0.5)` and `Embed(0.9)` are one recipe and one AST,
//! and `foreseen.changes` answers `{}` to that edit. The probe reads what an
//! object was constructed with and folds it into the fingerprint before
//! asking, so `STALE` — and the walk that turns it into `SUSPECT` — becomes
//! true of this axis for free.

use serde::Deserialize;
use std::collections::BTreeMap;

/// A name that moved because a name above it moved. Not where an edit is.
pub const DOWNSTREAM: &str = "DOWNSTREAM";
/// Settled at another state: retrained, or another version of a dataset.
pub const RESETTLED: &str = "RESETTLED";
/// The salt moved. A statement about the store, not about the code.
pub const SALTED: &str = "SALTED";
/// Its name did not move and its code did: the cache will hit.
pub const STALE: &str = "STALE";
/// Something above it is `STALE`, so what reaches it is last week's answer.
pub const SUSPECT: &str = "SUSPECT";

/// The findings that mean somebody typed something here.
///
/// `STALE` is one of them, and it is why this list is not just `CHANGED`: a
/// rewritten `forward` moves no name, so leaving it out would answer *nobody
/// edited anything* to the very edit this exists to catch.
const AN_EDIT: [&str; 4] = ["CHANGED", "ADDED", "GONE", STALE];
/// The findings a name moved by, that no code moved by.
const NOT_A_VARIANT: [&str; 2] = [RESETTLED, SALTED];

/// One step: what `foreseen.changes` said, and the one axis it was not given.
#[derive(Debug, Default, Clone, Deserialize, serde::Serialize)]
pub struct Findings {
    /// `{node: [finding, ...]}`. A node with nothing said about it is absent.
    pub findings: BTreeMap<String, Vec<String>>,
    /// `{node: [before, after]}` — the readable form of a declaration that
    /// moved, so a report can print `Embed(0.5) → Embed(0.9)` rather than the
    /// two digests the fold turned it into.
    pub declared: BTreeMap<String, [String; 2]>,
}

impl Findings {
    /// Whether anything at all was said.
    pub fn is_quiet(&self) -> bool {
        self.findings.is_empty()
    }

    /// The nodes carrying that finding.
    pub fn saying(&self, finding: &str) -> Vec<&str> {
        self.findings
            .iter()
            .filter(|(_, said)| said.iter().any(|one| one == finding))
            .map(|(node, _)| node.as_str())
            .collect()
    }

    /// Where somebody typed, with what merely inherited it left out.
    ///
    /// Without this separation, inserting one node in a graph of forty reports
    /// forty changes and says nothing about which edit caused them.
    pub fn the_edit(&self) -> Vec<&str> {
        self.findings
            .iter()
            .filter(|(_, said)| said.iter().any(|one| AN_EDIT.contains(&one.as_str())))
            .map(|(node, _)| node.as_str())
            .collect()
    }

    /// The nodes whose numbers cannot be compared with the ones from before.
    ///
    /// A retrained node is not one of them, and neither is what sits under it:
    /// its results moved without it becoming another variant, which is what a
    /// trial is.
    pub fn not_comparable(&self) -> Vec<&str> {
        if self.the_edit().is_empty() {
            return Vec::new();
        }
        self.findings
            .iter()
            .filter(|(_, said)| {
                said.iter()
                    .any(|one| !NOT_A_VARIANT.contains(&one.as_str()))
            })
            .map(|(node, _)| node.as_str())
            .collect()
    }
}
