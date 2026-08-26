//! What data sits under each version, and what belongs to none of them.
//!
//! Iterating five versions of one question in an afternoon leaves five sets of
//! intermediates in the store, and a month later nobody can say which was
//! whose. Nothing is written down to answer that: a probe already says what
//! every node's answer will be called, so attribution is two questions to the
//! store and no index anybody has to keep up to date.
//!
//! Two ways to attribute, kept both because they say different things. *By
//! key*: a key that matches means this is exactly the value that version would
//! ask for — exact, and fragile in one place, since a key is computed against
//! the probing interpreter's environment, so probing a three-month-old commit
//! today gives keys that match nothing stored back then. *By fingerprint*:
//! whoever ran wrote which node and which code version produced each value, so
//! it answers about old data, which is what nobody can attribute from memory.
//!
//! A value whose fingerprint belongs to no version nameable here comes out
//! anyway, saying so. Keeping it quiet would let the mute hashes back in
//! through the back door.

use crate::snapshot::Snapshot;
use somatize_core::Key;
use somatize_store::{Bound, Store};
use std::collections::{BTreeMap, HashMap};

/// How a value was found to belong to a version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum How {
    /// Named as that version will name that node: the value it would ask for,
    /// not one like it.
    Named,
    /// Produced by that version's code, per what whoever ran wrote beside it.
    /// Survives the environment of back then no longer existing.
    Written,
}

/// A value in the store, and which version it turned out to be from.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Belongs {
    /// The name it is bound under in the store.
    pub name: String,
    /// Which node produced it, if that was said.
    pub node: Option<String>,
    /// Which version of the code, if that was said.
    pub fingerprint: Option<String>,
    /// With what input, by the name its content has.
    pub input: Option<String>,
    /// Against what environment, by its short name.
    pub environment: Option<String>,
    /// When it was bound, in seconds since the epoch.
    pub when: u64,
    /// Which commits it is from, and how that is known. Empty is an answer:
    /// from none that can be named here.
    pub of: BTreeMap<String, How>,
}

impl Belongs {
    /// Whether it turned out to be from none of the versions asked about.
    ///
    /// **Not the same as being spare**: it may be from a branch nobody looked
    /// at, a commit that is gone, or an environment that cannot be reproduced.
    pub fn is_nobodys(&self) -> bool {
        self.of.is_empty()
    }
}

/// What is in the store, attributed to the versions passed in.
///
/// One walk of the store and not one blob read: what is needed is in the
/// record, which is this store's cost rule from the first day.
pub fn under(
    store: &dyn Store,
    known: &HashMap<&str, Snapshot>,
) -> Result<Vec<Belongs>, Box<dyn std::error::Error>> {
    // Both indices inverted once, rather than walking the versions per value:
    // with forty commits and a few thousand values that is the same work done
    // thousands of times.
    let mut by_name: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut by_code: HashMap<(&str, &str), Vec<&str>> = HashMap::new();
    let names: Vec<(&str, BTreeMap<String, String>)> = known
        .iter()
        .map(|(commit, taken)| (*commit, taken.names()))
        .collect();
    let codes: Vec<(&str, BTreeMap<String, String>)> = known
        .iter()
        .map(|(commit, taken)| (*commit, taken.fingerprints()))
        .collect();
    // A key names the **recipe**; a name is where that recipe's value is bound,
    // and they are not the same string. The store translates, so it is asked
    // rather than having its `format!` copied here.
    let bound_as: Vec<(&str, Vec<String>)> = names
        .iter()
        .map(|(commit, said)| {
            (
                *commit,
                said.values()
                    .map(|key| somatize_store::name_of(&Key::new(key.clone())))
                    .collect(),
            )
        })
        .collect();
    for (commit, said) in &bound_as {
        for name in said {
            by_name.entry(name.as_str()).or_default().push(commit);
        }
    }
    for (commit, said) in &codes {
        for (node, written) in said {
            by_code
                .entry((node.as_str(), written.as_str()))
                .or_default()
                .push(commit);
        }
    }

    let mut said: Vec<Belongs> = store
        .bound()?
        .into_iter()
        .filter(|bound| !bookkeeping(bound))
        .map(|bound| {
            let meta = |what: &str| {
                bound
                    .meta
                    .iter()
                    .find(|(said, _)| said == what)
                    .map(|(_, told)| told.clone())
            };
            let (node, fingerprint) = (meta(somatize_core::NODE), meta(somatize_core::FINGERPRINT));
            let mut of: BTreeMap<String, How> = BTreeMap::new();
            // Fingerprint first and key after, so `Named` wins where both
            // hold: it is the stronger of the two.
            if let (Some(node), Some(written)) = (&node, &fingerprint) {
                for commit in by_code
                    .get(&(node.as_str(), written.as_str()))
                    .into_iter()
                    .flatten()
                {
                    of.insert((*commit).to_string(), How::Written);
                }
            }
            for commit in by_name.get(bound.name.as_str()).into_iter().flatten() {
                of.insert((*commit).to_string(), How::Named);
            }
            Belongs {
                name: bound.name.clone(),
                node,
                fingerprint,
                input: meta(somatize_core::INPUT),
                environment: meta(ENVIRONMENT),
                when: bound.when,
                of,
            }
        })
        .collect();
    said.sort_by(|a, b| (b.when, &a.name).cmp(&(a.when, &b.name)));
    Ok(said)
}

/// What the environment a value was produced against is called in its `meta`.
///
/// The word is `somatize._environment`'s and not the engine's, so it is not
/// among the core's constants. Written here once and not at every reader.
pub const ENVIRONMENT: &str = "env";

/// What is not a run's data but the bookkeeping of whoever looks.
///
/// Three writers share this store and only one leaves intermediates: `exp/…`
/// is this tool's own notebook and carries the commit in its name, `snapshot:…`
/// is its probe cache, and `env/…` is a reading of an environment that soma
/// writes so the short name values carry can be understood. Everything else is
/// data, including what nobody turns out to own — a filter that kept only the
/// recognised would be a listing that can never show the case that matters.
fn bookkeeping(bound: &Bound) -> bool {
    ["exp/", "snapshot:", "env/"]
        .iter()
        .any(|prefix| bound.name.starts_with(prefix))
}
