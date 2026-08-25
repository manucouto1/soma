//! What somebody said about a commit: a verdict, a note, a reason for pruning.
//!
//! # Append-only, and the latest wins
//!
//! A verdict is mutable — a line looks promising in the morning and is a dead
//! end by six — and the obvious shape for that is a row somebody updates. It is
//! the wrong one here. A store lives on NFS or in a bucket, and
//! `soma_next_store` says why in as many words: *making an index the truth
//! would mean a single writer, and a single writer over NFS is exactly where
//! this breaks*.
//!
//! So nothing is ever updated. Saying something is **claiming the next slot**
//! under a commit, and what a commit's verdict *is* means the last one anybody
//! claimed. Two machines saying something at the same moment both succeed, one
//! after the other, because [`claim`](soma_next_store::Store::claim) either
//! takes a name or finds it taken — and neither loses what the other wrote.
//!
//! # What is in the record and what is in the blob
//!
//! The store's cost rule, which is not a detail: a record comes back free on a
//! scan, a blob is a fetch. So a verdict lives in the **record**, where reading
//! forty commits' worth costs one scan and nothing else, and the prose lives in
//! the **blob**, fetched only by whoever asked to read it.
//!
//! # What a verdict is now, and what left
//!
//! There were four: `promising`, `dead-end`, `superseded` and `invalid`. Only
//! the last was ever a property of the *code*. The other three were somebody
//! deciding where to go next, stuck onto a commit with no scope saying which
//! line they meant and no room for why — and they now live in layer 2 as
//! [`Kind::Decision`](crate::moves::Kind::Decision) with a
//! [`Course`](crate::moves::Course), where they hang under the question they
//! were answering. A commit's course is **derived** from there by
//! [`Moves::decided`](crate::moves::Moves::decided).
//!
//! Nothing was migrated, because nothing had to be: an old record saying
//! `verdict=dead-end` no longer reads as a verdict, so it comes back as a note
//! with its prose intact. What is lost is the word, not the writing.
//!
//! # Pruning and invalidation are not stored
//!
//! What is written down is *this commit is invalid*. That its descendants are
//! suspect is **derived**, by walking git at the time somebody asks — so a
//! commit made tomorrow under an invalid one is suspect the moment it exists,
//! without anybody having gone back to write it down.

use serde::{Deserialize, Serialize};
use soma_next_store::{Digest, Meta, Store};
use std::collections::BTreeMap;
use std::fmt;

/// How many slots to try before giving up on being heard.
///
/// Only ever more than one when somebody else claimed the same slot in the same
/// instant, so this is a bound on a race and not on a queue.
const PATIENCE: u32 = 32;

/// What somebody found out about a commit itself.
///
/// Two, and on purpose. These are the only judgements about the **code and its
/// measurements** rather than about where to go next, and
/// [`Invalid`](Self::Invalid) is the only one with a mechanical consequence:
/// everything under it is suspect. Deciding a line is dead, or promising, or
/// done better elsewhere, says nothing about whether what is already there was
/// measured correctly — those are decisions, they carry a scope and a reason,
/// and they live in layer 2 as [`Course`](crate::moves::Course).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    /// Something here was wrong — a bug in the data, a metric that lied. Every
    /// commit under it is suspect, and that is worked out and not written down.
    Invalid,
    /// Looked at and nothing wrong with it.
    ///
    /// Exists so that being wrong about being wrong is sayable. A commit
    /// nobody judged is already not invalid, so this says nothing new on its
    /// own — its whole use is to be the **last** word after an `invalid`, and
    /// to lift the doubt from everything that inherited it. Without it a
    /// mistaken `invalid` would poison a subtree for good, which is the one
    /// thing an append-only record must never let happen.
    Sound,
}

impl Verdict {
    /// A word we no longer read comes back as `None`, which makes the saying a
    /// note. That is the whole of the migration away from `promising`,
    /// `dead-end` and `superseded`: the prose was always the part worth
    /// keeping, and it is untouched.
    pub fn read(said: &str) -> Option<Self> {
        match said {
            "invalid" => Some(Self::Invalid),
            "sound" => Some(Self::Sound),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Invalid => "invalid",
            Self::Sound => "sound",
        }
    }

    /// Whether everything under it inherits doubt.
    pub fn reaches_down(&self) -> bool {
        matches!(self, Self::Invalid)
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One thing somebody said, as it comes back from a scan.
#[derive(Debug, Clone)]
pub struct Saying {
    pub commit: String,
    /// `None` for a note that judged nothing.
    pub verdict: Option<Verdict>,
    pub who: String,
    /// Seconds since the epoch, stamped by the store.
    pub when: u64,
    /// Where the prose is. Not the prose: that is a fetch, and a scan is not.
    pub said: Digest,
    /// Which slot under its commit, so the last word can be found.
    pub nth: u32,
}

/// The sayings of one investigation, kept in a store.
///
/// `tree` is what lets several investigations share one store without seeing
/// each other, and it is in the name for the same reason a study's is: a name
/// is the one part of this that cannot be refactored later.
pub struct Journal<'a> {
    kept: &'a dyn Store,
    tree: String,
}

impl<'a> Journal<'a> {
    pub fn of(tree: impl Into<String>, kept: &'a dyn Store) -> Self {
        Self {
            kept,
            tree: tree.into(),
        }
    }

    /// The name a commit's `nth` saying is bound under.
    ///
    /// The same shape a study's trials have — `<study>/trial/<n>/<attempt>` —
    /// one noun along, which is not a coincidence: `exp/<tree>/<commit>` **is**
    /// the study name, so trials for this version land underneath without a
    /// line of soma-next changing.
    fn named(&self, commit: &str, nth: u32) -> String {
        format!("exp/{}/{commit}/said/{nth}", self.tree)
    }

    /// Says something about a commit, and returns which slot it landed in.
    ///
    /// Claims rather than binds, so two people saying something at the same
    /// moment both get heard: whoever is told the slot is taken asks for the
    /// next one, exactly as a worker does with a trial.
    pub fn say(
        &self,
        commit: &str,
        verdict: Option<Verdict>,
        who: &str,
        prose: &str,
    ) -> Result<u32, Trouble> {
        let said = self.kept.put(prose.as_bytes()).map_err(Trouble::Store)?;
        let first = self.last_of(commit)?.map_or(0, |last| last + 1);
        // Each turn is the **next slot along**, not a retry of the same one:
        // being told a slot is taken means somebody else's saying is in it and
        // stands, and this one goes after it. Neither is lost, which is the
        // whole point of claiming rather than binding.
        for nth in first..first + PATIENCE {
            let meta: Meta = [
                ("what".to_string(), "said".to_string()),
                ("commit".to_string(), commit.to_string()),
                ("who".to_string(), who.to_string()),
            ]
            .into_iter()
            .chain(verdict.map(|one| ("verdict".to_string(), one.to_string())))
            .collect();
            if self
                .kept
                .claim(&self.named(commit, nth), &said, meta)
                .map_err(Trouble::Store)?
            {
                return Ok(nth);
            }
        }
        Err(Trouble::Crowded {
            commit: commit.to_string(),
        })
    }

    /// The highest slot already taken under a commit, if any.
    fn last_of(&self, commit: &str) -> Result<Option<u32>, Trouble> {
        Ok(self
            .all()?
            .iter()
            .filter(|saying| saying.commit == commit)
            .map(|saying| saying.nth)
            .max())
    }

    /// Everything anybody said in this investigation, oldest first.
    ///
    /// A scan and **no fetches**: what a verdict is comes back in the record,
    /// so reading forty commits costs one walk of the store. The prose is a
    /// digest here and stays one until somebody asks to read it.
    pub fn all(&self) -> Result<Vec<Saying>, Trouble> {
        let under = format!("exp/{}/", self.tree);
        let mut said: Vec<Saying> = self
            .kept
            .bound()
            .map_err(Trouble::Store)?
            .into_iter()
            .filter_map(|bound| {
                // A store holds whatever anybody put in it — snapshots, a
                // cache, another investigation — so this is a question and not
                // an assumption.
                let rest = bound.name.strip_prefix(&under)?;
                let (commit, nth) = rest.split_once("/said/")?;
                Some(Saying {
                    commit: commit.to_string(),
                    verdict: beside(&bound.meta, "verdict").and_then(Verdict::read),
                    who: beside(&bound.meta, "who").unwrap_or("nobody").to_string(),
                    when: bound.when,
                    said: bound.digest,
                    nth: nth.parse().ok()?,
                })
            })
            .collect();
        said.sort_by_key(|saying| (saying.commit.clone(), saying.nth));
        Ok(said)
    }

    /// What each commit's verdict **is**: the last one anybody claimed.
    ///
    /// Notes in between are not verdicts and do not overwrite one — somebody
    /// writing down what they saw has not thereby changed their mind.
    pub fn verdicts(&self) -> Result<BTreeMap<String, Verdict>, Trouble> {
        let mut latest = BTreeMap::new();
        for saying in self.all()? {
            if let Some(verdict) = saying.verdict {
                latest.insert(saying.commit, verdict);
            }
        }
        Ok(latest)
    }

    /// The prose of one saying.
    pub fn read(&self, saying: &Saying) -> Result<String, Trouble> {
        let bytes = self
            .kept
            .get(&saying.said)
            .map_err(Trouble::Store)?
            .unwrap_or_default();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// One field of a record, if it is there.
fn beside<'a>(meta: &'a Meta, what: &str) -> Option<&'a str> {
    meta.iter()
        .find(|(said, _)| said == what)
        .map(|(_, value)| value.as_str())
}

#[derive(Debug)]
pub enum Trouble {
    Store(soma_next_store::StoreError),
    /// Thirty-two slots taken while trying to use one. Either a great many
    /// people are talking about one commit at once, or something is wrong.
    Crowded {
        commit: String,
    },
}

impl fmt::Display for Trouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(why) => write!(f, "the journal could not be reached: {why}"),
            Self::Crowded { commit } => {
                write!(f, "too many people saying things about {commit} at once")
            }
        }
    }
}

impl std::error::Error for Trouble {}
