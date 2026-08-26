//! What was run with a version: the trials, their states and their curves.
//!
//! A commit is the version and does not change. What gets tried with it grows
//! without end — a hundred trials, three analyses, a report — and none of that
//! can touch a commit's hash, so it is **associated** with the version rather
//! than versioned. soma writes it from the machine running the study; here it
//! is only read, because whoever claimed a trial is its only writer and a
//! second one would invent a race that does not exist today.
//!
//! The name is the whole of the coupling. soma binds each trial to
//! `<study>/trial/<n>/<attempt>`, and a study's name is any string:
//!
//! ```text
//! exp/<tree>/<commit>              ← that version's study
//! exp/<tree>/<commit>/trial/3/0    ← its fourth trial, first attempt
//! exp/<tree>/<commit>/said/2       ← what somebody said about that commit
//! ```
//!
//! A commit's study **is** the prefix its journal already lives under, so the
//! trials land beneath it with no line of soma changing: no correspondence
//! record, no index to keep. And the store's cost rule holds — state, point and
//! score are in the **record** — so counting forty commits' trials is one scan
//! and only the curve is paid for when somebody asks to see it.
//!
//! What cannot be said from here is **which is best**: whether `0.0837` is good
//! depends on a direction that lives in the `Goal` handed to a sampler and is
//! written in no record. Guessing it would be the quiet lie this tool exists
//! not to let past, so either `soma-tree.toml` declares it or it is not said,
//! and the range — true without knowing the direction — is shown instead.

use serde::{Deserialize, Serialize};
use somatize_store::{Digest, Store};
use std::collections::BTreeMap;
use std::fmt;

/// Which way is better. Not in the store: declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Goal {
    /// Less is better: a loss, an error, a time.
    Min,
    /// More is better: an accuracy, an F1, a reward.
    Max,
}

impl Goal {
    pub fn read(said: &str) -> Option<Self> {
        match said {
            "min" | "minimize" => Some(Self::Min),
            "max" | "maximize" => Some(Self::Max),
            _ => None,
        }
    }

    /// The best of a few, in the declared direction.
    pub fn best_of(&self, values: impl IntoIterator<Item = f64>) -> Option<f64> {
        values
            .into_iter()
            .filter(|one| !one.is_nan())
            .reduce(|best, one| match self {
                Self::Min => best.min(one),
                Self::Max => best.max(one),
            })
    }
}

/// A trial, as it comes back from a scan.
///
/// The states are soma's and not this side's — `running`, `done`, `pruned`,
/// `failed` — and travel as text, so a growing vocabulary is not two places to
/// migrate.
#[derive(Debug, Clone, Serialize)]
pub struct Trial {
    pub trial: u32,
    /// Which attempt. The highest wins: claiming is a link, so a trial whose
    /// machine died is rescued by claiming the next one.
    pub attempt: u32,
    pub state: Option<String>,
    /// The configuration that ran, as `str(point)` wrote it.
    pub point: Option<String>,
    /// Absent while running. Present on a `pruned`, and **not comparable**
    /// with a `done`'s: it was measured after fewer epochs.
    pub score: Option<f64>,
    pub who: Option<String>,
    pub when: u64,
    /// Where the curve is. Not the curve: that is a fetch and this is not.
    #[serde(skip)]
    pub kept: Digest,
}

impl Trial {
    /// Whether its score can be compared with another `done`'s.
    pub fn comparable(&self) -> bool {
        self.state.as_deref() == Some("done") && self.score.is_some()
    }
}

/// A trial's curve, which is what costs a fetch.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Curve {
    #[serde(default)]
    pub point: String,
    #[serde(default)]
    pub reports: Vec<f64>,
    #[serde(default)]
    pub state: Option<String>,
    /// Why it stopped. What a `pruned` has and a list of numbers does not.
    #[serde(default)]
    pub because: Option<String>,
    #[serde(default)]
    pub took: Option<f64>,
}

/// What is seen of a commit's trials without reading a single blob.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Tally {
    pub trials: u32,
    pub running: u32,
    pub done: u32,
    pub pruned: u32,
    pub failed: u32,
    /// The range of what is comparable, true without knowing the direction.
    pub lowest: Option<f64>,
    pub highest: Option<f64>,
    /// The best **only if somebody declared which way is better**. `None`
    /// otherwise, and then the range is shown in its place.
    pub best: Option<f64>,
}

/// The trials of one investigation, kept in a store.
pub struct Trials<'a> {
    kept: &'a dyn Store,
    tree: String,
    goal: Option<Goal>,
}

impl<'a> Trials<'a> {
    pub fn of(tree: impl Into<String>, kept: &'a dyn Store) -> Self {
        Self {
            kept,
            tree: tree.into(),
            goal: None,
        }
    }

    /// With the declared direction, if there is one.
    pub fn towards(mut self, goal: Option<Goal>) -> Self {
        self.goal = goal;
        self
    }

    /// The name of a commit's study, which is the whole of the link.
    pub fn study(&self, commit: &str) -> String {
        format!("exp/{}/{commit}", self.tree)
    }

    /// A commit's trials, the highest attempt of each, in order.
    ///
    /// One scan and no fetches.
    pub fn of_commit(&self, commit: &str) -> Result<Vec<Trial>, Trouble> {
        let mut best: BTreeMap<u32, Trial> = BTreeMap::new();
        let under = format!("{}/trial/", self.study(commit));
        for bound in self.kept.bound().map_err(Trouble::Store)? {
            let Some((trial, attempt)) = numbered(&bound.name, &under) else {
                continue;
            };
            match best.get(&trial) {
                Some(had) if had.attempt >= attempt => {}
                _ => {
                    best.insert(
                        trial,
                        Trial {
                            trial,
                            attempt,
                            state: beside(&bound.meta, "state").map(str::to_string),
                            point: beside(&bound.meta, "point").map(str::to_string),
                            // Python's `repr(float(score))`, which Rust reads
                            // the same. A trial with no score is preferable to
                            // one with an invented score.
                            score: beside(&bound.meta, "score").and_then(|one| one.parse().ok()),
                            who: beside(&bound.meta, "who").map(str::to_string),
                            when: bound.when,
                            kept: bound.digest,
                        },
                    );
                }
            }
        }
        Ok(best.into_values().collect())
    }

    /// How many trials each commit has and how they are going, in **one scan**.
    ///
    /// Asking commit by commit would be forty walks of the store to draw a list
    /// of forty rows.
    pub fn counted(&self) -> Result<BTreeMap<String, Tally>, Trouble> {
        let under = format!("exp/{}/", self.tree);
        // The highest attempt of each `(commit, trial)` before counting:
        // counting the records would count a rescued trial twice.
        let mut best: BTreeMap<(String, u32), Highest> = BTreeMap::new();
        for bound in self.kept.bound().map_err(Trouble::Store)? {
            let Some(rest) = bound.name.strip_prefix(&under) else {
                continue;
            };
            let Some((commit, numbers)) = rest.split_once("/trial/") else {
                continue;
            };
            let Some((trial, attempt)) = numbered(numbers, "") else {
                continue;
            };
            let mine = (commit.to_string(), trial);
            match best.get(&mine) {
                Some(had) if had.attempt >= attempt => {}
                _ => {
                    best.insert(
                        mine,
                        Highest {
                            attempt,
                            state: beside(&bound.meta, "state").map(str::to_string),
                            score: beside(&bound.meta, "score").and_then(|one| one.parse().ok()),
                        },
                    );
                }
            }
        }

        let mut counted: BTreeMap<String, Tally> = BTreeMap::new();
        let mut comparable: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for ((commit, _), one) in best {
            let tally = counted.entry(commit.clone()).or_default();
            tally.trials += 1;
            match one.state.as_deref() {
                Some("running") => tally.running += 1,
                Some("done") => tally.done += 1,
                Some("pruned") => tally.pruned += 1,
                Some("failed") => tally.failed += 1,
                _ => {}
            }
            // Only a `done`'s enters the range: a `pruned`'s is real and not
            // comparable — measured after fewer epochs — and it would make the
            // range wider than anything anybody measured.
            if let (Some("done"), Some(score)) = (one.state.as_deref(), one.score) {
                comparable.entry(commit).or_default().push(score);
            }
        }
        for (commit, scores) in comparable {
            let Some(tally) = counted.get_mut(&commit) else {
                continue;
            };
            tally.lowest = Goal::Min.best_of(scores.iter().copied());
            tally.highest = Goal::Max.best_of(scores.iter().copied());
            tally.best = self.goal.and_then(|goal| goal.best_of(scores));
        }
        Ok(counted)
    }

    /// A trial's curve. **This one is a fetch**, which is why it is apart.
    pub fn curve(&self, of: &Trial) -> Result<Option<Curve>, Trouble> {
        let Some(bytes) = self.kept.get(&of.kept).map_err(Trouble::Store)? else {
            return Ok(None);
        };
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|why| Trouble::Garbled(why.to_string()))
    }
}

/// The highest attempt seen of a trial, while scanning.
struct Highest {
    attempt: u32,
    state: Option<String>,
    score: Option<f64>,
}

/// The `(trial, attempt)` that name is, or `None` if it is not one.
///
/// A question and not an assumption, as in soma and for the same reason: a
/// store holds whatever anybody put in it.
fn numbered(name: &str, under: &str) -> Option<(u32, u32)> {
    let rest = name.strip_prefix(under)?;
    let (trial, attempt) = rest.split_once('/')?;
    Some((trial.parse().ok()?, attempt.parse().ok()?))
}

fn beside<'a>(meta: &'a somatize_store::Meta, what: &str) -> Option<&'a str> {
    meta.iter()
        .find(|(said, _)| said == what)
        .map(|(_, value)| value.as_str())
}

#[derive(Debug)]
pub enum Trouble {
    Store(somatize_store::StoreError),
    Garbled(String),
}

impl fmt::Display for Trouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(why) => write!(f, "the trials could not be reached: {why}"),
            Self::Garbled(why) => write!(f, "a curve could not be read: {why}"),
        }
    }
}

impl std::error::Error for Trouble {}
