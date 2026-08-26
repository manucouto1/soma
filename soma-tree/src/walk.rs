//! A line of exploration, as data.
//!
//! What `log` prints and what a browser draws are the **same answer** read two
//! ways, so it is worked out once and here. A second copy of this in a request
//! handler would be a view that quietly disagreed with the terminal about what
//! an investigation contains, which is the one thing neither can afford.

use crate::findings::Findings;
use crate::journal::{Journal, Verdict};
use crate::moves::{Course, Moves};
use crate::revision;
use crate::snapshot::{Probing, Snapshot};
use crate::trials::{Goal, Tally, Trials};
use serde::Serialize;
use somatize_store::Store;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// One commit of a line, with everything known about it that is not a step.
#[derive(Debug, Serialize)]
pub struct Stop {
    pub commit: String,
    /// Twelve characters, which is what a person reads.
    pub short: String,
    pub subject: String,
    /// Who it comes from. **The edges of the DAG**: a range flattens two
    /// branches into an order, and drawing that order would be drawing a lie.
    pub parents: Vec<String>,
    /// Whether somebody found something wrong with this commit itself.
    pub verdict: Option<Verdict>,
    /// What the reasoning decided about the line this commit is on, if
    /// anything. **Derived** — from a decision's scope, down through the moves
    /// it covers, out to the commits those cite — so abandoning a question's
    /// line reaches an attempt hung under it tomorrow with nobody writing
    /// anything down. It deliberately does not reach a fork off an abandoned
    /// attempt: that is a sibling and starts clean, because trying something
    /// else is the move you make *because* it was a dead end.
    pub decided: Option<Course>,
    /// Whether something above it is [`Verdict::Invalid`]. Worked out from git
    /// rather than stored, so a commit made after the verdict is marked the
    /// moment it exists.
    pub doubted: bool,
    /// What was run with this version: how many trials and how they are going.
    ///
    /// A commit is the version and does not change; trials grow and are
    /// **associated** with it, not versioned. From the same scan as everything
    /// else, because soma put the state and the score in the record — counting
    /// forty versions is one walk and only the curve is paid for apart.
    pub trials: Tally,
    /// Whether it folds when drawn: it is on a line somebody decided to
    /// abandon or call superseded, and nobody found anything wrong with it.
    ///
    /// **Pruning is not drawing, never deleting.** The stop still comes back
    /// whole; all this says is that a tree of forty variants does not read, and
    /// whoever draws may fold this one. Computed here and not by whoever draws,
    /// or the rule would live in two languages and the terminal and the view
    /// would fold different things, both looking right.
    pub pruned: bool,
    /// Whether it is only here so the one above it has something to be
    /// compared against. A range says which commits to *show*.
    pub context: bool,
    /// When it was made. Which of three variants was tried first is a question
    /// about this and not about the order a walk arrived in.
    pub when: u64,
}

/// One step: what the edit from `from` to `to` did.
#[derive(Debug, Serialize)]
pub struct Step {
    pub from: String,
    pub to: String,
    #[serde(flatten)]
    pub found: Findings,
    /// What was built differently around the two probes. Usually empty.
    pub drift: Vec<(String, String, String)>,
}

/// A whole line, ready to print or to draw.
#[derive(Debug, Serialize)]
pub struct Walk {
    pub tree: String,
    pub built_from: String,
    pub stops: Vec<Stop>,
    pub steps: Vec<Step>,
}

impl Walk {
    /// The step arriving at that commit, if it has one.
    pub fn step_to(&self, commit: &str) -> Option<&Step> {
        self.steps.iter().find(|step| step.to == commit)
    }
}

/// Whether a stop folds when a pruned line is drawn.
///
/// The whole rule in one place, because the terminal and the view both use it.
///
/// What somebody decided to abandon or call superseded folds. **What somebody
/// judged wrong does not**: an `invalid` commit is what casts doubt on the
/// measurement the decision leaned on, and hiding it would hide the very
/// reason to look again — same for whatever inherits that doubt. A `sound`
/// does fold: it says somebody looked and found nothing, so the decision
/// stands.
pub fn folds(decided: Option<Course>, judged: Option<Verdict>, doubted: bool) -> bool {
    if !matches!(decided, Some(Course::Abandon) | Some(Course::Superseded)) {
        return false;
    }
    !doubted && !matches!(judged, Some(Verdict::Invalid))
}

/// What it takes to read what is already known of an investigation: its name,
/// where it is kept, and which way is better.
///
/// The three travel together because they come out of one `soma-tree.toml` and
/// none means anything without the others: a store without the tree's name
/// returns another investigation's records, and a score without the direction
/// does not say whether it is good.
pub struct Remembered<'a> {
    pub tree: &'a str,
    pub kept: &'a dyn Store,
    pub goal: Option<Goal>,
}

/// Works out a line: what was probed, what each step did, and what was said.
///
/// `shown` is what a range named; `commits` is that plus the one underneath,
/// which is probed and never drawn as a stop of its own.
pub fn walked(
    repo: &Path,
    known_as: Remembered<'_>,
    probing: &Probing,
    shown: &[String],
    commits: &[String],
    known: &HashMap<&str, Snapshot>,
) -> Result<Walk, Box<dyn std::error::Error>> {
    let Remembered { tree, kept, goal } = known_as;
    // Probing is optional. A repository from before soma — a finished paper,
    // work nobody runs any more — has a history, a journal, trials and a line
    // of reasoning worth reading, and no graph to probe. Without a probe there
    // are stops and no steps: what is missing is **what each edit did**.
    let probing = if known.is_empty() && !commits.is_empty() {
        None
    } else {
        if commits
            .iter()
            .any(|commit| !known.contains_key(commit.as_str()))
        {
            return Err("some commit was never probed".into());
        }
        Some(probing)
    };
    let parents: HashMap<String, Vec<String>> =
        revision::parents_of(repo, commits).into_iter().collect();

    // A step is an **edge**, so it is read off the parents. Pairing adjacent
    // entries of a walk would, with three branches off one commit, compare
    // three different lines of exploration with each other and answer
    // confidently about an edit nobody made.
    let pairs: Vec<(String, String)> = commits
        .iter()
        .flat_map(|commit| {
            parents
                .get(commit)
                .into_iter()
                .flatten()
                .filter(|parent| known.contains_key(parent.as_str()))
                .map(move |parent| (parent.clone(), commit.clone()))
        })
        .collect();
    let found = match probing {
        Some(probing) => probing.compared(known, &pairs)?,
        None => Vec::new(),
    };

    // A verdict is written about **one** commit; that its descendants are
    // suspect is worked out here. Walked over the parents already in hand
    // rather than asked of git: an ancestry-path question needs a tip to walk
    // **towards**, and with three branches the tip is usually on somebody
    // else's, so a verdict cast on one variant would reach nothing.
    let journal = Journal::of(tree, kept);
    let verdicts = journal.verdicts()?;
    let mut doubted: HashSet<String> = HashSet::new();
    let mut asking: Vec<&String> = verdicts
        .iter()
        .filter(|(_, verdict)| verdict.reaches_down())
        .map(|(commit, _)| commit)
        .collect();
    while let Some(commit) = asking.pop() {
        if !doubted.insert(commit.clone()) {
            continue;
        }
        // Its children: whoever names it as a parent.
        asking.extend(
            parents
                .iter()
                .filter(|(_, of)| of.contains(commit))
                .map(|(child, _)| child),
        );
    }

    // Not being able to read the reasoning is no reason not to draw the
    // record — a tree with no decisions is what there is on day one — and the
    // same goes for not being able to count what was run.
    let decided = Moves::of(tree, kept).decided().unwrap_or_default();
    let mut counted = Trials::of(tree, kept)
        .towards(goal)
        .counted()
        .unwrap_or_default();

    let told = revision::told(repo, commits);
    let stops = commits
        .iter()
        .map(|commit| Stop {
            short: commit[..12.min(commit.len())].to_string(),
            when: told.get(commit).map(|(when, _)| *when).unwrap_or_default(),
            subject: told
                .get(commit)
                .map(|(_, said)| said.clone())
                .unwrap_or_default(),
            parents: parents.get(commit).cloned().unwrap_or_default(),
            verdict: verdicts.get(commit).copied(),
            decided: decided.get(commit).copied(),
            pruned: folds(
                decided.get(commit).copied(),
                verdicts.get(commit).copied(),
                doubted.contains(commit),
            ),
            trials: counted.remove(commit).unwrap_or_default(),
            doubted: doubted.contains(commit),
            context: !shown.contains(commit),
            commit: commit.clone(),
        })
        .collect();

    let steps = pairs
        .into_iter()
        .zip(found)
        .map(|((from, to), found)| Step {
            drift: known[from.as_str()]
                .drifted_from(&known[to.as_str()])
                .into_iter()
                .map(|(what, was, is)| (what.to_string(), was, is))
                .collect(),
            from,
            to,
            found,
        })
        .collect();

    Ok(Walk {
        tree: tree.to_string(),
        // Said out loud rather than left blank, which would look like a fault.
        built_from: if probing.is_none() {
            "no probe — this repository does not declare what to build".to_string()
        } else {
            commits
                .first()
                .and_then(|first| known.get(first.as_str()))
                .map(|first| first.built_from.clone())
                .unwrap_or_default()
        },
        stops,
        steps,
    })
}
