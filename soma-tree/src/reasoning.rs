//! The reasoning, read back: every move with what it stands at, and what folds.
//!
//! [`walk`](crate::walk) is the record read this way and the argument is the
//! same one: what the terminal prints and what a notebook draws are the **same
//! answer** read twice, so it is worked out once and here. A second copy in
//! Python would be a view that quietly disagreed with the terminal about what
//! an investigation contains.
//!
//! It answers in **names** and not in ids. A move carries a name because the
//! store's slot stops identifying it the moment nobody is holding it in a
//! variable — and reading it back is exactly that moment. The id is kept as a
//! field, because it is what says in which order these were made.
//!
//! What is added here and is in no store: whether a move is on a line somebody
//! abandoned. [`Moves::decided`] answers that for commits only, and an attempt
//! nobody ever ran cites none — which is precisely the move a decision needs to
//! be able to abandon.

use crate::moves::{
    Cited, Course, Kind, Move, MoveId, Moves, Says, Scope, Standing, Trouble, Undernath,
};
use serde::Serialize;
use somatize_store::Store;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One move, derived and in names.
#[derive(Debug, Clone, Serialize)]
pub struct Seen {
    pub name: String,
    /// The store's slot. Kept because it is what orders siblings by when they
    /// were made, which no walk can recover.
    pub id: MoveId,
    pub kind: Kind,
    pub prose: String,
    pub who: String,
    pub when: u64,
    pub under: Vec<String>,
    /// Where it belongs on the page when nothing hangs it: a decision's scope
    /// names what is abandoned, and a decision drawn floating beside the line
    /// it ended is the one thing a reader cannot join up.
    pub about: Vec<String>,
    pub scope: Vec<String>,
    pub cites: Vec<Cited>,
    /// Only a decision carries one.
    pub course: Option<Course>,
    /// Only a question or a hypothesis. `None` and not `open`: an attempt is
    /// not a question nobody has answered.
    pub standing: Option<Standing>,
    /// Whether something abandoned the line it is on. Derived, never stored.
    pub pruned: bool,
}

/// One thing said from a move towards another, in names.
#[derive(Debug, Clone, Serialize)]
pub struct Told {
    pub from: String,
    pub says: Says,
    pub to: String,
    pub scope: Vec<String>,
    pub partly: bool,
}

/// A line somebody abandoned: what it hides, and why, in words.
#[derive(Debug, Clone, Serialize)]
pub struct Folded {
    /// The move the decision named. Everything under it is in `hides`.
    pub root: String,
    /// The decision that said so.
    pub by: String,
    pub course: Course,
    /// The decision's own prose. **Pruning says why or it is deletion with a
    /// nicer name.**
    pub why: String,
    /// What folds with it, in the order they were made, the root among them.
    pub hides: Vec<String>,
}

/// A whole reasoning, ready to print or to draw.
#[derive(Debug, Clone, Serialize)]
pub struct Reasoning {
    pub tree: String,
    /// In the order they were made, which is the order siblings are drawn in.
    pub moves: Vec<Seen>,
    pub says: Vec<Told>,
    pub folded: Vec<Folded>,
}

impl Reasoning {
    /// The move of that name, if there is one.
    pub fn went(&self, name: &str) -> Option<&Seen> {
        self.moves.iter().find(|seen| seen.name == name)
    }

    /// What hangs under that move, in the order they were made — a decision
    /// that named it and hangs nowhere included.
    pub fn below(&self, name: &str) -> Vec<&Seen> {
        self.moves
            .iter()
            .filter(|other| {
                other
                    .under
                    .iter()
                    .chain(&other.about)
                    .any(|one| one == name)
            })
            .collect()
    }

    /// What a scope with those roots reaches: the roots and everything under
    /// them, in the order they were made.
    ///
    /// The one derivation a reader cannot redo by hand and get right — `under`
    /// is multivalued, so it is a walk over a DAG and not a subtree. With it,
    /// *do these two scopes touch* is an intersection. Fails if a name is not
    /// one of these moves.
    pub fn covers(&self, by: &[String]) -> Result<Vec<String>, Trouble> {
        let mut children: HashMap<&str, Vec<&str>> = HashMap::new();
        for seen in &self.moves {
            for parent in &seen.under {
                children.entry(parent).or_default().push(&seen.name);
            }
        }
        let mut reached: BTreeSet<&str> = BTreeSet::new();
        let mut asking: Vec<&str> = Vec::new();
        for name in by {
            let seen = self
                .went(name)
                .ok_or_else(|| Trouble::NoSuchName { name: name.clone() })?;
            asking.push(&seen.name);
        }
        while let Some(one) = asking.pop() {
            if !reached.insert(one) {
                continue;
            }
            asking.extend(children.get(one).into_iter().flatten().copied());
        }
        Ok(self
            .moves
            .iter()
            .filter(|seen| reached.contains(seen.name.as_str()))
            .map(|seen| seen.name.clone())
            .collect())
    }
}

/// Reads a whole reasoning out of a store. Nothing is run and no repository is
/// touched: it is what was written down, plus what follows from it.
pub fn reasoned(tree: &str, kept: &dyn Store) -> Result<Reasoning, Trouble> {
    let moves = Moves::of(tree, kept);
    let known = moves.all()?;
    let under = moves.under()?;
    let standing = moves.standing()?;
    let courses = moves.courses()?;
    let named = |id: MoveId| known.get(&id).map(|body| body.name.clone());
    // A name that resolves to nothing is a move somebody wrote an edge to and
    // then could not read back — dropped rather than drawn as a blank, since a
    // box with no name is worse than an edge that is not there.
    let names = |ids: Vec<MoveId>| ids.into_iter().filter_map(named).collect();

    let seen = known
        .iter()
        .map(|(id, body)| Seen {
            name: body.name.clone(),
            id: *id,
            kind: body.kind,
            prose: body.prose.clone(),
            who: body.who.clone(),
            when: body.when,
            under: names(under.parents_of(*id)),
            about: match body.kind == Kind::Decision && under.parents_of(*id).is_empty() {
                true => names(body.scope.0.iter().copied().collect()),
                false => Vec::new(),
            },
            scope: names(body.scope.0.iter().copied().collect()),
            cites: body.cites.clone(),
            course: body.course,
            standing: standing.get(id).copied(),
            pruned: matches!(
                courses.get(id),
                Some((_, Course::Abandon | Course::Superseded))
            ),
        })
        .collect();

    let mut said: Vec<(MoveId, MoveId, Told)> = moves
        .says()?
        .into_iter()
        .filter_map(|one| {
            Some((
                one.from,
                one.to,
                Told {
                    from: named(one.from)?,
                    says: one.says,
                    to: named(one.to)?,
                    scope: names(one.scope.0.iter().copied().collect()),
                    partly: one.in_part,
                },
            ))
        })
        .collect();
    said.sort_by_key(|(from, to, told)| (*from, *to, told.says.as_str()));

    Ok(Reasoning {
        tree: tree.to_string(),
        moves: seen,
        says: said.into_iter().map(|(_, _, told)| told).collect(),
        folded: folded(&known, &under, &courses),
    })
}

/// The lines somebody abandoned, one row per root a decision named.
///
/// Only the decision that **won** the root gets a row: pursuing a line again is
/// deciding again, and yesterday's abandonment is still written with its reason
/// without being what folds today.
fn folded(
    known: &BTreeMap<MoveId, Move>,
    under: &Undernath,
    courses: &BTreeMap<MoveId, (MoveId, Course)>,
) -> Vec<Folded> {
    let mut rows = Vec::new();
    for (id, body) in known {
        let Some(course) = body.course else { continue };
        if !matches!(course, Course::Abandon | Course::Superseded) {
            continue;
        }
        for root in crate::moves::abandoning(*id, body, under).0 {
            if courses.get(&root).map(|(by, _)| *by) != Some(*id) {
                continue;
            }
            let Some(named) = known.get(&root) else {
                continue;
            };
            // What actually folds, and not everything below: a branch that was
            // taken up again is under an abandoned root and is not hidden.
            let mut hides: Vec<MoveId> = Scope::of([root])
                .covers(under)
                .into_iter()
                .filter(|one| {
                    matches!(
                        courses.get(one),
                        Some((_, Course::Abandon | Course::Superseded))
                    )
                })
                .collect();
            hides.sort_unstable();
            rows.push(Folded {
                root: named.name.clone(),
                by: body.name.clone(),
                course,
                why: body.prose.clone(),
                hides: hides
                    .into_iter()
                    .filter_map(|one| known.get(&one).map(|body| body.name.clone()))
                    .collect(),
            });
        }
    }
    rows.sort_by(|a, b| a.root.cmp(&b.root));
    rows
}

/// An indented outline of what is there, the way the terminal prints it.
///
/// One line per move, because what an outline is read for is the shape. Here
/// and not in the command for the reason the whole module is: an outline that
/// disagreed with the figure about what folds would be two tools.
pub fn outlined(reasoning: &Reasoning, from: Option<&str>, all_lines: bool) -> Vec<String> {
    let mut lines = Vec::new();
    let hidden: BTreeMap<&str, &Folded> = match all_lines {
        true => BTreeMap::new(),
        false => reasoning
            .folded
            .iter()
            .map(|one| (one.root.as_str(), one))
            .collect(),
    };
    let roots: Vec<&Seen> = match from {
        Some(name) => reasoning.moves.iter().filter(|s| s.name == name).collect(),
        // What hangs nowhere is a root, which is how a move nobody hung is
        // drawn: work waiting for a place, not a move that hides.
        None => reasoning
            .moves
            .iter()
            .filter(|s| s.under.is_empty() && s.about.is_empty())
            .collect(),
    };
    let mut drawn: BTreeSet<&str> = BTreeSet::new();
    for root in roots {
        outline_from(reasoning, root, 0, &hidden, &mut drawn, &mut lines);
    }
    lines
}

/// How much of a move's prose fits on its line before it is cut.
const ENOUGH: usize = 64;

fn outline_from<'a>(
    reasoning: &'a Reasoning,
    seen: &'a Seen,
    depth: usize,
    hidden: &BTreeMap<&str, &Folded>,
    drawn: &mut BTreeSet<&'a str>,
    lines: &mut Vec<String>,
) {
    let pad = "  ".repeat(depth);
    // A move under two parents is written under both, and its subtree once: the
    // second reading says where it also belongs without repeating the branch.
    let again = !drawn.insert(seen.name.as_str());
    let said = match (seen.standing, seen.course) {
        (Some(standing), _) => format!(" · {standing}"),
        (_, Some(course)) => format!(" · {course}"),
        _ => String::new(),
    };
    lines.push(format!(
        "{pad}{} · {}{said} · {}{}",
        seen.name,
        seen.kind,
        shortened(&seen.prose),
        match again {
            true => " · (again)",
            false => "",
        }
    ));
    if again {
        return;
    }
    if let Some(one) = hidden.get(seen.name.as_str()) {
        lines.push(format!(
            "{pad}  ⋯ {} folded · {} · {}",
            one.hides.len(),
            one.course,
            shortened(&one.why)
        ));
        return;
    }
    for child in reasoning.below(&seen.name) {
        outline_from(reasoning, child, depth + 1, hidden, drawn, lines);
    }
}

/// The first line of some prose, cut where a terminal stops reading it.
fn shortened(prose: &str) -> String {
    let first = prose.lines().next().unwrap_or_default().trim();
    match first.char_indices().nth(ENOUGH) {
        Some((at, _)) => format!("{}…", &first[..at].trim_end()),
        None => first.to_string(),
    }
}
