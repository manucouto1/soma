//! The reasoning: questions, hypotheses, attempts, findings and decisions.
//!
//! Layer 1 — commits, snapshots, findings per node, trials — answers *what was
//! run and what came out*. This answers *what somebody was trying to find out*,
//! and shares none of its units: a commit is nobody's decision, a question
//! nobody tried has no commit, and one move can produce three branches. What
//! decides which layer something belongs to: if it can be recomputed it is
//! record, and if somebody thought it, it is reasoning.
//!
//! It is a **DAG**, and one case forces it. Two live questions — does more
//! capacity improve interpretability? does it improve performance? — one
//! variant validating each, and then the question neither contained: what if I
//! put them together? That attempt hangs under **both**. One parent would mean
//! choosing, or duplicating the node, and a duplicated node is two that go out
//! of step. Hence [`Undernath`] being multivalued, and hence refusing cycles as
//! they are written: a walk over one does not end.
//!
//! Everything carries a scope, including what is said. A question is about
//! some moves and not the whole investigation, and an answer holds **where it
//! holds**. Without that, *validated* and *refuted* on one hypothesis look like
//! a contradiction when normally they are two facts about two situations — A
//! alone worked, A+B cancel out. There is a dispute only when two edges of
//! opposite sign have scopes that **touch**.

use serde::{Deserialize, Serialize};
use somatize_store::{Bound as Record, Digest, Meta, Store};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;

/// How many slots to try before giving up. More than one turn only when
/// somebody claimed the same one in the same instant: a race, not a queue.
const PATIENCE: u32 = 32;

/// What identifies a move. Its slot, because a move is mutable — you reword
/// its prose — and so cannot be addressed by its content.
pub type MoveId = u32;

/// The five kinds, and there are no more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    /// What is not known. It gets **answered**. The only kind that can exist
    /// with nothing under it: a question nobody tried is work outstanding.
    Question,
    /// A proposed, falsifiable answer. It gets **validated** or **refuted** —
    /// verbs a question does not have, which is why it is not a question
    /// reworded.
    Hypothesis,
    /// What was tried, citing layer 1. The only kind that touches it.
    Attempt,
    /// What the evidence says. The verb edges come from here, and it is the
    /// only kind exportable to a knowledge lake.
    Finding,
    /// What is done about it. Apart from the finding because two people can
    /// agree on one and disagree on the other.
    Decision,
}

impl Kind {
    pub fn read(said: &str) -> Option<Self> {
        match said {
            "question" => Some(Self::Question),
            "hypothesis" => Some(Self::Hypothesis),
            "attempt" => Some(Self::Attempt),
            "finding" => Some(Self::Finding),
            "decision" => Some(Self::Decision),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Question => "question",
            Self::Hypothesis => "hypothesis",
            Self::Attempt => "attempt",
            Self::Finding => "finding",
            Self::Decision => "decision",
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What something is about: some moves and whatever hangs under them.
///
/// **Roots and not a free set**, which is what makes it affordable. *The whole
/// encoder branch* is a root, *this step* is a root, the whole investigation is
/// none. An arbitrary set would be truer and would turn *do they overlap?* into
/// something to materialise rather than walk.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Scope(pub BTreeSet<MoveId>);

impl Scope {
    /// About everything. What makes a general question general.
    pub fn everything() -> Self {
        Self(BTreeSet::new())
    }

    pub fn of(roots: impl IntoIterator<Item = MoveId>) -> Self {
        Self(roots.into_iter().collect())
    }

    pub fn is_everything(&self) -> bool {
        self.0.is_empty()
    }

    /// The moves it covers: its roots and everything hanging under them.
    pub fn covers(&self, under: &Undernath) -> HashSet<MoveId> {
        let mut reached = HashSet::new();
        let mut asking: Vec<MoveId> = self.0.iter().copied().collect();
        while let Some(one) = asking.pop() {
            if !reached.insert(one) {
                continue;
            }
            asking.extend(under.children_of(one));
        }
        reached
    }

    /// Whether two scopes touch. What separates a contradiction from two facts
    /// about two different situations.
    pub fn touches(&self, other: &Self, under: &Undernath) -> bool {
        // What covers everything touches everything, including another such.
        if self.is_everything() || other.is_everything() {
            return true;
        }
        let mine = self.covers(under);
        other.covers(under).iter().any(|one| mine.contains(one))
    }
}

/// What a finding says, and towards what.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Says {
    /// Towards a question.
    Answers,
    /// Towards a hypothesis.
    Validates,
    /// Towards a hypothesis.
    Refutes,
    /// From an attempt towards the attempts it composes. Not `under`: it says
    /// this attempt **is** the composition of those, which is what lets *each
    /// worked alone, together they cancel* read as what it is.
    Combines,
}

impl Says {
    pub fn read(said: &str) -> Option<Self> {
        match said {
            "answers" => Some(Self::Answers),
            "validates" => Some(Self::Validates),
            "refutes" => Some(Self::Refutes),
            "combines" => Some(Self::Combines),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Answers => "answers",
            Self::Validates => "validates",
            Self::Refutes => "refutes",
            Self::Combines => "combines",
        }
    }

    /// Who may say it and to whom: a `validates` pointing at an attempt means
    /// nothing, and accepting it stores a sentence nobody can read.
    fn between(&self) -> (&'static [Kind], &'static [Kind]) {
        match self {
            Self::Answers => (&[Kind::Finding], &[Kind::Question]),
            Self::Validates | Self::Refutes => (&[Kind::Finding], &[Kind::Hypothesis]),
            Self::Combines => (&[Kind::Attempt], &[Kind::Attempt]),
        }
    }
}

impl fmt::Display for Says {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One thing said from a move towards another.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Said {
    pub from: MoveId,
    pub to: MoveId,
    pub says: Says,
    /// Where it holds. Almost never everywhere, and that is the point.
    #[serde(default)]
    pub scope: Scope,
    /// Whether it settles the question or only pushes it. *Does more capacity
    /// help?* is not settled at once: three attempts each answer part.
    #[serde(default)]
    pub in_part: bool,
}

/// What was decided about the line a decision is about.
///
/// This used to be a verdict stuck on a commit — `promising`, `dead-end`,
/// `superseded` — and it was never a property of the code. Here it has what it
/// lacked there: a **scope** saying which line it is about, a **reason** in the
/// prose, and a place in the DAG under the question it was answering. `invalid`
/// is not here and will not be: that one really is the code's, and it stays in
/// the journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Course {
    /// Carry on this way. The default reading of a line nobody judged, so
    /// saying it is only needed to take an abandonment back.
    Pursue,
    /// Explored and not worth carrying on. Kept, never deleted: a line that
    /// did not work is the most reusable thing an investigation produces, and
    /// the only thing that stops it being discovered again.
    Abandon,
    /// Somebody did it better elsewhere. Not wrong, not the way.
    Superseded,
}

impl Course {
    pub fn read(said: &str) -> Option<Self> {
        match said {
            "pursue" => Some(Self::Pursue),
            "abandon" => Some(Self::Abandon),
            "superseded" => Some(Self::Superseded),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pursue => "pursue",
            Self::Abandon => "abandon",
            Self::Superseded => "superseded",
        }
    }
}

impl fmt::Display for Course {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A move, without its edges.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Move {
    pub id: MoveId,
    /// What its author calls it, unique within the tree.
    ///
    /// The id is the store's — it says what order these were written in, and
    /// it works as an identity for exactly as long as somebody is holding it
    /// in a variable. Picking an investigation up again is the ordinary case:
    /// another process, a tool call, or the same person a week later, none of
    /// whom remember that the capacity question was `7`. So a move is reached
    /// by a word somebody chose, and that is what makes `go` possible at all.
    pub name: String,
    pub kind: Kind,
    /// What it is about. Only questions and hypotheses carry one; in the rest
    /// it is everything and is not read.
    #[serde(default)]
    pub scope: Scope,
    pub prose: String,
    /// What it cites of layer 1: commits, trials, artifacts.
    ///
    /// Carried by an attempt — the commit that ran, and the trials it ran — and
    /// by a finding — the trial it was seen in. A question, a hypothesis and a
    /// decision are about moves and not about layer-1 pieces, and letting them
    /// cite would let a question point at a commit with nobody knowing what
    /// that means.
    #[serde(default)]
    pub cites: Vec<Cited>,
    /// What was decided. Only a [`Kind::Decision`] carries one; in the rest it
    /// is `None` and is not read.
    #[serde(default)]
    pub course: Option<Course>,
    pub who: String,
    pub when: u64,
}

/// A move somebody is writing, before the store gives it an id.
///
/// A struct and not seven arguments: four callers are coming — asking,
/// trying, finding and deciding — and each cares about a different subset, so
/// positionally they would be four call sites of `None, Vec::new(), None`
/// where nobody can see which blank is which.
pub struct Writing<'a> {
    pub kind: Kind,
    /// Unique within the tree. See [`Move::name`].
    pub name: &'a str,
    pub prose: &'a str,
    pub who: &'a str,
    /// Everything, unless this is a question, a hypothesis or a decision.
    pub scope: Scope,
    /// Only an attempt and a finding may carry one.
    pub cites: Vec<Cited>,
    /// Only a decision may carry one.
    pub course: Option<Course>,
}

impl<'a> Writing<'a> {
    /// The ordinary case: about everything, citing nothing, deciding nothing.
    pub fn new(kind: Kind, name: &'a str, prose: &'a str, who: &'a str) -> Self {
        Self {
            kind,
            name,
            prose,
            who,
            scope: Scope::everything(),
            cites: Vec::new(),
            course: None,
        }
    }
}

/// One piece of evidence from layer 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cited {
    /// `commit`, `trial`, `artifact`. Open on purpose: the vocabulary is the
    /// citer's, and this layer keeps it without learning it.
    pub what: String,
    pub id: String,
}

/// How a question stands, counting what has been said to it.
///
/// **Derived, never stored.** A *state* field somebody overwrites loses the
/// previous fact, and the previous fact is what makes a hypothesis go back to
/// open on its own when what refuted it is invalidated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Standing {
    /// Nobody has said anything yet.
    Open,
    /// Answered, and fully.
    Answered,
    /// Pushed along: everything that reached it said *in part*.
    Partly,
    Validated,
    PartlyValidated,
    Refuted,
    PartlyRefuted,
    /// Edges of opposite sign reach it **with scopes that touch**. The
    /// interesting state, and the one a field cannot express.
    Disputed,
    /// Validated in some situations and refuted in others, without touching.
    ///
    /// Not *in part* and not a dispute: the answer **depends**. *A alone
    /// improves, A+B cancel out* is the most informative outcome an
    /// investigation gives, and calling it `Partly` hid it under the word for a
    /// half-answered question.
    Depends,
}

/// Who hangs under whom. An index over the `under` edges, built so it can be
/// walked up and down without scanning again.
#[derive(Debug, Default)]
pub struct Undernath {
    over: BTreeMap<MoveId, BTreeSet<MoveId>>,
    below: BTreeMap<MoveId, BTreeSet<MoveId>>,
}

impl Undernath {
    pub fn add(&mut self, child: MoveId, parent: MoveId) {
        self.over.entry(child).or_default().insert(parent);
        self.below.entry(parent).or_default().insert(child);
    }

    pub fn parents_of(&self, child: MoveId) -> Vec<MoveId> {
        self.over
            .get(&child)
            .into_iter()
            .flatten()
            .copied()
            .collect()
    }

    pub fn children_of(&self, parent: MoveId) -> Vec<MoveId> {
        self.below
            .get(&parent)
            .into_iter()
            .flatten()
            .copied()
            .collect()
    }

    /// Whether `maybe` is above `one`, looking upwards.
    ///
    /// What it takes to refuse a cycle before writing it: with `under`
    /// multivalued the shape can no longer be trusted, and a cycle hangs every
    /// later walk — including the one that would draw it.
    pub fn is_over(&self, maybe: MoveId, one: MoveId) -> bool {
        let mut seen = HashSet::new();
        let mut asking = vec![one];
        while let Some(which) = asking.pop() {
            if which == maybe && !seen.is_empty() {
                return true;
            }
            if !seen.insert(which) {
                continue;
            }
            asking.extend(self.parents_of(which));
        }
        false
    }
}

/// The reasoning of one investigation, kept in a store.
pub struct Moves<'a> {
    kept: &'a dyn Store,
    tree: String,
}

impl<'a> Moves<'a> {
    pub fn of(tree: impl Into<String>, kept: &'a dyn Store) -> Self {
        Self {
            kept,
            tree: tree.into(),
        }
    }

    fn named(&self, id: MoveId, what: &str, nth: u32) -> String {
        format!("exp/{}/move/{id}/{what}/{nth}", self.tree)
    }

    /// Writes a new move and returns its id.
    ///
    /// Claims the slot exactly as a trial does: no coordinator, and whoever
    /// finds it taken asks for the next. Two people writing at once get two
    /// moves, not one lost.
    pub fn add(&self, writing: Writing<'_>) -> Result<MoveId, Trouble> {
        let Writing {
            kind,
            name,
            prose,
            who,
            scope,
            cites,
            course,
        } = writing;
        if course.is_some() && kind != Kind::Decision {
            return Err(Trouble::NotADecision { kind });
        }
        let name = name.trim();
        let first = self.all()?.keys().copied().max().map_or(0, |last| last + 1);

        // The name is claimed before anything is written, and `claim` and not
        // read-then-write: between reading that a name is free and taking it,
        // somebody else does the same, and two moves answer to one word while
        // the store says nothing. It is the same primitive that hands out a
        // trial, for the same reason.
        //
        // It is claimed at the id we mean to take, and rebound below if the
        // slot loop had to move on. If that loop gives up altogether the name
        // is left held, pointing at a move that was never written — which
        // `went` reports as a name reaching nothing rather than as silence.
        let held = self.holds(name, first)?;
        if let Some(by) = held {
            return Err(Trouble::NameTaken {
                name: name.to_string(),
                by,
            });
        }

        for id in first..first + PATIENCE {
            let body = Move {
                id,
                name: name.to_string(),
                kind,
                scope: scope.clone(),
                prose: prose.trim().to_string(),
                cites: cites.clone(),
                course,
                who: who.to_string(),
                when: 0,
            };
            let bytes =
                serde_json::to_vec(&body).map_err(|why| Trouble::Garbled(why.to_string()))?;
            let digest = self.kept.put(&bytes).map_err(Trouble::Store)?;
            let mut meta: Meta = vec![
                ("what".into(), "move".into()),
                ("kind".into(), kind.to_string()),
                ("who".into(), who.to_string()),
            ];
            if let Some(course) = course {
                meta.push(("course".into(), course.to_string()));
            }
            if self
                .kept
                .claim(&self.named(id, "said", 0), &digest, meta)
                .map_err(Trouble::Store)?
            {
                if id != first {
                    // Ours to overwrite: nobody else got past the claim above.
                    self.point(name, id)?;
                }
                return Ok(id);
            }
        }
        Err(Trouble::Crowded)
    }

    /// Where a name lives, which is its own record and not a scan of the moves.
    ///
    /// A name is asked far more often than the whole reasoning is drawn — every
    /// `go`, every `--under` — and answering it by walking every move would
    /// make the cheapest question in the tool cost the most expensive read.
    fn calls(&self, name: &str) -> String {
        format!("exp/{}/named/{name}", self.tree)
    }

    /// Takes the name for this id, or says who already has it.
    fn holds(&self, name: &str, id: MoveId) -> Result<Option<MoveId>, Trouble> {
        let digest = self
            .kept
            .put(id.to_string().as_bytes())
            .map_err(Trouble::Store)?;
        let meta: Meta = vec![
            ("what".into(), "names".into()),
            ("move".into(), id.to_string()),
        ];
        match self
            .kept
            .claim(&self.calls(name), &digest, meta)
            .map_err(Trouble::Store)?
        {
            true => Ok(None),
            false => Ok(Some(self.went(name)?)),
        }
    }

    /// Points an already-held name at the id it ended up with.
    fn point(&self, name: &str, id: MoveId) -> Result<(), Trouble> {
        let digest = self
            .kept
            .put(id.to_string().as_bytes())
            .map_err(Trouble::Store)?;
        let meta: Meta = vec![
            ("what".into(), "names".into()),
            ("move".into(), id.to_string()),
        ];
        self.kept
            .bind(&self.calls(name), &digest, meta)
            .map_err(Trouble::Store)
    }

    /// The move that name reaches. One lookup, no scan.
    pub fn went(&self, name: &str) -> Result<MoveId, Trouble> {
        let name = name.trim();
        let bound = self
            .kept
            .resolve(&self.calls(name))
            .map_err(Trouble::Store)?
            .ok_or_else(|| Trouble::NoSuchName {
                name: name.to_string(),
            })?;
        let bytes = self
            .kept
            .get(&bound.digest)
            .map_err(Trouble::Store)?
            .ok_or_else(|| Trouble::NoSuchName {
                name: name.to_string(),
            })?;
        String::from_utf8_lossy(&bytes)
            .trim()
            .parse()
            .map_err(|_| Trouble::Garbled(format!("`{name}` does not name a move")))
    }

    /// Rewords a move. A new slot, and the last wins: what came before is
    /// still there, as in the journal.
    ///
    /// What arrives as `None` stays as it was, so correcting the prose does not
    /// wipe the scope or the other way round. The scope **has** to be
    /// correctable: in a decision it says which line is meant, and getting it
    /// wrong — reaching a finding, which is not a line, instead of the attempt
    /// it came from — leaves the decision reaching no commit at all with nothing
    /// warning. A course changes but is never removed: a decision that decides
    /// nothing any more is `pursue`.
    pub fn reword(
        &self,
        id: MoveId,
        prose: Option<&str>,
        scope: Option<Scope>,
        course: Option<Course>,
        who: &str,
    ) -> Result<u32, Trouble> {
        let mut body = self.all()?.remove(&id).ok_or(Trouble::NoSuchMove { id })?;
        if course.is_some() && body.kind != Kind::Decision {
            return Err(Trouble::NotADecision { kind: body.kind });
        }
        if let Some(prose) = prose {
            body.prose = prose.trim().to_string();
        }
        if let Some(scope) = scope {
            body.scope = scope;
        }
        if course.is_some() {
            body.course = course;
        }
        self.redrafted(id, body, who)
    }

    /// Writes one drafting of a move into the next slot.
    fn redrafted(&self, id: MoveId, mut body: Move, who: &str) -> Result<u32, Trouble> {
        body.who = who.to_string();
        let bytes = serde_json::to_vec(&body).map_err(|why| Trouble::Garbled(why.to_string()))?;
        let digest = self.kept.put(&bytes).map_err(Trouble::Store)?;
        let first = self.slots(id, "said")? + 1;
        for nth in first..first + PATIENCE {
            let mut meta: Meta = vec![
                ("what".into(), "move".into()),
                ("kind".into(), body.kind.to_string()),
                ("who".into(), who.to_string()),
            ];
            // The same meta `add` writes and not a poorer one: a record that
            // says less than the one before lies about what is underneath.
            if let Some(course) = body.course {
                meta.push(("course".into(), course.to_string()));
            }
            if self
                .kept
                .claim(&self.named(id, "said", nth), &digest, meta)
                .map_err(Trouble::Store)?
            {
                return Ok(nth);
            }
        }
        Err(Trouble::Crowded)
    }

    /// Adds one piece of evidence to a move.
    ///
    /// A new drafting and the last wins, as with everything here: evidence
    /// arrives after the attempt is written, because the trials run afterwards.
    /// Citing the same thing twice does not duplicate it — two people looking
    /// at one screen would ask for it, and a list with a trial twice says
    /// nothing a list with it once does not.
    pub fn cite(&self, id: MoveId, cited: Cited, who: &str) -> Result<u32, Trouble> {
        let known = self.all()?;
        let body = known.get(&id).ok_or(Trouble::NoSuchMove { id })?;
        if !matches!(body.kind, Kind::Attempt | Kind::Finding) {
            return Err(Trouble::CannotCite { kind: body.kind });
        }
        if body.cites.contains(&cited) {
            return self.slots(id, "said");
        }
        let mut body = body.clone();
        body.cites.push(cited);
        self.redrafted(id, body, who)
    }

    /// Hangs a move under another.
    ///
    /// The cycle is refused here, the only place it is cheap: reading it later
    /// means discovering it by having a walk hang.
    pub fn hang(&self, child: MoveId, parent: MoveId) -> Result<(), Trouble> {
        if child == parent {
            return Err(Trouble::Circular { child, parent });
        }
        let known = self.all()?;
        for one in [child, parent] {
            if !known.contains_key(&one) {
                return Err(Trouble::NoSuchMove { id: one });
            }
        }
        if self.under()?.is_over(child, parent) {
            return Err(Trouble::Circular { child, parent });
        }
        self.bind(
            child,
            "under",
            &parent.to_string(),
            &[("parent", &parent.to_string())],
        )
    }

    /// Says something from a move towards another.
    pub fn say(&self, said: Said) -> Result<(), Trouble> {
        let known = self.all()?;
        let (from, to) = (
            known
                .get(&said.from)
                .ok_or(Trouble::NoSuchMove { id: said.from })?,
            known
                .get(&said.to)
                .ok_or(Trouble::NoSuchMove { id: said.to })?,
        );
        let (says_from, says_to) = said.says.between();
        if !says_from.contains(&from.kind) || !says_to.contains(&to.kind) {
            return Err(Trouble::Nonsense {
                says: said.says,
                from: from.kind,
                to: to.kind,
            });
        }
        let body = serde_json::to_vec(&said).map_err(|why| Trouble::Garbled(why.to_string()))?;
        let digest = self.kept.put(&body).map_err(Trouble::Store)?;
        let target = said.to.to_string();
        let says = said.says.to_string();
        self.bound(
            said.from,
            "says",
            &digest,
            &[("says", says.as_str()), ("to", target.as_str())],
        )
    }

    /// Every move, by id, with its latest drafting.
    pub fn all(&self) -> Result<BTreeMap<MoveId, Move>, Trouble> {
        let under = format!("exp/{}/move/", self.tree);
        let mut latest: BTreeMap<MoveId, (u32, Digest, u64)> = BTreeMap::new();
        for bound in self.kept.bound().map_err(Trouble::Store)? {
            // A store holds whatever anybody put in it — a cache, another
            // investigation, an artifact — so this is a question.
            let Some(rest) = bound.name.strip_prefix(&under) else {
                continue;
            };
            let Some((id, nth)) = rest.split_once("/said/") else {
                continue;
            };
            let (Ok(id), Ok(nth)) = (id.parse::<MoveId>(), nth.parse::<u32>()) else {
                continue;
            };
            match latest.get(&id) {
                Some((had, _, _)) if *had >= nth => {}
                _ => {
                    latest.insert(id, (nth, bound.digest, bound.when));
                }
            }
        }

        let mut said = BTreeMap::new();
        for (id, (_, digest, when)) in latest {
            let Some(bytes) = self.kept.get(&digest).map_err(Trouble::Store)? else {
                continue;
            };
            if let Ok(mut body) = serde_json::from_slice::<Move>(&bytes) {
                body.when = when;
                said.insert(id, body);
            }
        }
        Ok(said)
    }

    /// The index of who hangs under whom.
    pub fn under(&self) -> Result<Undernath, Trouble> {
        let mut said = Undernath::default();
        for (child, bound) in self.records("under")? {
            // From the record and not the name: a name's last segment is the
            // **slot**, and reading it as the parent builds an index that looks
            // right and points at moves that do not exist.
            if let Some(parent) = beside(&bound.meta, "parent").and_then(|one| one.parse().ok()) {
                said.add(child, parent);
            }
        }
        Ok(said)
    }

    /// Everything anybody said from one move towards another.
    ///
    /// The last wins per `(from, to, verb)`, the same rule a drafting follows
    /// in `all`. Without it a scope could not be corrected: saying it again
    /// would leave both edges and the count would take both, so widening a
    /// scope would look like saying it twice. Saying it again **is** the gesture
    /// for changing your mind about a scope; withdrawing the verb entirely
    /// still has no gesture.
    ///
    /// The triple comes from the meta and not the body, so keeping the last one
    /// needs no read of the earlier ones.
    pub fn says(&self) -> Result<Vec<Said>, Trouble> {
        let under = format!("exp/{}/move/", self.tree);
        let mut latest: BTreeMap<(MoveId, String, String), (u32, Digest)> = BTreeMap::new();
        for bound in self.kept.bound().map_err(Trouble::Store)? {
            let Some(rest) = bound.name.strip_prefix(&under) else {
                continue;
            };
            let Some((from, nth)) = rest.split_once("/says/") else {
                continue;
            };
            let (Ok(from), Ok(nth)) = (from.parse::<MoveId>(), nth.parse::<u32>()) else {
                continue;
            };
            let (Some(verb), Some(to)) = (
                beside(&bound.meta, "says").map(str::to_string),
                beside(&bound.meta, "to").map(str::to_string),
            ) else {
                continue;
            };
            match latest.get(&(from, verb.clone(), to.clone())) {
                Some((had, _)) if *had >= nth => {}
                _ => {
                    latest.insert((from, verb, to), (nth, bound.digest));
                }
            }
        }

        let mut said = Vec::new();
        for (_, digest) in latest.into_values() {
            let Some(bytes) = self.kept.get(&digest).map_err(Trouble::Store)? else {
                continue;
            };
            if let Ok(one) = serde_json::from_slice::<Said>(&bytes) {
                said.push(one);
            }
        }
        Ok(said)
    }

    /// What was decided about each commit, derived from the reasoning.
    ///
    /// The bridge between the layers, and it runs this way: a commit does not
    /// store that it is abandoned. It is reached by going down — decision, its
    /// scope, the attempts that scope covers, the commits those attempts cite —
    /// so a commit made tomorrow under an abandoned line comes out abandoned
    /// with nobody writing anything again.
    ///
    /// **A decision with no scope is about where it hangs**, which is where it
    /// parts company with a question or a hypothesis, for which no scope means
    /// about everything. In a decision that would be a quiet trap: writing
    /// *this line is dead* while looking at one attempt would mark the whole
    /// tree. Abandoning everything means hanging it off the root or naming it.
    ///
    /// The last wins: changing your mind is deciding again, and yesterday's
    /// abandonment is still written with its reason.
    pub fn decided(&self) -> Result<BTreeMap<String, Course>, Trouble> {
        let known = self.all()?;
        let under = self.under()?;
        let mut said: BTreeMap<String, Course> = BTreeMap::new();
        // Oldest first, which here is the order of the ids: whoever speaks
        // last about a commit is the one that counts.
        for (id, body) in &known {
            let Some(course) = body.course else { continue };
            let scope = if body.scope.is_everything() {
                Scope::of(under.parents_of(*id))
            } else {
                body.scope.clone()
            };
            // Hanging off nothing and with no scope: it is about no line in
            // particular, so it colours none rather than colouring all.
            if scope.is_everything() {
                continue;
            }
            for one in scope.covers(&under) {
                let Some(reached) = known.get(&one) else {
                    continue;
                };
                for cited in &reached.cites {
                    if cited.what == "commit" {
                        said.insert(cited.id.clone(), course);
                    }
                }
            }
        }
        Ok(said)
    }

    /// How each question and hypothesis stands, counting what reached it.
    pub fn standing(&self) -> Result<BTreeMap<MoveId, Standing>, Trouble> {
        let known = self.all()?;
        let under = self.under()?;
        let says = self.says()?;
        Ok(known
            .iter()
            .filter(|(_, body)| matches!(body.kind, Kind::Question | Kind::Hypothesis))
            .map(|(id, body)| {
                let mine: Vec<&Said> = says.iter().filter(|one| one.to == *id).collect();
                (*id, stands(body.kind, &mine, &under))
            })
            .collect())
    }

    /// The prose behind a citation, or whatever was kept there.
    pub fn read(&self, digest: &Digest) -> Result<Option<Vec<u8>>, Trouble> {
        self.kept.get(digest).map_err(Trouble::Store)
    }

    /// The records under `exp/<tree>/move/<id>/<what>/…`, with their id.
    ///
    /// The whole record and not just the name: what is needed of an edge — who
    /// it points at — is in its meta, which a scan brings back free.
    fn records(&self, what: &str) -> Result<Vec<(MoveId, Record)>, Trouble> {
        let under = format!("exp/{}/move/", self.tree);
        let mark = format!("/{what}/");
        Ok(self
            .kept
            .bound()
            .map_err(Trouble::Store)?
            .into_iter()
            .filter_map(|bound| {
                let rest = bound.name.strip_prefix(&under)?;
                let (id, _) = rest.split_once(&mark)?;
                Some((id.parse().ok()?, bound))
            })
            .collect())
    }

    /// How many slots of one kind are taken under a move.
    fn slots(&self, id: MoveId, what: &str) -> Result<u32, Trouble> {
        let mark = format!("/move/{id}/{what}/");
        Ok(self
            .records(what)?
            .iter()
            .filter(|(which, bound)| *which == id && bound.name.contains(&mark))
            .filter_map(|(_, bound)| bound.name.rsplit('/').next()?.parse::<u32>().ok())
            .max()
            .unwrap_or(0))
    }

    /// Claims a slot for a fact with no body of its own: the name is the data.
    fn bind(
        &self,
        id: MoveId,
        what: &str,
        body: &str,
        meta: &[(&str, &str)],
    ) -> Result<(), Trouble> {
        let digest = self.kept.put(body.as_bytes()).map_err(Trouble::Store)?;
        self.bound(id, what, &digest, meta)
    }

    fn bound(
        &self,
        id: MoveId,
        what: &str,
        digest: &Digest,
        meta: &[(&str, &str)],
    ) -> Result<(), Trouble> {
        let first = self.slots(id, what)? + 1;
        for nth in first..first + PATIENCE {
            let meta: Meta = meta
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect();
            if self
                .kept
                .claim(&self.named(id, what, nth), digest, meta)
                .map_err(Trouble::Store)?
            {
                return Ok(());
            }
        }
        Err(Trouble::Crowded)
    }
}

/// How a question stands given what has been said to it.
///
/// The scopes do the work: two edges of opposite sign are a contradiction only
/// if they are about situations that touch. *A alone worked* and *A+B cancel
/// out* are two facts, not a conflict.
fn stands(kind: Kind, said: &[&Said], under: &Undernath) -> Standing {
    if said.is_empty() {
        return Standing::Open;
    }
    if kind == Kind::Question {
        let answers: Vec<&&Said> = said
            .iter()
            .filter(|one| one.says == Says::Answers)
            .collect();
        return match answers.iter().any(|one| !one.in_part) {
            true => Standing::Answered,
            false if answers.is_empty() => Standing::Open,
            false => Standing::Partly,
        };
    }

    let (yes, no): (Vec<&&Said>, Vec<&&Said>) = said
        .iter()
        .filter(|one| matches!(one.says, Says::Validates | Says::Refutes))
        .partition(|one| one.says == Says::Validates);
    if yes.is_empty() && no.is_empty() {
        return Standing::Open;
    }
    // Dispute is measured by overlap and not by presence: if nobody is talking
    // about the same thing, there is nothing to dispute.
    let disputed = yes
        .iter()
        .any(|a| no.iter().any(|b| a.scope.touches(&b.scope, under)));
    if disputed {
        return Standing::Disputed;
    }
    match (yes.is_empty(), no.is_empty()) {
        (false, true) if yes.iter().any(|one| !one.in_part) => Standing::Validated,
        (false, true) => Standing::PartlyValidated,
        (true, false) if no.iter().any(|one| !one.in_part) => Standing::Refuted,
        (true, false) => Standing::PartlyRefuted,
        // Both signs without touching: the answer depends on where you look.
        _ => Standing::Depends,
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
    Store(somatize_store::StoreError),
    Garbled(String),
    NoSuchMove { id: MoveId },
    Circular { child: MoveId, parent: MoveId },
    Nonsense { says: Says, from: Kind, to: Kind },
    NotADecision { kind: Kind },
    NameTaken { name: String, by: MoveId },
    NoSuchName { name: String },
    CannotCite { kind: Kind },
    Crowded,
}

impl fmt::Display for Trouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(why) => write!(f, "the reasoning could not be reached: {why}"),
            Self::Garbled(why) => write!(f, "something could not be written or read: {why}"),
            Self::NoSuchMove { id } => write!(f, "there is no move {id}"),
            Self::Circular { child, parent } => write!(
                f,
                "hanging {child} under {parent} would make a cycle, and a walk over one does not end"
            ),
            Self::Nonsense { says, from, to } => {
                write!(f, "a `{says}` from a {from} to a {to} means nothing")
            }
            Self::NotADecision { kind } => {
                write!(f, "a course is carried by a decision, and this is a {kind}")
            }
            Self::NameTaken { name, by } => write!(
                f,
                "`{name}` already names move {by}. A name is how a move is found again,                  so two of them answering to one word would be a move nobody can reach"
            ),
            Self::NoSuchName { name } => write!(f, "nothing here is called `{name}`"),
            Self::CannotCite { kind } => write!(
                f,
                "a {kind} is about moves and not about commits or trials: citing belongs \
                 to an attempt or a finding"
            ),
            Self::Crowded => write!(f, "too many people writing at once"),
        }
    }
}

impl std::error::Error for Trouble {}
