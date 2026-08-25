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
    /// anything down.
    ///
    /// Note what it deliberately does **not** reach: forking off an abandoned
    /// attempt makes a sibling, not a child, so the new line starts clean.
    /// Trying something else after hitting a dead end is the move you make
    /// *because* it was a dead end, and inheriting the abandonment down git
    /// ancestry would mark it as more of the same.
    pub decided: Option<Course>,
    /// Whether something above it is [`Verdict::Invalid`]. Worked out from git
    /// rather than stored, so a commit made after the verdict is marked the
    /// moment it exists.
    pub doubted: bool,
    /// Lo que se corrió con esta versión: cuántos ensayos y cómo van.
    ///
    /// Un commit es la versión y no cambia; los ensayos crecen sin parar y van
    /// **asociados** a ella, no versionados. Vienen del mismo recorrido que
    /// todo lo demás, porque soma puso el estado y la puntuación en el
    /// registro: contar cuarenta versiones cuesta un recorrido, y sólo la
    /// curva se paga aparte.
    pub trials: Tally,
    /// Si se pliega al dibujar: está en una línea que alguien decidió
    /// abandonar o dar por superada, y nadie le ha encontrado nada malo.
    ///
    /// **Podar es dejar de dibujar y nunca borrar.** Esta parada sigue viniendo
    /// entera —con su diario, sus ensayos y su paso—: lo único que dice esto es
    /// que un árbol de cuarenta variantes no se lee, y que quien dibuja puede
    /// plegar ésta si quiere. Quien procesa la respuesta la ignora.
    ///
    /// Viene calculado y no lo calcula quien dibuja, porque si no la regla
    /// viviría en dos idiomas: el día que cambiara, el terminal y la vista
    /// plegarían cosas distintas y las dos parecerían correctas.
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

/// Si una parada se pliega al dibujar una línea podada.
///
/// La regla entera, en un sitio, porque la usan el terminal y la vista: si cada
/// uno la escribiera en su idioma, el día que cambiara plegarían cosas
/// distintas y las dos parecerían correctas.
///
/// Se pliega lo que alguien decidió abandonar o dar por superado. **No se
/// pliega lo que alguien ha juzgado mal**: un commit `invalid` es lo que pone
/// en duda la medida en la que se apoyó la decisión de abandonar la línea, y
/// esconderlo sería esconder justo la razón para volver a mirarla. Lo mismo
/// para el que hereda esa duda, que es la misma razón un nivel más abajo.
///
/// Un `sound` sí se pliega: dice que se miró y no había nada malo, así que no
/// hay ninguna razón nueva para volver — la decisión sigue en pie.
pub fn folds(decided: Option<Course>, judged: Option<Verdict>, doubted: bool) -> bool {
    if !matches!(decided, Some(Course::Abandon) | Some(Course::Superseded)) {
        return false;
    }
    !doubted && !matches!(judged, Some(Verdict::Invalid))
}

/// Lo que hace falta para leer lo que ya se sabe de una investigación: cómo se
/// llama, dónde está guardada y hacia dónde es mejor.
///
/// Las tres viajan juntas porque las tres salen del mismo `soma-tree.toml` y
/// ninguna significa nada sin las otras: un store sin el nombre del árbol
/// devuelve los registros de otra investigación, y una puntuación sin la
/// dirección no dice si es buena.
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
    // Sondear es opcional. Un repositorio anterior a soma —un paper terminado,
    // un trabajo que ya nadie ejecuta— tiene una historia, un diario, unos
    // ensayos y un razonamiento que valen la pena leer, y ningún grafo que
    // sondear. Sin sondeo hay paradas y no hay pasos: lo que falta es **qué
    // hizo cada edición**, y sólo eso.
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

    // A verdict is written about **one** commit. That its descendants are
    // suspect is worked out here and not stored, which is why a commit made
    // tomorrow under an invalid one needs nobody to go back and say so.
    //
    // Walked over the parents already in hand rather than asked of git: an
    // ancestry-path question needs a tip to walk **towards**, and with three
    // branches the tip is usually on somebody else's — so a verdict cast on
    // one variant would quietly reach nothing at all.
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

    // Que no se pueda leer el razonamiento no es razón para no dibujar el
    // registro: un árbol sin decisiones es exactamente lo que hay al empezar.
    let decided = Moves::of(tree, kept).decided().unwrap_or_default();
    // Y lo mismo: no poder contar lo que se corrió no es razón para no dibujar
    // lo que se escribió.
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
            // Lo que ha juzgado alguien no se pliega nunca. Un commit `invalid`
            // es lo que pone en duda la medida en la que se apoyó la decisión
            // de abandonar la línea, y esconderlo sería esconder justo la razón
            // para volver a mirarla; y uno que hereda esa duda, igual.
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
        // Sin sondeo no hay de qué se construyó: se dice, en vez de dejar el
        // hueco vacío pareciendo un fallo.
        built_from: if probing.is_none() {
            "sin sondeo — este repositorio no declara qué construir".to_string()
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
