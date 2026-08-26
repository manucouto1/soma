//! What a graph was, at one commit. The probe's answer, typed.
//!
//! Nothing here holds a graph — a graph is Python and lives for the length of a
//! subprocess. What crosses back is this, and the fields are the probe's to
//! add: `src/soma_tree_probe.py` is the contract, and this is one reader of
//! it.

use crate::findings::Findings;
use serde::Deserialize;
use somatize_store::{Digest, Meta, Store};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;
use std::process::Command;

/// A graph as one checkout had it.
#[derive(Debug, Deserialize, serde::Serialize)]
pub struct Snapshot {
    /// What checkout this was, for saying so afterwards.
    pub commit: String,
    /// The `module:function` that built it.
    pub built_from: String,
    /// `"sentinel"` when no real input was hashed. Two snapshots are only
    /// comparable if they were taken the same way: the names come out of the
    /// snapshot, so one taken with an input against one taken without has
    /// **everything** moved and nothing saying why.
    pub input: String,
    /// What the graph was built against: the interpreter, and the version of
    /// every distribution it reached for.
    ///
    /// The axis git does not cover — a checkout pins its own code and not the
    /// interpreter outside it — and deliberately **not** part of the recipe a
    /// snapshot is remembered under. Two probes months apart are meant to
    /// disagree here out loud.
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    /// `foreseen.snapshot`'s own answer, carried **opaque**.
    ///
    /// Never read on this side and never reshaped: what a name is made of is
    /// the model's business, and a reader here that understood its insides
    /// would be a second model with a delay on it.
    pub snapshot: serde_json::Value,
    /// The class behind each node: where it lives and what it says.
    ///
    /// Read while the graph existed, because a snapshot outlives the process
    /// that made it — and the checkout it was read from is a worktree that was
    /// removed minutes later.
    #[serde(default)]
    pub code: BTreeMap<String, Written>,
    /// What each node is made of inside: `{node: [piece, ...]}`.
    ///
    /// A node is a box and what it holds is usually what the experiment is
    /// about — `Pure` is a wrapper and the router inside it is the piece — so
    /// drawing the box and saying nothing about the inside draws the wrapper.
    ///
    /// Read without running anything, so it is the **declared** composition:
    /// what `__init__` built. `somatize.torch.architecture` answers better —
    /// it sees what is not a module — and executes the graph to do it, which
    /// is what this side never does. Opaque, like the rest.
    #[serde(default)]
    pub inside: serde_json::Value,
    /// What **files** each node is made of, and where the count stops.
    ///
    /// `code` shows one class, which is what somebody clicking a node wants.
    /// But a network is often written across four modules joined in an
    /// `__init__`, and `inspect.getsourcefile` knows only one of the four.
    ///
    /// Not a second model of what depends on what: it is the transitive closure
    /// soma's fingerprint already walked in order to hash it, said out loud, so
    /// it moves when what goes into a fingerprint moves.
    ///
    /// **No source inside**, on purpose: forty commits of whole files are
    /// nearly all of the answer and none of it read. What is here are paths,
    /// and the content is asked for by its own when somebody opens one.
    #[serde(default)]
    pub reaches: serde_json::Value,
    /// The orthogonal facts of a graph beside what each node computes: who
    /// implements it, where it runs, on which device, what is kept, what is
    /// frozen, and in what order it would run.
    ///
    /// Opaque like `snapshot`: the vocabulary is soma's, and all that is needed
    /// here is that it reaches whoever draws intact.
    #[serde(default)]
    pub architecture: serde_json::Value,
    /// The code that **declares** the graph: the body of `build`, with the
    /// `>>` and the `|`.
    ///
    /// The one part of a graph that cannot be read node by node — each class
    /// says what it does and none says how they connect — so without it the
    /// topology is only ever seen drawn and never written.
    ///
    /// `None` when there is no source to read, which is the absence
    /// `UNVERSIONED` names a level below.
    #[serde(default)]
    pub declaring: Option<Written>,
    /// The nodes named by the content of their items, which nobody has before
    /// a run. Carried so a report can say *cannot tell* out loud.
    #[serde(default)]
    pub mapped: Vec<String>,
    /// What would not have to run at all, because something under it is kept.
    /// Empty without a real store, and that is not the same as "nothing".
    #[serde(default)]
    pub unneeded: Vec<String>,
}

impl Snapshot {
    /// What each node's answer will be called, `{node: key}`.
    ///
    /// Reading inside the opaque `snapshot` is the one thing this side does not
    /// do — what a name is made of is the model's business — and this is not
    /// that: using a name **as a name**, to look it up in a store, is what the
    /// model publishes it for. Decomposing a key to get something out of it
    /// would be the other thing, and it would live in `foreseen`.
    ///
    /// Nodes are missing and it is not an oversight: a `.mapped()` is named by
    /// the content of its items, which nobody has before a run. That absence
    /// reads *cannot tell* and never *no data*.
    pub fn names(&self) -> BTreeMap<String, String> {
        read_map(&self.snapshot, "names")
    }

    /// What version of the code each node had, `{node: fingerprint}`.
    ///
    /// The side of attribution that survives what the other does not: a key is
    /// computed against the probing interpreter's environment, so probing a
    /// three-month-old commit today gives keys matching nothing kept then,
    /// while the fingerprint was written beside the value by whoever ran.
    pub fn fingerprints(&self) -> BTreeMap<String, String> {
        read_map(&self.snapshot, "fingerprints")
    }

    /// What was built differently around the two of them: name, before, after.
    ///
    /// Usually nothing, because two commits probed in one sitting share an
    /// interpreter. It is a cached probe from months ago against a fresh one
    /// that answers something here — which is the whole reason the environment
    /// is left out of what a snapshot is remembered under.
    pub fn drifted_from<'a>(&'a self, other: &'a Self) -> Vec<(&'a str, String, String)> {
        let absent = "—".to_string();
        let mut said: Vec<(&str, String, String)> = Vec::new();
        for name in self.environment.keys().chain(other.environment.keys()) {
            let (was, is) = (self.environment.get(name), other.environment.get(name));
            if was != is && !said.iter().any(|(said, _, _)| said == name) {
                said.push((
                    name.as_str(),
                    was.cloned().unwrap_or(absent.clone()),
                    is.cloned().unwrap_or(absent.clone()),
                ));
            }
        }
        said
    }
}

/// A `{text: text}` from inside the model's answer, or nothing.
///
/// Nothing and not a failure: an old probe, kept before the model published
/// this field, is still a good answer to everything else. Falling over would
/// throw away an investigation's record for a function added afterwards.
fn read_map(said: &serde_json::Value, what: &str) -> BTreeMap<String, String> {
    said.get(what)
        .and_then(|found| found.as_object())
        .map(|found| {
            found
                .iter()
                .filter_map(|(node, told)| Some((node.clone(), told.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// One node's class, as it was at that commit.
#[derive(Debug, Deserialize, serde::Serialize)]
pub struct Written {
    /// Relative to the checkout. `None` for a class with no file behind it.
    pub file: Option<String>,
    /// Where the class starts, so an editor can open at it.
    pub line: u32,
    /// `None` when it is long enough that reading it is opening a file, not
    /// glancing at a panel.
    pub source: Option<String>,
    pub lines: u32,
}

/// Everything a probe needs that is the same for every commit it is asked
/// about. Held together so a call says only what varies: the checkout.
pub struct Probing<'a> {
    pub python: &'a Path,
    pub probe: &'a Path,
    pub build: &'a str,
    /// Handed to the probe so it can say what is already computed. Not the
    /// store snapshots are remembered in, though it is usually the same one.
    pub store: Option<&'a Path>,
    pub given: Option<&'a Path>,
    /// What identifies this probing, everything but the commit: the build, the
    /// input, and **the probe's own source**.
    ///
    /// That last is not belt and braces. A snapshot is a pure function of a
    /// commit only *given a fixed probe*, so the day `declared` learned to read
    /// an object's attributes, every snapshot taken before it became wrong — in
    /// exactly the way this tool exists to catch.
    pub recipe: Digest,
}

impl Probing<'_> {
    /// The name this checkout's snapshot is kept under. Content-addressed and
    /// immutable: a commit does not change, so neither does the answer.
    pub fn named(&self, commit: &str) -> String {
        format!("snapshot:{commit}:{}", self.recipe)
    }

    /// What was already probed for this commit, without touching a checkout.
    ///
    /// Its own method because a walk asks this of **every** commit first and
    /// only then lays out the ones nobody has an answer for. On a line of
    /// exploration that has been looked at once, that is no worktrees at all.
    pub fn recalled(&self, kept: &dyn Store, commit: &str) -> Option<Snapshot> {
        match recall(kept, &self.named(commit)) {
            Ok(snapshot) => snapshot,
            Err(why) => {
                eprintln!("what was already probed could not be looked up: {why}");
                None
            }
        }
    }

    /// The snapshot for this checkout, from the store if it is there.
    ///
    /// A store that cannot answer is **not** the end of it, exactly as a keeper
    /// that cannot answer is not the end of a run: the probe is asked instead
    /// and the trouble is said out loud. A cache gone cold is slow; a cache that
    /// stops the tool is broken.
    pub fn remembered(
        &self,
        kept: &dyn Store,
        working: &Path,
        commit: &str,
    ) -> Result<Snapshot, Trouble> {
        let name = self.named(commit);
        match recall(kept, &name) {
            Ok(Some(snapshot)) => return Ok(snapshot),
            Ok(None) => {}
            Err(why) => eprintln!("what was already probed could not be looked up: {why}"),
        }

        let (snapshot, bytes) = self.taken(working, commit)?;
        if let Err(why) = keep(kept, &name, &bytes, commit, self.build) {
            eprintln!("this probe could not be kept for next time: {why}");
        }
        Ok(snapshot)
    }

    /// What the edit did, for each of these pairs of `(older, newer)`.
    ///
    /// Pairs and not an ordered list, because **a step is an edge**: two
    /// entries next to each other in a walk of three branches are two different
    /// lines of exploration, and comparing them would answer confidently about
    /// an edit nobody made.
    ///
    /// One subprocess for the whole walk — comparing needs no checkout and no
    /// graph, only the model and the snapshots. The model is
    /// `somatize.foreseen`'s and nothing here decides what a finding means.
    pub fn compared(
        &self,
        taken: &HashMap<&str, Snapshot>,
        pairs: &[(String, String)],
    ) -> Result<Vec<Findings>, Trouble> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let garbled = |why: serde_json::Error| Trouble::Garbled {
            commit: "comparing".into(),
            why: why.to_string(),
        };
        let asked = serde_json::json!({"snapshots": taken, "pairs": pairs})
            .to_string()
            .into_bytes();
        let written = tempfile::tempdir().map_err(|why| Trouble::Garbled {
            commit: "comparing".into(),
            why: why.to_string(),
        })?;
        let at = written.path().join("asked.json");
        std::fs::write(&at, &asked).map_err(|why| Trouble::Garbled {
            commit: "comparing".into(),
            why: why.to_string(),
        })?;

        let said = Command::new(self.python)
            .arg(self.probe)
            .arg("--compare")
            .arg(&at)
            .output()
            .map_err(|why| Trouble::Unreachable {
                python: self.python.display().to_string(),
                why: why.to_string(),
            })?;
        if !said.status.success() {
            return Err(Trouble::Refused {
                commit: "comparing".into(),
                said: String::from_utf8_lossy(&said.stderr).trim().to_string(),
            });
        }
        serde_json::from_slice(&said.stdout).map_err(garbled)
    }

    /// Whether an edit survives: it parses, a linter is quiet, the graph still
    /// builds, and the node runs on what its predecessors left in the store.
    ///
    /// Asked in a checkout that is **the same tree a fork would commit**, so a
    /// green light is about the thing that would land and not about something
    /// near it.
    pub fn checked(&self, working: &Path, node: &str) -> Result<serde_json::Value, Trouble> {
        let mut asking = Command::new(self.python);
        asking
            .arg(self.probe)
            .arg("--build")
            .arg(self.build)
            .arg("--check")
            .arg(node)
            .current_dir(working);
        if let Some(store) = self.store {
            asking.arg("--store").arg(store);
        }
        if let Some(given) = self.given {
            asking.arg("--input").arg(given);
        }
        let said = asking.output().map_err(|why| Trouble::Unreachable {
            python: self.python.display().to_string(),
            why: why.to_string(),
        })?;
        if !said.status.success() {
            return Err(Trouble::Refused {
                commit: node.to_string(),
                said: String::from_utf8_lossy(&said.stderr).trim().to_string(),
            });
        }
        serde_json::from_slice(&said.stdout).map_err(|why| Trouble::Garbled {
            commit: node.to_string(),
            why: why.to_string(),
        })
    }

    /// The same source, formatted — or the same source back and a reason.
    ///
    /// `ruff format` if this environment has one. Refused rather than
    /// half-done when it does not: handing back something that looks formatted
    /// and is not is worse than a button that says it cannot.
    pub fn prettified(&self, source: &str) -> Result<serde_json::Value, Trouble> {
        use std::io::Write as _;
        let mut running = Command::new(self.python)
            .arg(self.probe)
            .args(["--build", "x:y", "--format"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|why| Trouble::Unreachable {
                python: self.python.display().to_string(),
                why: why.to_string(),
            })?;
        let asked = |why: String| Trouble::Garbled {
            commit: "formatting".into(),
            why,
        };
        running
            .stdin
            .take()
            .ok_or_else(|| asked("no stdin".into()))?
            .write_all(source.as_bytes())
            .map_err(|why| asked(why.to_string()))?;
        let said = running
            .wait_with_output()
            .map_err(|why| asked(why.to_string()))?;
        serde_json::from_slice(&said.stdout).map_err(|why| asked(why.to_string()))
    }

    /// Runs the probe in a checkout and reads what it wrote, with the bytes it
    /// wrote — which are what gets kept.
    ///
    /// A subprocess and not a library call, and it is why this tool has two
    /// languages in it: the graph only exists once the checkout's own code has
    /// been imported and run, against the soma *that* checkout pins. Reaching
    /// into it from here would run one version of the engine over another
    /// version's declarations.
    pub fn taken(&self, working: &Path, commit: &str) -> Result<(Snapshot, Vec<u8>), Trouble> {
        let mut asking = Command::new(self.python);
        asking
            .arg(self.probe)
            .arg("--build")
            .arg(self.build)
            .arg("--commit")
            .arg(commit)
            .current_dir(working);
        if let Some(store) = self.store {
            asking.arg("--store").arg(store);
        }
        if let Some(given) = self.given {
            asking.arg("--input").arg(given);
        }
        let said = asking.output().map_err(|why| Trouble::Unreachable {
            python: self.python.display().to_string(),
            why: why.to_string(),
        })?;
        if !said.status.success() {
            return Err(Trouble::Refused {
                commit: commit.to_string(),
                said: String::from_utf8_lossy(&said.stderr).trim().to_string(),
            });
        }
        let snapshot = serde_json::from_slice(&said.stdout).map_err(|why| Trouble::Garbled {
            commit: commit.to_string(),
            why: why.to_string(),
        })?;
        Ok((snapshot, said.stdout))
    }
}

/// What is kept under that name, if anything readable is.
fn recall(kept: &dyn Store, name: &str) -> Result<Option<Snapshot>, String> {
    let Some(bound) = kept.resolve(name).map_err(|why| why.to_string())? else {
        return Ok(None);
    };
    let Some(bytes) = kept.get(&bound.digest).map_err(|why| why.to_string())? else {
        // Named, but the blob is not there. That is what a half-copied store
        // looks like and not a reason to stop: the probe still works.
        return Ok(None);
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|why| why.to_string())
}

/// Puts a probe's answer where the next one will find it.
fn keep(
    kept: &dyn Store,
    name: &str,
    bytes: &[u8],
    commit: &str,
    build: &str,
) -> Result<(), String> {
    let digest = kept.put(bytes).map_err(|why| why.to_string())?;
    // Said beside it so that a scan of the store reads as something. The
    // records are the truth, and any index over them is built from these.
    let meta: Meta = vec![
        ("what".into(), "snapshot".into()),
        ("commit".into(), commit.into()),
        ("built_from".into(), build.into()),
    ];
    kept.bind(name, &digest, meta)
        .map_err(|why| why.to_string())
}

/// What can go wrong between here and a graph.
#[derive(Debug)]
pub enum Trouble {
    Unreachable { python: String, why: String },
    Refused { commit: String, said: String },
    Garbled { commit: String, why: String },
}

impl fmt::Display for Trouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // The commonest failure by far, and worth naming precisely: the
            // interpreter that can import somatize is rarely the one on PATH.
            Self::Unreachable { python, why } => {
                write!(f, "`{python}` could not be run: {why}")
            }
            Self::Refused { commit, said } => {
                write!(f, "building the graph at {commit} failed:\n{said}")
            }
            Self::Garbled { commit, why } => {
                write!(f, "the probe at {commit} said something unreadable: {why}")
            }
        }
    }
}

impl std::error::Error for Trouble {}
