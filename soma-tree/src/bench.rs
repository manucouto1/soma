//! Everything a command needs standing up before it can ask anything.
//!
//! In the library and not in the binary because a request handler needs exactly
//! the same things a terminal command does, and two ways of finding the probe
//! or of deciding where answers are remembered would be two tools wearing one
//! name.

use crate::journal::Journal;
use crate::moves::Moves;
use crate::revision::{self, Worktree};
use crate::snapshot::{Probing, Snapshot};
use crate::walk::{self, Walk};
use serde::Deserialize;
use somatize_store::{Digest, Local};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// `soma-tree.toml`, at the root of the repository being explored.
#[derive(Deserialize)]
pub struct Config {
    /// `module:function` — takes nothing, returns a `Graph`.
    ///
    /// Optional, and not as a convenience: **reading an investigation's
    /// reasoning should not require knowing how to build its graph**. Finished
    /// work — a paper, a repository nobody runs, one from before soma — has
    /// reasoning worth reading and may have nothing to probe. Without it, what
    /// needs a probe says so and the rest works.
    #[serde(default)]
    pub build: Option<String>,
    /// The interpreter that can import somatize. Rarely the one on `PATH`.
    #[serde(default = "python_on_path")]
    pub python: PathBuf,
    /// What this investigation is called, so several can share one store
    /// without seeing each other. Defaults to the repository's own name.
    ///
    /// **It is in the name records are bound under**, which is the one part of
    /// this that cannot be changed later without moving somebody's directories.
    pub tree: Option<String>,
    /// Which way is better: `min` for a loss, `max` for an accuracy.
    ///
    /// Declared here because it is **not in the store**: the direction lives in
    /// the `Goal` handed to a sampler and is written in no record. Without it
    /// trials are shown by their range, which is true anyway.
    pub goal: Option<String>,
}

fn python_on_path() -> PathBuf {
    PathBuf::from("python3")
}

impl Config {
    /// How to build the graph, or why it cannot be.
    ///
    /// The message is for whoever is about to probe and not for whoever reads:
    /// somebody looking at the reasoning never gets here.
    pub fn building(&self) -> Result<&str, String> {
        self.build.as_deref().ok_or_else(|| {
            format!(
                "{} does not say what to build, so there is no graph to probe.\n\n    \
                 build = \"experiments.encoder:build\"\n\n\
                 The reasoning and the journal read the same without it.",
                "soma-tree.toml"
            )
        })
    }

    /// Read from the repository and not from the checkout: how an experiment
    /// is built is a fact about the project now, and reading it out of each
    /// commit would leave one predating the file unprobeable.
    pub fn read(repo: &Path) -> Result<Self, String> {
        let at = repo.join("soma-tree.toml");
        let text = std::fs::read_to_string(&at).map_err(|why| {
            format!(
                "{} could not be read: {why}\n\nIt says what to build:\n\n    \
                 build = \"experiments.encoder:build\"\n    python = \".venv/bin/python\"",
                at.display()
            )
        })?;
        toml::from_str(&text).map_err(|why| format!("{} is not readable: {why}", at.display()))
    }

    /// Which way is better, if it was declared. What is not understood is
    /// refused rather than read as *not declared*: a typo in `goal` would stop
    /// saying which was best with nothing saying why.
    pub fn towards(&self) -> Result<Option<crate::trials::Goal>, String> {
        match self.goal.as_deref() {
            None => Ok(None),
            Some(said) => crate::trials::Goal::read(said).map(Some).ok_or_else(|| {
                format!("`goal = \"{said}\"` does not say which way: `min` for a loss, `max` for an accuracy")
            }),
        }
    }

    pub fn tree(&self, repo: &Path) -> String {
        self.tree.clone().unwrap_or_else(|| {
            repo.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "tree".to_string())
        })
    }

    /// The interpreter, made absolute against the repo so that a relative
    /// `.venv/bin/python` still resolves once the probe runs in a worktree
    /// somewhere else entirely.
    pub fn interpreter(&self, repo: &Path) -> PathBuf {
        match self.python.is_absolute() || self.python.components().count() == 1 {
            true => self.python.clone(),
            false => repo.join(&self.python),
        }
    }
}

/// It owns what [`Probing`] borrows, which is the whole reason it is a struct:
/// the interpreter's path and the build's name outlive every commit asked
/// about, and threading them through each call said nothing.
pub struct Bench {
    pub repo: PathBuf,
    pub config: Config,
    pub remembering: Local,
    python: PathBuf,
    probe: PathBuf,
    recipe: Digest,
}

impl Bench {
    pub fn set_up(
        repo: &Path,
        store: Option<&Path>,
        given: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let repo = repo.canonicalize()?;
        let config = Config::read(&repo)?;
        let python = config.interpreter(&repo);
        let probe = probe_beside_the_binary()?;
        // The recipe identifies the probe and what it builds. With nothing to
        // build there is no probing, and an empty recipe is never used: it is
        // the path by which reasoning is read with nothing runnable.
        let recipe = match config.build.as_deref() {
            Some(build) => recipe(&probe, build, given)?,
            None => Digest::of(b""),
        };
        let remembering = Local::at(match store {
            Some(store) => store.to_path_buf(),
            None => where_probes_are_remembered(),
        })?;
        Ok(Self {
            repo,
            config,
            remembering,
            python,
            probe,
            recipe,
        })
    }

    pub fn probing<'a>(&'a self, store: Option<&'a Path>, given: Option<&'a Path>) -> Probing<'a> {
        Probing {
            python: &self.python,
            probe: &self.probe,
            build: self.config.build.as_deref().unwrap_or_default(),
            store,
            given,
            recipe: self.recipe.clone(),
        }
    }

    pub fn journal(&self) -> Journal<'_> {
        Journal::of(self.config.tree(&self.repo), &self.remembering)
    }

    pub fn moves(&self) -> Moves<'_> {
        Moves::of(self.config.tree(&self.repo), &self.remembering)
    }
}

/// A whole line of exploration, probed and compared and judged.
///
/// The one entry point both a terminal and a request handler use, so neither
/// can drift from the other about what an investigation contains.
pub fn walking(
    repo: &Path,
    store: Option<&Path>,
    given: Option<&Path>,
    range: &str,
    most: usize,
) -> Result<Walk, Box<dyn std::error::Error>> {
    let bench = Bench::set_up(repo, store, given)?;
    let probing = bench.probing(store, given);
    let shown = revision::commits_in(&bench.repo, range, most)?;
    if shown.is_empty() {
        return Err(format!("`{range}` names no commits").into());
    }
    // The range says what to show; every line in it needs the commit under it
    // to be compared against, and three branches have three of those.
    let mut commits = shown.clone();
    commits.extend(revision::beneath(&bench.repo, &shown));

    // With nothing to build nothing is probed, and that is not an error: the
    // history, the journal, the trials and the reasoning read the same. What
    // is missing is what each edit did.
    let known = match bench.config.build {
        Some(_) => probed(&bench, &probing, &commits)?,
        None => HashMap::new(),
    };
    walk::walked(
        &bench.repo,
        walk::Remembered {
            tree: &bench.config.tree(&bench.repo),
            kept: &bench.remembering,
            goal: bench.config.towards()?,
        },
        &probing,
        &shown,
        &commits,
        &known,
    )
}

/// The tree's name and the store its journal lives in.
pub fn journalling(
    repo: &Path,
    store: Option<&Path>,
) -> Result<(String, Local), Box<dyn std::error::Error>> {
    let repo = repo.canonicalize()?;
    let tree = Config::read(&repo)?.tree(&repo);
    let kept = Local::at(match store {
        Some(store) => store.to_path_buf(),
        None => where_probes_are_remembered(),
    })?;
    Ok((tree, kept))
}

/// A snapshot for every one of these commits.
///
/// Asked of the store first and of a checkout second. On a line somebody has
/// already looked at, this lays out no worktrees at all — which is the whole
/// reason a walk of ten commits is affordable.
pub fn probed<'a>(
    bench: &Bench,
    probing: &Probing,
    commits: &'a [String],
) -> Result<HashMap<&'a str, Snapshot>, Box<dyn std::error::Error>> {
    // Cut here, which is where the reason is known. Letting it through sent an
    // empty build to Python and came back `one of --build or --compare`, which
    // tells nobody that what is missing is a line in their soma-tree.toml.
    bench.config.building()?;
    let mut known: HashMap<&str, Snapshot> = HashMap::new();
    for commit in commits {
        if let Some(snapshot) = probing.recalled(&bench.remembering, commit) {
            known.insert(commit, snapshot);
        }
    }
    let missing: Vec<&String> = commits
        .iter()
        .filter(|commit| !known.contains_key(commit.as_str()))
        .collect();
    if missing.is_empty() {
        return Ok(known);
    }

    eprintln!(
        "probing {} of {} commits; {} were already known",
        missing.len(),
        commits.len(),
        commits.len() - missing.len(),
    );
    let laid_out = tempfile::tempdir()?;
    let trees: Vec<Worktree> = missing
        .iter()
        .enumerate()
        .map(|(n, commit)| Worktree::of(&bench.repo, commit, laid_out.path(), &n.to_string()))
        .collect::<Result<_, _>>()?;

    // Threads and not tasks: what this waits on is a Python interpreter
    // importing torch, which is somebody else's CPU and not an idle socket.
    // There is nothing here for an executor to interleave.
    let fresh = std::thread::scope(|scope| {
        let remembering = &bench.remembering;
        let asking: Vec<_> = trees
            .iter()
            .map(|tree| {
                scope.spawn(move || probing.remembered(remembering, tree.path(), tree.commit()))
            })
            .collect();
        asking
            .into_iter()
            .map(|one| one.join().expect("a probe thread does not panic"))
            .collect::<Vec<_>>()
    });
    for (commit, snapshot) in missing.iter().zip(fresh) {
        known.insert(commit, snapshot?);
    }
    Ok(known)
}

/// Where a probe's answer is kept when nobody said where.
///
/// A cache and not a store of record: it holds only what can be worked out
/// again from a commit, so deleting it costs time and nothing else.
pub fn where_probes_are_remembered() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("somatize-tree")
}

/// What a remembered answer depends on, other than the commit.
///
/// The probe's **own source** is in here, and that is the point: a snapshot is
/// a pure function of a commit only for a fixed probe.
fn recipe(probe: &Path, build: &str, given: Option<&Path>) -> Result<Digest, String> {
    let source = std::fs::read(probe).map_err(|why| format!("{}: {why}", probe.display()))?;
    let input = match given {
        Some(given) => std::fs::read(given).map_err(|why| format!("{}: {why}", given.display()))?,
        None => b"sentinel".to_vec(),
    };
    Ok(Digest::of(
        &[&source[..], build.as_bytes(), &input[..]].concat(),
    ))
}

/// The probe, which ships beside the binary and not inside the repo being
/// explored: it is this tool's, and a project should not have to vendor it.
fn probe_beside_the_binary() -> Result<PathBuf, String> {
    let beside = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|at| at.join("soma_tree_probe.py")))
        .filter(|at| at.exists());
    let developing = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/soma_tree_probe.py");
    beside
        .or_else(|| developing.exists().then_some(developing))
        .ok_or_else(|| "soma_tree_probe.py is not beside the binary".to_string())
}
