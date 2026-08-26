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
        // The recipe identifies the probe and what it builds. With nothing to
        // build there is no probing, and an empty recipe is never used: it is
        // the path by which reasoning is read with nothing runnable.
        let recipe = match config.build.as_deref() {
            Some(build) => recipe(build, given)?,
            None => Digest::of(b""),
        };
        // Laid down whatever the config says: `prettified` and `compared` ask
        // the probe without needing a `build`, and this costs one write, once
        // ever, into a directory `Local::at` below already has to be able to
        // create `blobs/` in.
        let probe = probe_laid_down(&where_probes_are_remembered())?;
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
/// The bench is handed in and not built here.
///
/// It used to take the paths and stand one up of its own, which quietly made
/// this the second place that reads `soma-tree.toml` — so a name said on the
/// command line reached the journal and not the walk, and a verdict written
/// one moment was invisible the next with nothing saying why.
pub fn walking(
    bench: &Bench,
    store: Option<&Path>,
    given: Option<&Path>,
    range: &str,
    most: usize,
) -> Result<Walk, Box<dyn std::error::Error>> {
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
        Some(_) => probed(bench, &probing, &commits)?,
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

/// How many probes run at once.
///
/// Not the core count, which is the number this looks like it should be. What a
/// probe holds is an interpreter with the checkout's own `somatize` imported,
/// and that is torch: a quarter of a gigabyte, each. So the bound is memory and
/// it does not grow with the machine — twenty cores would ask for five
/// gigabytes to walk one line, and `cargo test` runs twenty of *those* at once.
const AT_ONCE: usize = 4;

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
    //
    // A pool of `AT_ONCE` and not a thread per commit, and what the pool shares
    // is an index rather than a chunk each: a commit that takes a minute holds
    // up nothing behind it. Answers come back by position, because what names a
    // snapshot here is the revspec that was asked for and a probe only knows
    // the hash it resolved to.
    use std::sync::atomic::{AtomicUsize, Ordering};
    // Both outlive the scope on purpose: a worker borrows them, so declaring
    // them inside would be lending what is about to go out of scope.
    let next = AtomicUsize::new(0);
    let (tell, heard) = std::sync::mpsc::channel();
    let fresh = std::thread::scope(|scope| {
        let remembering = &bench.remembering;
        for _ in 0..AT_ONCE.min(trees.len()) {
            let (tell, next, trees) = (tell.clone(), &next, &trees);
            scope.spawn(move || {
                loop {
                    let n = next.fetch_add(1, Ordering::Relaxed);
                    let Some(tree) = trees.get(n) else { break };
                    let said = probing.remembered(remembering, tree.path(), tree.commit());
                    // The receiver is this scope, which outlives every worker.
                    let _ = tell.send((n, said));
                }
            });
        }
        // Or the collector waits on a sender nobody is holding.
        drop(tell);
        let mut fresh: Vec<Option<_>> = trees.iter().map(|_| None).collect();
        for (n, said) in heard {
            fresh[n] = Some(said);
        }
        fresh
    });
    let fresh = fresh
        .into_iter()
        .map(|said| said.expect("every commit laid out was probed"));
    for (commit, snapshot) in missing.iter().zip(fresh) {
        known.insert(commit, snapshot?);
    }
    Ok(known)
}

/// Where a probe's answer is kept when nobody said where.
///
/// A cache and not a store of record: it holds only what can be worked out
/// again from a commit, so deleting it costs time and nothing else.
/// The probe itself, compiled in.
///
/// **Not found at run time, because there was nowhere honest to look.** It used
/// to be sought beside the binary and then, failing that, at the
/// `CARGO_MANIFEST_DIR` of whoever compiled it — so a `cargo install` left a
/// binary depending on a registry checkout it does not own, and copying the
/// file beside the executable was a step nobody performs.
///
/// And it does not go in the wheel either, which is the other tempting answer:
/// the probe belongs to **this tool** while `somatize` belongs to the checkout
/// being explored, so `python -m somatize.tree.probe` would run the explored
/// project's probe against its own graph and quietly answer a different
/// question.
const PROBE: &str = include_str!("soma_tree_probe.py");

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
fn recipe(build: &str, given: Option<&Path>) -> Result<Digest, String> {
    let input = match given {
        Some(given) => std::fs::read(given).map_err(|why| format!("{}: {why}", given.display()))?,
        None => b"sentinel".to_vec(),
    };
    Ok(Digest::of(
        &[PROBE.as_bytes(), build.as_bytes(), &input[..]].concat(),
    ))
}

/// The compiled-in probe, on disk under `cache`, because `python` is handed a
/// path.
///
/// Named by its own digest under the cache this tool already owns, so it is
/// written once and can never be stale: a probe that changed is a different
/// name and the old file is simply not asked for. Three of the four ways the
/// probe is called pass it as `argv[1]`, and the fourth is already using stdin
/// for the source it formats, so there is no reading it from a pipe.
///
/// Keeping the name in the file keeps it in a traceback, where somebody
/// debugging their own `build()` will read it — and a file under a cache is
/// still there when they go and look, which a temporary directory is not.
///
/// The cache is a parameter and not read from the environment in here: it is
/// **this tool's** and never the store `--store` points at, and saying which of
/// the two it is at the call site is the whole difference.
pub fn probe_laid_down(cache: &Path) -> Result<PathBuf, String> {
    let digest = Digest::of(PROBE.as_bytes());
    let hex = digest
        .as_str()
        .rsplit(':')
        .next()
        .unwrap_or(digest.as_str());
    let at = cache.join("probe");
    let laid = at.join(format!("soma_tree_probe-{hex}.py"));
    if laid.exists() {
        return Ok(laid);
    }
    std::fs::create_dir_all(&at).map_err(|why| format!("{}: {why}", at.display()))?;
    // Written beside and moved into place: two walks starting at once must not
    // hand `python` a file that is half there.
    let landing = at.join(format!("soma_tree_probe-{hex}.{}.py", std::process::id()));
    std::fs::write(&landing, PROBE).map_err(|why| format!("{}: {why}", landing.display()))?;
    std::fs::rename(&landing, &laid).map_err(|why| format!("{}: {why}", laid.display()))?;
    Ok(laid)
}
