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
    /// Opcional, y esto no es una comodidad: **leer el razonamiento de una
    /// investigación no debería exigir saber construir su grafo**. Un trabajo
    /// terminado —un paper, un repositorio que ya nadie ejecuta, uno anterior a
    /// soma— tiene un razonamiento que vale la pena leer y puede no tener nada
    /// que sondear. Exigirlo aquí ataba la capa 2 a la 1 por el sitio
    /// equivocado: por la configuración, no por los hechos.
    ///
    /// Sin él, lo que necesita sondear dice que falta y lo demás funciona.
    #[serde(default)]
    pub build: Option<String>,
    /// The interpreter that can import soma-next. Rarely the one on `PATH`.
    #[serde(default = "python_on_path")]
    pub python: PathBuf,
    /// What this investigation is called, so several can share one store
    /// without seeing each other. Defaults to the repository's own name.
    ///
    /// **It is in the name records are bound under**, which is the one part of
    /// this that cannot be changed later without moving somebody's directories.
    pub tree: Option<String>,
    /// Hacia dónde es mejor: `min` para una pérdida, `max` para una exactitud.
    ///
    /// Se declara aquí porque **no está en el store**: la dirección vive en el
    /// `Goal` que se le pasa al sampler y no se escribe en ningún registro. Sin
    /// ella los ensayos se enseñan por su rango, que es cierto de todos modos.
    /// Con ella se puede decir cuál fue el mejor, que es distinto de decirlo
    /// suponiéndolo.
    pub goal: Option<String>,
}

fn python_on_path() -> PathBuf {
    PathBuf::from("python3")
}

impl Config {
    /// Read from the repository, not from the checkout: how an experiment is
    /// built is a fact about the project now, and reading it out of each commit
    /// would mean an old one that predates the file cannot be probed.
    /// Cómo construir el grafo, o por qué no se puede.
    ///
    /// El mensaje es del que va a sondear y no del que lee: quien mira el
    /// razonamiento nunca llega aquí.
    pub fn building(&self) -> Result<&str, String> {
        self.build.as_deref().ok_or_else(|| {
            format!(
                "{} no dice qué construir, así que no hay grafo que sondear.\n\n    \
                 build = \"experiments.encoder:build\"\n\n\
                 El razonamiento y el diario se leen igual sin esto.",
                "soma-tree.toml"
            )
        })
    }

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

    /// Hacia dónde es mejor, si se declaró. Se rechaza lo que no se entienda
    /// en vez de leerlo como «no declarado»: una errata en `goal` dejaría de
    /// decir cuál fue el mejor sin que nada avisara de por qué.
    pub fn towards(&self) -> Result<Option<crate::trials::Goal>, String> {
        match self.goal.as_deref() {
            None => Ok(None),
            Some(said) => crate::trials::Goal::read(said).map(Some).ok_or_else(|| {
                format!("`goal = \"{said}\"` no dice hacia dónde: `min` para una pérdida, `max` para una exactitud")
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
        // La receta identifica al sondeador y a lo que construye. Sin nada que
        // construir no hay sondeo, y una receta vacía no se usa nunca: es el
        // camino por el que se lee un razonamiento sin poder ejecutar nada.
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

    // Sin nada que construir no se sondea nada, y no es un error: la historia,
    // el diario, los ensayos y el razonamiento se leen igual. Lo único que
    // falta es qué hizo cada edición.
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
    // Cortado aquí, que es donde se sabe por qué. Dejarlo pasar mandaba una
    // construcción vacía a Python y volvía «one of --build or --compare», que
    // no le dice a nadie que lo que falta es una línea en su soma-tree.toml.
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
