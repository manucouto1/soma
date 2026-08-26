//! A commit, put somewhere it can be imported from.
//!
//! `git` as a subprocess and not `gix`, for now. What is needed here is
//! `rev-parse` and a worktree, both of which the binary already does correctly,
//! and a library earns its place when there is work it does better — reading
//! many blobs, or naming a commit's contents without materialising them. There
//! is none of that yet.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A checkout of one commit, removed when it goes out of scope.
pub struct Worktree {
    repo: PathBuf,
    at: PathBuf,
    /// The full hash. `HEAD~2` is not something to print back at somebody.
    commit: String,
}

impl Worktree {
    /// Resolves a revspec and lays that commit out under `beneath`, in a
    /// directory called `as_`.
    ///
    /// The name is the caller's and not the commit's, because the two sides of
    /// a comparison are allowed to be the **same** commit — asking what a store
    /// already holds is a diff of one revision against itself — and two
    /// worktrees of one commit under one name is a refusal from git. Detached
    /// for the same reason: a branch checked out twice is another one.
    pub fn of(repo: &Path, revspec: &str, beneath: &Path, as_: &str) -> Result<Self, Trouble> {
        let commit = git(
            repo,
            &["rev-parse", "--verify", &format!("{revspec}^{{commit}}")],
        )
        .map_err(|said| Trouble::NoSuchRevision {
            revspec: revspec.to_string(),
            said,
        })?;
        let at = beneath.join(as_);
        git(
            repo,
            &[
                "worktree",
                "add",
                "--detach",
                "--quiet",
                &at.display().to_string(),
                &commit,
            ],
        )
        .map_err(|said| Trouble::NoWorktree {
            commit: commit.clone(),
            said,
        })?;
        Ok(Self {
            repo: repo.to_path_buf(),
            at,
            commit,
        })
    }

    pub fn path(&self) -> &Path {
        &self.at
    }

    /// The short hash, which is what a person reads.
    pub fn named(&self) -> &str {
        &self.commit[..12.min(self.commit.len())]
    }

    /// The whole hash, which is what a cache is keyed on: twelve characters is
    /// plenty to read and too few to name a stored answer after.
    pub fn commit(&self) -> &str {
        &self.commit
    }
}

impl Drop for Worktree {
    fn drop(&mut self) {
        // A worktree left behind is not just a directory: git keeps a record
        // of it and the next `worktree add` on the same commit refuses. Said
        // out loud, because the fix is `git worktree prune` and nobody guesses
        // that.
        let removing = git(
            &self.repo,
            &[
                "worktree",
                "remove",
                "--force",
                &self.at.display().to_string(),
            ],
        );
        if let Err(said) = removing {
            eprintln!(
                "the worktree at {} could not be removed: {said}\n\
                 `git worktree prune` in {} clears the record it left.",
                self.at.display(),
                self.repo.display(),
            );
        }
    }
}

/// What to ask for when what is wanted is the whole investigation.
pub const ALL: &str = "--all";

/// The commits to walk, newest first — git's own order.
///
/// Two ways of asking, because there are two questions:
///
/// - a **range**, `main~10..main`, which is exactly what somebody meant;
/// - a **revspec**, `HEAD`, meaning the history back from there, capped —
///   which is what `git log` does, and the only form that works on a
///   repository whose length nobody knows;
/// - [`ALL`], every branch, which is the default because that is the shape an
///   investigation has.
///
/// The second exists because the first has a trap: `HEAD~10..HEAD` in a
/// repository with four commits is not an empty answer, it is
/// *revisión desconocida*. A default that asks for ten commits cannot be a
/// range, and being told off by git for it is not something to hand a person
/// who typed nothing at all.
pub fn commits_in(repo: &Path, asked: &str, most: usize) -> Result<Vec<String>, Trouble> {
    let capped = most.to_string();
    let how: Vec<&str> = match (asked, asked.contains("..")) {
        // Every branch, which is what an investigation **is**. Trying three
        // variants means three branches off one commit, and a walk from one tip
        // cannot see its own siblings: `rev-list HEAD` follows ancestry, and a
        // sibling is not an ancestor. A tool for exploring branches whose
        // default cannot see them is a tool for exploring one line.
        (ALL, _) => vec!["rev-list", "--all", "-n", &capped],
        (_, true) => vec!["rev-list", asked],
        (_, false) => vec!["rev-list", "-n", &capped, asked],
    };
    let said = git(repo, &how).map_err(|said| Trouble::NoSuchRevision {
        revspec: asked.to_string(),
        said,
    })?;
    Ok(said.lines().map(str::to_string).collect())
}

/// The commit under each of these that is not one of them.
///
/// A range says which commits to **show**, and a step needs the one below it:
/// with three branches that is one under **each**, since every line needs
/// something of its own to be compared against.
pub fn beneath(repo: &Path, commits: &[String]) -> Vec<String> {
    let inside: std::collections::HashSet<&str> = commits.iter().map(String::as_str).collect();
    let mut under: Vec<String> = parents_of(repo, commits)
        .into_iter()
        .flat_map(|(_, parents)| parents)
        .filter(|parent| !inside.contains(parent.as_str()))
        .collect();
    under.sort();
    under.dedup();
    under
}

/// The commit before this one, if it has one.
///
/// Reaches past the range for the same reason `git log -p` does: otherwise the
/// oldest commit shown could not say what it did.
pub fn parent_of(repo: &Path, commit: &str) -> Option<String> {
    git(repo, &["rev-parse", "--verify", &format!("{commit}^")]).ok()
}

/// The full hash a revspec names.
///
/// Resolved before anything is written down, never stored as somebody typed
/// it: `HEAD~2` is another commit tomorrow, and a note has to stay about the
/// commit it was about.
pub fn named(repo: &Path, revspec: &str) -> Result<String, Trouble> {
    git(
        repo,
        &["rev-parse", "--verify", &format!("{revspec}^{{commit}}")],
    )
    .map_err(|said| Trouble::NoSuchRevision {
        revspec: revspec.to_string(),
        said,
    })
}

/// One file, as that commit had it.
///
/// `git show` y no una lectura del disco: lo que se quiere es el fichero **en
/// ese commit**, y el que hay en el árbol de trabajo es el de otro. De paso se
/// ahorra el worktree, que es la parte cara de todo lo demás de aquí.
///
/// Sin recortar nada. El resto de este fichero lee líneas de git y las limpia;
/// esto devuelve el contenido, y un `trim` en el camino se comería la línea en
/// blanco del final —la que git quiere— y las de arriba. Un fichero que vuelve
/// distinto de como salió es una edición que borra código sin decirlo.
pub fn read(repo: &Path, commit: &str, file: &str) -> Result<String, Trouble> {
    let of = |said: String| Trouble::NoSuchFile {
        file: file.to_string(),
        said,
    };
    within(file).map_err(of)?;
    let said = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["show", &format!("{commit}:{file}")])
        .output()
        .map_err(|why| of(format!("git could not be run: {why}")))?;
    match said.status.success() {
        true => Ok(String::from_utf8_lossy(&said.stdout).into_owned()),
        false => Err(of(String::from_utf8_lossy(&said.stderr).trim().to_string())),
    }
}

/// La ruta, comprobada de que está dentro del repositorio.
///
/// Una con `..` dentro sale de él, y atenderla sería un navegador leyendo y
/// escribiendo donde le apeteciera en el disco de alguien. Es la misma
/// comprobación que hace falta para leer y para bifurcar, así que está una vez.
fn within(file: &str) -> Result<PathBuf, String> {
    let inside = PathBuf::from(file);
    match inside.is_absolute() || inside.components().any(|of| of.as_os_str() == "..") {
        true => Err(format!("`{file}` is not a path inside the repository")),
        false => Ok(inside),
    }
}

/// Who each of these commits comes from, in one call.
///
/// A walk prints a line and a **DAG has edges**: a range flattens three
/// branches into an order that says nothing about which came from which.
pub fn parents_of(repo: &Path, commits: &[String]) -> Vec<(String, Vec<String>)> {
    let mut asking = vec!["rev-list", "--no-walk", "--parents"];
    asking.extend(commits.iter().map(String::as_str));
    git(repo, &asking)
        .map(|said| {
            said.lines()
                .filter_map(|line| {
                    let mut of = line.split_whitespace().map(str::to_string);
                    Some((of.next()?, of.collect()))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Which lines of a file one class occupies.
#[derive(Debug, Clone, Copy)]
pub struct Splice {
    /// 1-based, as `inspect.getsourcelines` and every editor count.
    pub line: u32,
    pub lines: u32,
}

impl Splice {
    /// The whole file with those lines swapped for `what`.
    fn into(self, whole: &str, what: &str) -> Result<String, String> {
        let lines: Vec<&str> = whole.lines().collect();
        let from = self.line.saturating_sub(1) as usize;
        let to = from + self.lines as usize;
        if from > lines.len() || to > lines.len() {
            // The file at that commit is not the file the panel read. Refusing
            // is the only safe answer: splicing at a guessed offset would cut
            // the class in half and commit it.
            return Err(format!(
                "lines {}..{} are not in a file of {}: the source moved since it was read",
                self.line,
                to,
                lines.len()
            ));
        }
        let mut said = lines[..from].to_vec();
        said.extend(what.trim_end_matches('\n').lines());
        said.extend_from_slice(&lines[to..]);
        let mut out = said.join("\n");
        // A file git is happy with ends in a newline, and so did this one.
        if whole.ends_with('\n') {
            out.push('\n');
        }
        Ok(out)
    }
}

/// A checkout of `from` with one class replaced, and nothing committed.
///
/// Shared by checking and forking on purpose: what gets measured has to be the
/// same tree that gets committed, or a green light means nothing. `branch`
/// cuts one; `None` leaves it detached, which is what a check wants — asking
/// whether an edit survives should not litter a repository with the noes.
pub fn laid_out(
    repo: &Path,
    from: &str,
    branch: Option<&str>,
    file: &str,
    at: Splice,
    what: &str,
) -> Result<(tempfile::TempDir, PathBuf), Trouble> {
    let named = branch.unwrap_or("");
    let of = |said: String| Trouble::NoSuchBranch {
        branch: named.to_string(),
        said,
    };
    if let Some(branch) = branch
        && (branch.is_empty() || branch.starts_with('-') || branch.contains(".."))
    {
        return Err(of(
            "a branch name cannot be empty, start with `-`, or contain `..`".into(),
        ));
    }
    let inside = within(file).map_err(of)?;

    let held = tempfile::tempdir().map_err(|why| Trouble::NoWorktree {
        commit: from.to_string(),
        said: why.to_string(),
    })?;
    let working = held.path().join("apart");
    let mut how = vec!["worktree", "add", "--quiet"];
    if let Some(branch) = branch {
        how.extend(["-b", branch]);
    } else {
        how.push("--detach");
    }
    let at_path = working.display().to_string();
    how.extend([at_path.as_str(), from]);
    git(repo, &how).map_err(of)?;

    let writing = working.join(&inside);
    // The panel shows one class and a file usually holds four. Writing what
    // the panel showed would silently drop the imports, the sibling classes
    // and the `build()` that ties them together.
    let spliced = std::fs::read_to_string(&writing)
        .map_err(|why| of(why.to_string()))
        .and_then(|whole| at.into(&whole, what).map_err(of))?;
    std::fs::write(&writing, spliced).map_err(|why| of(why.to_string()))?;
    Ok((held, working))
}

/// Takes a worktree back out, so the next `worktree add` on that commit is not
/// refused by a record git kept of one nobody removed.
pub fn forget(repo: &Path, working: &Path) -> Result<String, String> {
    git(
        repo,
        &[
            "worktree",
            "remove",
            "--force",
            &working.display().to_string(),
        ],
    )
}

/// Cuts a branch at `from`, replaces one class in `file`, and commits it.
///
/// **Editing is forking.** A commit is a version that has already been
/// measured, so wanting to change one is wanting another variant from here:
/// this never touches an existing branch and never rewrites anything, and the
/// worst it can do is leave a branch nobody asked for. In a worktree of its
/// own, so somebody's checkout, index and unstaged work are left alone.
pub fn forked(
    repo: &Path,
    from: &str,
    branch: &str,
    file: &str,
    at: Splice,
    what: &str,
    said: &str,
) -> Result<String, Trouble> {
    let (_held, working) = laid_out(repo, from, Some(branch), file, at, what)?;
    let of = |trouble: String| Trouble::NoSuchBranch {
        branch: branch.to_string(),
        said: trouble,
    };
    let done = git(&working, &["add", "--", file])
        .and_then(|_| git(&working, &["commit", "-q", "-m", said]))
        .and_then(|_| git(&working, &["rev-parse", "HEAD"]));

    // The worktree goes either way. The branch stays: on success it is the
    // point, and on failure it is the only trace of what was attempted.
    let _ = git(
        repo,
        &[
            "worktree",
            "remove",
            "--force",
            &working.display().to_string(),
        ],
    );
    done.map_err(of)
}

/// Who is saying it. Git's idea of who you are, or the account's.
pub fn whoami(repo: &Path) -> String {
    git(repo, &["config", "user.name"])
        .ok()
        .filter(|said| !said.is_empty())
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "nobody".to_string())
}

/// What each commit was called and when it was made, in one call.
///
/// The time is not decoration: which of three variants was tried first is a
/// question about **when**, and the order a walk arrives in cannot answer it —
/// for commits made in the same second `rev-list` falls back to the order it
/// traverses refs, which is their names, so branches would come out
/// alphabetically and look chronological.
pub fn told(repo: &Path, commits: &[String]) -> HashMap<String, (u64, String)> {
    let mut asking = vec!["log", "--no-walk", "--format=%H%x00%ct%x00%s"];
    asking.extend(commits.iter().map(String::as_str));
    git(repo, &asking)
        .map(|said| {
            said.lines()
                .filter_map(|line| {
                    let mut of = line.split('\0');
                    let commit = of.next()?.to_string();
                    let when = of.next()?.parse().ok()?;
                    Some((commit, (when, of.next().unwrap_or_default().to_string())))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Runs git in a repo and returns its trimmed output.
fn git(repo: &Path, args: &[&str]) -> Result<String, String> {
    let said = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .map_err(|why| format!("git could not be run: {why}"))?;
    match said.status.success() {
        true => Ok(String::from_utf8_lossy(&said.stdout).trim().to_string()),
        false => Err(String::from_utf8_lossy(&said.stderr).trim().to_string()),
    }
}

#[derive(Debug)]
pub enum Trouble {
    NoSuchRevision { revspec: String, said: String },
    NoSuchFile { file: String, said: String },
    NoWorktree { commit: String, said: String },
    NoSuchBranch { branch: String, said: String },
}

impl fmt::Display for Trouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSuchRevision { revspec, said } => {
                write!(f, "`{revspec}` is not a commit here: {said}")
            }
            Self::NoSuchFile { file, said } => {
                write!(f, "`{file}` could not be read: {said}")
            }
            Self::NoWorktree { commit, said } => {
                write!(f, "{commit} could not be laid out: {said}")
            }
            Self::NoSuchBranch { branch, said } => {
                write!(f, "`{branch}` could not be cut: {said}")
            }
        }
    }
}

impl std::error::Error for Trouble {}
