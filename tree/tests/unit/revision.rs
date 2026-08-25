//! Reading a repository, against a real one built for the occasion.
//!
//! `git` and not a double: what is being checked here is that the arguments
//! handed to it mean what they are believed to mean, and a double would agree
//! with the belief rather than with git.

use somatize_tree::revision::{ALL, beneath, commits_in, parent_of};
use std::path::Path;
use std::process::Command;

/// A repository with `n` commits on top of a first one, oldest called `base`.
fn a_line_of(n: usize) -> tempfile::TempDir {
    let at = tempfile::tempdir().expect("a temporary directory");
    let run = |args: &[&str]| {
        let said = Command::new("git")
            .arg("-C")
            .arg(at.path())
            .args(args)
            .output()
            .expect("git runs");
        assert!(said.status.success(), "git {args:?}: {said:?}");
    };
    run(&["init", "-q"]);
    run(&["config", "user.email", "t@t"]);
    run(&["config", "user.name", "t"]);
    for which in 0..=n {
        std::fs::write(at.path().join("a.txt"), format!("{which}")).expect("a file");
        run(&["add", "-A"]);
        run(&[
            "commit",
            "-q",
            "-m",
            &match which {
                0 => "base".to_string(),
                _ => format!("step {which}"),
            },
        ]);
    }
    at
}

fn subjects(repo: &Path, commits: &[String]) -> Vec<String> {
    let told = somatize_tree::revision::told(repo, commits);
    commits
        .iter()
        .map(|commit| told[commit].1.clone())
        .collect()
}

#[test]
fn a_range_comes_back_newest_first() {
    let at = a_line_of(3);

    let commits = commits_in(at.path(), "HEAD~3..HEAD", 10).expect("a range");

    assert_eq!(
        subjects(at.path(), &commits),
        ["step 3", "step 2", "step 1"]
    );
}

#[test]
fn a_range_leaves_out_where_it_starts_which_is_why_the_parent_is_asked_for() {
    // The behaviour the walk depends on: `A..B` does not include `A`, so the
    // oldest commit shown has nothing under it to be compared against until
    // `parent_of` reaches past the range — the same reach `git log -p` makes.
    let at = a_line_of(3);
    let commits = commits_in(at.path(), "HEAD~3..HEAD", 10).expect("a range");

    let under = parent_of(at.path(), commits.last().expect("three of them"));

    assert_eq!(
        subjects(at.path(), &[under.expect("step 1 has a parent")])[0],
        "base",
    );
}

#[test]
fn the_first_commit_of_all_has_nothing_under_it() {
    // Not an error: a walk that reaches the root prints it as a row with no
    // step, which is what it is.
    let at = a_line_of(0);
    let commits = commits_in(at.path(), "HEAD", 10).expect("one commit");

    assert_eq!(parent_of(at.path(), &commits[0]), None);
}

#[test]
fn a_revspec_that_is_not_one_is_said_so_and_not_guessed_at() {
    let at = a_line_of(1);

    assert!(commits_in(at.path(), "no-such-branch..HEAD", 10).is_err());
}

#[test]
fn asking_for_more_than_there_are_is_an_answer_and_not_an_error() {
    // The bug this is here for: `HEAD~10..HEAD` in a repository with four
    // commits is not an empty walk, it is git saying *revisión desconocida*.
    // A default nobody typed cannot be a range for exactly that reason.
    let at = a_line_of(3);

    let commits = commits_in(at.path(), "HEAD", 10).expect("as many as there are");

    assert_eq!(
        commits.len(),
        4,
        "everything, and no complaint about the rest"
    );
}

#[test]
fn and_a_range_that_reaches_past_the_root_still_says_so() {
    // The other half: somebody who typed a range meant it, and being told it
    // names nothing is more use than four commits they did not ask for.
    let at = a_line_of(3);

    assert!(commits_in(at.path(), "HEAD~10..HEAD", 10).is_err());
}

#[test]
fn a_revspec_is_capped_where_it_was_told_to_be() {
    let at = a_line_of(9);

    let commits = commits_in(at.path(), "HEAD", 3).expect("three of them");

    assert_eq!(
        subjects(at.path(), &commits),
        ["step 9", "step 8", "step 7"]
    );
}

#[test]
fn a_repository_with_one_commit_walks_it_and_stops() {
    let at = a_line_of(0);

    let commits = commits_in(at.path(), "HEAD", 10).expect("the only one");

    assert_eq!(commits.len(), 1);
    assert_eq!(parent_of(at.path(), &commits[0]), None);
}

// ── Three variants of one idea are three branches ──

/// A base, and `n` branches of two commits each off it.
fn a_fan_of(n: usize) -> tempfile::TempDir {
    let at = a_line_of(0);
    let run = |args: &[&str]| {
        let said = Command::new("git")
            .arg("-C")
            .arg(at.path())
            .args(args)
            .output()
            .expect("git runs");
        assert!(said.status.success(), "git {args:?}");
    };
    let base = commits_in(at.path(), "HEAD", 1).expect("the base")[0].clone();
    for which in 1..=n {
        run(&["checkout", "-q", "-b", &format!("variant-{which}"), &base]);
        for step in 1..=2 {
            std::fs::write(at.path().join("a.txt"), format!("{which}-{step}")).expect("a file");
            run(&["add", "-A"]);
            run(&[
                "commit",
                "-q",
                "-m",
                &format!("variant {which}, step {step}"),
            ]);
        }
    }
    run(&["checkout", "-q", &base]);
    at
}

#[test]
fn a_walk_from_one_tip_cannot_see_its_own_siblings() {
    // Why the default is every branch. `rev-list HEAD` follows **ancestry**,
    // and a sibling is not an ancestor: a tool for exploring branches whose
    // default cannot see them is a tool for exploring one line.
    let at = a_fan_of(3);

    let one = commits_in(at.path(), "HEAD", 50).expect("from the tip");
    let every = commits_in(at.path(), ALL, 50).expect("every branch");

    assert_eq!(
        one.len(),
        1,
        "checked out at the base, its ancestry is itself"
    );
    assert_eq!(every.len(), 7, "the base and three branches of two");
}

#[test]
fn every_line_gets_the_commit_under_it_and_not_just_the_oldest() {
    // A step needs something below it to be compared against. With one line
    // that is a single commit; with three branches it is one under **each**,
    // and stopping at the oldest would leave two lines with nothing.
    let at = a_fan_of(3);
    let every = commits_in(at.path(), ALL, 50).expect("every branch");

    let under = beneath(at.path(), &every);

    assert!(
        under.is_empty(),
        "the base is already in the walk, so nothing is missing: {under:?}",
    );

    // Ask for only the tips — by name, because commits made in one second are
    // not ordered by which branch they are on — and each needs its own.
    let tips: Vec<String> = (1..=3)
        .map(|which| {
            commits_in(at.path(), &format!("variant-{which}"), 1).expect("a tip")[0].clone()
        })
        .collect();

    assert_eq!(
        beneath(at.path(), &tips).len(),
        3,
        "three lines, three commits underneath them",
    );
}

// ── Editing is forking ──

#[test]
fn a_class_is_spliced_back_into_its_file_and_nothing_around_it_moves() {
    // The one that must not be wrong. The panel shows ONE class and a module
    // holds several: writing what the panel showed as the whole file would
    // silently drop the imports, the siblings and the `build()` that ties them
    // together — and commit that on a new branch without a word.
    let at = a_line_of(0);
    std::fs::write(
        at.path().join("mod.py"),
        "import soma\n\n\nclass A:\n    x = 1\n\n\nclass B:\n    y = 2\n",
    )
    .expect("a module");
    for args in [vec!["add", "-A"], vec!["commit", "-q", "-m", "two classes"]] {
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(at.path())
                .args(&args)
                .output()
                .expect("git runs")
                .status
                .success()
        );
    }
    let from = commits_in(at.path(), "HEAD", 1).expect("the commit")[0].clone();

    somatize_tree::revision::forked(
        at.path(),
        &from,
        "variant",
        "mod.py",
        somatize_tree::revision::Splice { line: 4, lines: 2 },
        "class A:\n    x = 99\n",
        "A only",
    )
    .expect("a fork");

    let said = Command::new("git")
        .arg("-C")
        .arg(at.path())
        .args(["show", "variant:mod.py"])
        .output()
        .expect("git runs");
    let after = String::from_utf8_lossy(&said.stdout);

    assert!(after.contains("x = 99"), "the edit landed: {after}");
    assert!(
        after.starts_with("import soma"),
        "the import survived: {after}"
    );
    assert!(after.contains("class B:"), "the sibling survived: {after}");
    assert!(after.contains("y = 2"), "and so did its body: {after}");
}

#[test]
fn a_fork_leaves_the_branch_it_came_from_exactly_as_it_was() {
    // A commit has already been measured. Nothing here changes one, and the
    // worst this can do is leave a branch nobody asked for.
    let at = a_line_of(1);
    let before = commits_in(at.path(), "HEAD", 10).expect("before");

    somatize_tree::revision::forked(
        at.path(),
        &before[0],
        "variant",
        "a.txt",
        somatize_tree::revision::Splice { line: 1, lines: 1 },
        "something else\n",
        "a variant",
    )
    .expect("a fork");

    assert_eq!(
        commits_in(at.path(), "HEAD", 10).expect("after"),
        before,
        "the branch it was cut from did not move",
    );
}

#[test]
fn a_splice_that_does_not_fit_is_refused_rather_than_guessed_at() {
    // The file at that commit is not the file the panel read. Splicing at a
    // guessed offset would cut a class in half and commit it.
    let at = a_line_of(0);

    let said = somatize_tree::revision::forked(
        at.path(),
        &commits_in(at.path(), "HEAD", 1).expect("the commit")[0].clone(),
        "variant",
        "a.txt",
        somatize_tree::revision::Splice {
            line: 400,
            lines: 9,
        },
        "whatever",
        "nope",
    );

    assert!(said.is_err());
}

#[test]
fn a_path_that_climbs_out_of_the_repository_is_refused() {
    // It is a browser asking a server to write a file. Nothing above the
    // repository is any of its business.
    let at = a_line_of(0);
    let from = commits_in(at.path(), "HEAD", 1).expect("the commit")[0].clone();

    for path in ["../escaped.txt", "/etc/passwd", "a/../../out.txt"] {
        assert!(
            somatize_tree::revision::forked(
                at.path(),
                &from,
                "variant",
                path,
                somatize_tree::revision::Splice { line: 1, lines: 1 },
                "x",
                "nope",
            )
            .is_err(),
            "`{path}` should have been refused",
        );
    }
}

#[test]
fn a_file_comes_back_from_the_commit_and_not_from_the_working_tree() {
    // Lo que se abre en un panel es el fichero **de ese commit**. El que hay en
    // el disco es el de otro, y enseñarlo con el nombre de éste sería la peor
    // clase de casi acertar.
    let at = a_line_of(2);
    let old = commits_in(at.path(), "HEAD~2..HEAD", 10).expect("a range")[1].clone();
    std::fs::write(at.path().join("a.txt"), "lo que hay sin commitear").expect("a file");

    let said = somatize_tree::revision::read(at.path(), &old, "a.txt").expect("the file");

    assert_eq!(said, "1");
}

#[test]
fn a_file_comes_back_exactly_as_it_was_written() {
    // Sin recortar. El resto de este módulo lee líneas de git y las limpia, y
    // aplicar eso aquí se comería la línea en blanco del final —la que git
    // quiere— y las de arriba. Un fichero que vuelve distinto de como salió es
    // una edición que borra código sin decirlo.
    let at = a_line_of(0);
    let whole = "\n\nclass Head:\n    pass\n\n\n";
    std::fs::write(at.path().join("net.py"), whole).expect("a file");
    let run = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(at.path())
            .args(args)
            .output()
            .expect("git runs");
    };
    run(&["add", "-A"]);
    run(&["commit", "-q", "-m", "net"]);
    let head = commits_in(at.path(), "HEAD", 1).expect("the commit")[0].clone();

    let said = somatize_tree::revision::read(at.path(), &head, "net.py").expect("the file");

    assert_eq!(said, whole);
}

#[test]
fn a_file_that_climbs_out_of_the_repository_is_refused_the_same_as_a_fork() {
    // La misma comprobación y no una parecida: leer donde a alguien le apetezca
    // es la mitad del problema que escribir donde a alguien le apetezca.
    let at = a_line_of(0);
    let from = commits_in(at.path(), "HEAD", 1).expect("the commit")[0].clone();

    for path in ["../escaped.txt", "/etc/passwd", "a/../../out.txt"] {
        assert!(
            somatize_tree::revision::read(at.path(), &from, path).is_err(),
            "`{path}` should have been refused",
        );
    }
}

#[test]
fn a_whole_file_is_a_splice_over_all_of_its_lines() {
    // Lo que hace que editar un fichero entero no necesite un segundo camino:
    // un fichero es su propio trozo, `line: 1` y tantas líneas como tenga. El
    // empalme que devuelve una clase a su sitio devuelve el fichero al suyo.
    //
    // Contadas como las cuenta `str::lines`, que es quien las cuenta al otro
    // lado: `"a\nb\n"` son dos y no tres. Quien las cuente de la otra manera
    // pide un trozo más largo que el fichero y se le dice que no.
    let at = a_line_of(0);
    let whole = "import torch\n\n\nclass Head:\n    pass\n";
    std::fs::write(at.path().join("net.py"), whole).expect("a file");
    let run = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(at.path())
            .args(args)
            .output()
            .expect("git runs");
    };
    run(&["add", "-A"]);
    run(&["commit", "-q", "-m", "net"]);
    let from = commits_in(at.path(), "HEAD", 1).expect("the commit")[0].clone();
    let rewritten = "import torch\n\n\nclass Head:\n    def forward(self, x):\n        return x\n";

    let made = somatize_tree::revision::forked(
        at.path(),
        &from,
        "whole-file",
        "net.py",
        somatize_tree::revision::Splice {
            line: 1,
            lines: whole.lines().count() as u32,
        },
        rewritten,
        "el fichero entero",
    )
    .expect("the fork");

    assert_eq!(
        somatize_tree::revision::read(at.path(), &made, "net.py").expect("the file"),
        rewritten,
        "vuelve exactamente lo que se mandó, sin nada de alrededor pegado",
    );
}
