//! The binary, against a real repository, with a real interpreter.
//!
//! Everything else in `tests/` checks a piece with the rest held still. This
//! checks the one thing none of those can: that `git`, a worktree, a Python
//! subprocess, `somatize.foreseen` and the store all still agree once they are
//! in the same room.
//!
//! The repository comes from `examples/an-investigation.sh --only-build`, the
//! **same fixture the example uses**: one definition, two consumers, so the
//! example cannot rot without this going red.
//!
//! It skips rather than fails without an interpreter. Building a graph needs a
//! Python that can import `somatize`, which is a `maturin develop` away and
//! not something a checkout has; `SOMA_TREE_PYTHON` names one, and without it
//! the workspace's own `.venv` is tried. A test that failed here would be
//! reporting on somebody's environment and calling it a bug in the tool.

use std::path::{Path, PathBuf};
use std::process::Command;

/// An interpreter that can import `somatize`, if there is one.
fn an_interpreter() -> Option<PathBuf> {
    let named = std::env::var_os("SOMA_TREE_PYTHON").map(PathBuf::from);
    let beside = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|at| at.join(".venv/bin/python"));
    named
        .into_iter()
        .chain(beside)
        .find(|python| imports_somatize(python))
}

fn imports_somatize(python: &Path) -> bool {
    Command::new(python)
        .args(["-c", "import somatize"])
        .output()
        .map(|said| said.status.success())
        .unwrap_or(false)
}

/// A base with `n` branches of two commits off it, an hour apart, which is the
/// shape an investigation has: one idea, several variants tried from it.
///
/// Hours apart on purpose. Commits made in the same second send `rev-list`
/// back to the order it traverses refs — their **names** — so a fixture built
/// in one instant would lay branches out alphabetically and look chronological.
fn a_fan_of(n: usize, python: &Path) -> tempfile::TempDir {
    let at = an_investigation(python);
    let git = |args: &[&str]| {
        let said = Command::new("git")
            .arg("-C")
            .arg(at.path())
            .args(args)
            .output()
            .expect("git runs");
        assert!(said.status.success(), "git {args:?}: {said:?}");
        String::from_utf8_lossy(&said.stdout).trim().to_string()
    };
    // Only a commit is stamped. Handing a date to `checkout` is a date git
    // has nothing to do with, and it says so.
    let commit = |said: &str, when: u64| {
        let done = Command::new("git")
            .arg("-C")
            .arg(at.path())
            .args(["commit", "-q", "-m", said])
            .env("GIT_AUTHOR_DATE", format!("{when} +0000"))
            .env("GIT_COMMITTER_DATE", format!("{when} +0000"))
            .output()
            .expect("git runs");
        assert!(done.status.success(), "committing `{said}`: {done:?}");
    };

    let base = git(&["rev-parse", "HEAD"]);
    for which in 1..=n as u64 {
        git(&["checkout", "-q", "-b", &format!("variant-{which}"), &base]);
        for step in 1..=2u64 {
            std::fs::write(at.path().join("variant.txt"), format!("{which}-{step}"))
                .expect("a file");
            git(&["add", "-A"]);
            commit(
                &format!("variant {which}, step {step}"),
                1_750_000_000 + which * 3600 + step * 600,
            );
        }
        git(&["checkout", "-q", &base]);
    }
    at
}

/// The example's repository, laid down in a temporary directory.
fn an_investigation(python: &Path) -> tempfile::TempDir {
    let at = tempfile::tempdir().expect("a temporary directory");
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/an-investigation.sh");
    let said = Command::new("bash")
        .arg(&script)
        .arg("--only-build")
        .arg(at.path())
        .env("SOMA_TREE_PYTHON", python)
        .output()
        .expect("bash runs");
    assert!(
        said.status.success(),
        "the example could not lay down its repository: {}",
        String::from_utf8_lossy(&said.stderr),
    );
    at
}

/// Runs the binary in that repository, and returns what it said.
fn somatize_tree(at: &Path, args: &[&str]) -> String {
    let said = Command::new(env!("CARGO_BIN_EXE_somatize-tree"))
        .args(args)
        .arg("--repo")
        .arg(at)
        // Its own store, so one test never reads what another remembered.
        .env("XDG_CACHE_HOME", at.join("cache"))
        .current_dir(at)
        .output()
        .expect("the binary runs");
    assert!(
        said.status.code() != Some(2),
        "somatize-tree {args:?} could not run: {}",
        String::from_utf8_lossy(&said.stderr),
    );
    String::from_utf8_lossy(&said.stdout).into_owned()
}

/// What it said when it refused, which is stderr and not stdout.
fn soma_tree_refusing(at: &Path, args: &[&str]) -> String {
    let said = Command::new(env!("CARGO_BIN_EXE_somatize-tree"))
        .args(args)
        .arg("--repo")
        .arg(at)
        .env("XDG_CACHE_HOME", at.join("cache"))
        .current_dir(at)
        .output()
        .expect("the binary runs");
    assert!(!said.status.success(), "a refusal was expected: {args:?}");
    String::from_utf8_lossy(&said.stderr).into_owned()
}

/// Every test needs the same two things, and skips for the same reason.
macro_rules! given {
    ($at:ident) => {
        let Some(python) = an_interpreter() else {
            eprintln!("no interpreter that imports somatize: skipped");
            return;
        };
        let $at = an_investigation(&python);
        let $at = $at.path();
    };
    // For the tests that lay down a shape of their own.
    ($python:ident, $unused:ident) => {
        let Some($python) = an_interpreter() else {
            eprintln!("no interpreter that imports somatize: skipped");
            return;
        };
    };
}

#[test]
fn a_constructor_argument_moves_the_name_so_the_cache_misses() {
    given!(at);

    let said = somatize_tree(at, &["diff", "HEAD~3", "HEAD~2"]);

    assert!(said.contains("strict"), "{said}");
    assert!(said.contains("CHANGED"), "{said}");
    assert!(
        said.contains("Classify(threshold=1.0) → Classify(threshold=2.0)"),
        "it says what somebody typed, not two digests: {said}",
    );
    assert!(
        !said.contains("STALE"),
        "the name moved, so nothing stale is served: {said}",
    );
}

#[test]
fn the_body_of_a_forward_moves_no_name_and_the_cache_will_hit() {
    // The finding the whole tool exists for. soma answers this edit with a
    // line on stderr during a run; here it is said before paying for one.
    given!(at);

    let said = somatize_tree(at, &["diff", "HEAD~2", "HEAD~1"]);

    assert!(said.contains("embed"), "{said}");
    assert!(said.contains("STALE"), "{said}");
    assert!(
        said.contains("the cache will HIT"),
        "and it says what that costs: {said}",
    );
}

#[test]
fn what_reads_a_stale_answer_is_suspect_even_though_nobody_edited_it() {
    given!(at);

    let said = somatize_tree(at, &["diff", "HEAD~2", "HEAD~1"]);

    assert!(said.contains("SUSPECT"), "{said}");
    for under in ["strict", "loose", "vote"] {
        assert!(said.contains(under), "`{under}` reads it too: {said}");
    }
}

#[test]
fn retraining_is_another_trial_and_not_another_variant() {
    // The steer this was built around: weights are associated with a version
    // rather than being one.
    given!(at);

    let said = somatize_tree(at, &["diff", "HEAD~1", "HEAD"]);

    assert!(said.contains("RESETTLED"), "{said}");
    assert!(
        said.contains("No node will be handed a cached value"),
        "nothing stale is served: {said}",
    );
    assert!(
        !said.contains("The edit is in"),
        "nobody edited anything: {said}",
    );
}

#[test]
fn a_walk_says_where_each_step_edited() {
    given!(at);

    let said = somatize_tree(at, &["log", "HEAD~3..HEAD"]);

    assert!(said.contains("edit: strict"), "{said}");
    assert!(said.contains("edit: embed"), "{said}");
    assert!(
        said.contains("no edit · resettled: embed"),
        "the retrain edited nothing, and it says which node was retrained: {said}",
    );
}

#[test]
fn a_walk_reaches_one_commit_past_its_range() {
    // `A..B` does not include `A`, so the oldest commit shown would have
    // nothing under it to be compared against. Three commits shown, three
    // steps, exactly as `git log -p` reaches past a range too.
    given!(at);

    let said = somatize_tree(at, &["log", "HEAD~3..HEAD"]);

    assert_eq!(said.matches('│').count(), 3, "{said}");
}

#[test]
fn a_verdict_is_written_down_and_shows_up_in_the_walk() {
    given!(at);
    somatize_tree(
        at,
        &["verdict", "invalid", "HEAD~2", "-m", "the loader lied"],
    );

    let said = somatize_tree(at, &["log", "HEAD~3..HEAD"]);

    assert!(said.contains("[invalid]"), "{said}");
}

#[test]
fn changing_your_mind_does_not_erase_what_you_thought_before() {
    // The reason the journal is append-only. An `invalid` that turned out to be
    // a misread split is the most instructive thing in an investigation — and
    // being able to take it back is the reason `sound` exists at all: without
    // it a mistaken `invalid` would leave a whole subtree suspect for good.
    given!(at);
    somatize_tree(
        at,
        &["verdict", "invalid", "HEAD~2", "-m", "recall impossible"],
    );
    somatize_tree(
        at,
        &[
            "verdict",
            "sound",
            "HEAD~2",
            "-m",
            "it was the split I read",
        ],
    );

    let said = somatize_tree(at, &["show", "HEAD~2"]);

    assert!(said.contains("recall impossible"), "{said}");
    assert!(said.contains("it was the split I read"), "{said}");
    assert!(
        somatize_tree(at, &["log", "HEAD~3..HEAD"]).contains("[sound]"),
        "and the last word is the one that counts",
    );
}

#[test]
fn a_note_does_not_count_as_changing_your_mind() {
    given!(at);
    somatize_tree(at, &["verdict", "invalid", "HEAD~2", "-m", "no"]);
    somatize_tree(at, &["note", "HEAD~2", "-m", "recall was 0.61"]);

    assert!(
        somatize_tree(at, &["log", "HEAD~3..HEAD"]).contains("[invalid]"),
        "writing down what you saw is not a verdict",
    );
}

#[test]
fn doubt_reaches_a_commit_that_did_not_exist_when_it_was_cast() {
    // The one that justifies deriving this rather than storing it. The verdict
    // is written about **one** commit; that its descendants are suspect is
    // worked out from git when somebody asks, so a commit made afterwards is
    // marked the moment it exists and nobody goes back to write anything.
    given!(at);
    somatize_tree(
        at,
        &["verdict", "invalid", "HEAD~2", "-m", "the dataloader lied"],
    );

    let said = somatize_tree(at, &["log", "HEAD~3..HEAD"]);

    assert!(said.contains("[invalid]"), "{said}");
    assert_eq!(
        said.matches("[under something invalid]").count(),
        2,
        "the two commits under it, neither of them judged: {said}",
    );
}

#[test]
fn having_looked_and_found_nothing_does_not_put_numbers_in_doubt() {
    given!(at);
    somatize_tree(at, &["verdict", "sound", "HEAD~2", "-m", "checked it"]);

    let said = somatize_tree(at, &["log", "HEAD~3..HEAD"]);

    assert!(!said.contains("under something invalid"), "{said}");
}

#[test]
fn a_word_that_went_to_layer_2_says_where_it_went() {
    // Whoever writes it has the old habit. What they need to know is not that
    // they were wrong but where what they meant to say is said now.
    given!(at);
    let said = soma_tree_refusing(at, &["verdict", "dead-end", "HEAD~2", "-m", "no way"]);

    assert!(said.contains("reasoning"), "{said}");
    assert!(said.contains("scope"), "{said}");
}

#[test]
fn asking_twice_gives_the_same_answer_and_touches_no_worktree() {
    // A snapshot is a pure function of a commit, so the second walk is a scan
    // of the store. If these two ever disagree, the cache is keyed on too
    // little — which is exactly the mistake this tool reports about caches.
    given!(at);

    let once = somatize_tree(at, &["log", "HEAD~3..HEAD"]);
    let twice = somatize_tree(at, &["log", "HEAD~3..HEAD"]);

    assert_eq!(once, twice);
}

#[test]
fn no_worktree_is_left_behind() {
    // Not tidiness: git keeps a record of a worktree in the repository, and the
    // next `worktree add` on the same commit refuses.
    given!(at);
    somatize_tree(at, &["log", "HEAD~3..HEAD"]);

    let listed = Command::new("git")
        .arg("-C")
        .arg(at)
        .args(["worktree", "list"])
        .output()
        .expect("git runs");

    assert_eq!(
        String::from_utf8_lossy(&listed.stdout).lines().count(),
        1,
        "only the repository itself",
    );
}

#[test]
fn a_revspec_that_names_nothing_is_said_out_loud() {
    given!(at);

    let said = Command::new(env!("CARGO_BIN_EXE_somatize-tree"))
        .args(["diff", "no-such-thing", "HEAD", "--repo"])
        .arg(at)
        .env("XDG_CACHE_HOME", at.join("cache"))
        .output()
        .expect("the binary runs");

    assert_eq!(said.status.code(), Some(2), "trouble, not a difference");
    assert!(
        String::from_utf8_lossy(&said.stderr).contains("no-such-thing"),
        "and it says which one",
    );
}

#[test]
fn a_walk_with_nothing_typed_works_on_a_repository_of_any_length() {
    // What somebody hit: the default was a range asking for ten commits, and a
    // range reaching past the root is an unknown-revision error rather than a
    // short answer. A default nobody typed has to work on the repository they
    // have.
    given!(at);

    let said = somatize_tree(at, &["log"]);

    assert!(said.contains("4 commits"), "{said}");
    assert!(!said.contains("unknown revision"), "{said}");
}

#[test]
fn a_range_somebody_did_type_is_still_taken_at_their_word() {
    // The other half: being told the range names nothing is more use than four
    // commits they did not ask for.
    given!(at);

    let said = Command::new(env!("CARGO_BIN_EXE_somatize-tree"))
        .args(["log", "HEAD~40..HEAD", "--repo"])
        .arg(at)
        .env("XDG_CACHE_HOME", at.join("cache"))
        .output()
        .expect("the binary runs");

    assert_eq!(said.status.code(), Some(2), "trouble, not a short walk");
}

#[test]
fn every_branch_is_walked_and_not_only_the_one_checked_out() {
    // `rev-list HEAD` follows ancestry, and a sibling is not an ancestor: a
    // walk from one tip cannot see the other variants at all.
    given!(python, at);
    let at = a_fan_of(3, &python);
    let at = at.path();

    let said = somatize_tree(at, &["log"]);

    for which in 1..=3 {
        assert!(said.contains(&format!("variant {which}, step 2")), "{said}");
    }
}

#[test]
fn a_step_comes_from_a_parent_and_not_from_the_line_above_it() {
    // With three branches interleaved in a walk, adjacent entries are three
    // different lines of exploration. A step that paired them would answer
    // confidently about an edit nobody made.
    given!(python, at);
    let at = a_fan_of(3, &python);
    let at = at.path();

    let walk: serde_json::Value =
        serde_json::from_str(&somatize_tree(at, &["log", "--json"])).expect("json");
    let steps = walk["steps"].as_array().expect("steps");

    // Somebody has three steps going out of them, and that somebody is where
    // the branches were cut. Which commit it is does not matter; that the fan
    // exists at all is the whole assertion, and a walk that paired adjacent
    // entries would have every commit with exactly one.
    let mut out: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for step in steps {
        *out.entry(step["from"].as_str().expect("a hash"))
            .or_default() += 1;
    }
    let fan = out.values().copied().max().unwrap_or(0);

    assert_eq!(
        fan, 3,
        "one step out per variant, off the commit they were cut from"
    );
}

#[test]
fn doubt_goes_down_a_branch_and_not_across_to_its_siblings() {
    // The bug this is here for: doubt was worked out by asking git for the
    // ancestry path to **a** tip, and with three branches that tip is usually
    // on somebody else's — so a verdict cast on one variant reached nothing.
    given!(python, at);
    let at = a_fan_of(3, &python);
    let at = at.path();
    somatize_tree(
        at,
        &["verdict", "invalid", "variant-2~1", "-m", "the split lied"],
    );

    let walk: serde_json::Value =
        serde_json::from_str(&somatize_tree(at, &["log", "--json"])).expect("json");
    let doubted: Vec<&str> = walk["stops"]
        .as_array()
        .expect("stops")
        .iter()
        .filter(|stop| stop["doubted"] == serde_json::json!(true))
        .map(|stop| stop["subject"].as_str().unwrap_or_default())
        .collect();

    assert!(
        doubted
            .iter()
            .any(|said| said.contains("variant 2, step 2")),
        "it reaches the commit under it: {doubted:?}",
    );
    assert!(
        !doubted
            .iter()
            .any(|said| said.contains("variant 1") || said.contains("variant 3")),
        "and nothing on anybody else's branch: {doubted:?}",
    );
}

#[test]
fn an_edit_that_does_not_parse_is_caught_before_anything_else_runs() {
    // The cheapest question, and the one that has to come first: importing a
    // module to find out it has a typo in it costs an interpreter and a torch.
    given!(python, unused);
    let at = an_investigation(&python);

    let said = Command::new(&python)
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/soma_tree_probe.py"))
        .args(["--build", "experiments.encoder:build", "--check", "embed"])
        .current_dir(at.path())
        .output()
        .expect("the probe runs");
    let answer: serde_json::Value =
        serde_json::from_slice(&said.stdout).expect("the probe writes json");
    let checks = answer["checks"].as_array().expect("checks");

    assert_eq!(checks[0]["what"], "syntax");
    assert_eq!(checks[0]["ok"], true, "the example's own code parses");
}

#[test]
fn the_graph_still_building_is_its_own_question() {
    // What a linter cannot see: rename a class and `build()` goes on calling
    // the old name. It parses, it lints clean, and it is a commit nobody can
    // run — which is exactly the kind of variant this exists to stop.
    given!(python, unused);
    let at = an_investigation(&python);
    let file = at.path().join("experiments/encoder.py");
    let was = std::fs::read_to_string(&file).expect("the module");
    std::fs::write(&file, was.replace("class Embed(", "class Embedding(")).expect("an edit");

    let said = Command::new(&python)
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/soma_tree_probe.py"))
        .args(["--build", "experiments.encoder:build", "--check", "embed"])
        .current_dir(at.path())
        .output()
        .expect("the probe runs");
    let answer: serde_json::Value =
        serde_json::from_slice(&said.stdout).expect("the probe writes json");
    let checks = answer["checks"].as_array().expect("checks");

    assert!(
        checks
            .iter()
            .any(|one| one["what"] == "syntax" && one["ok"] == true),
        "it parses: {checks:?}",
    );
    assert!(
        checks
            .iter()
            .any(|one| one["what"] == "the graph builds" && one["ok"] == false),
        "and it does not build: {checks:?}",
    );
}

#[test]
fn running_on_real_data_says_so_rather_than_inventing_some() {
    // With no store there is nothing kept to hand the node. A green light from
    // a fabricated value would be worse than no light: it would say an edit was
    // safe on data that never existed.
    given!(python, unused);
    let at = an_investigation(&python);

    let said = Command::new(&python)
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/soma_tree_probe.py"))
        .args(["--build", "experiments.encoder:build", "--check", "embed"])
        .current_dir(at.path())
        .output()
        .expect("the probe runs");
    let answer: serde_json::Value =
        serde_json::from_slice(&said.stdout).expect("the probe writes json");

    let run = answer["checks"]
        .as_array()
        .expect("checks")
        .iter()
        .find(|one| one["what"] == "it runs on real data")
        .expect("it was asked");

    assert_eq!(run["skipped"], true, "skipped, not passed: {run}");
}

#[test]
fn a_version_with_no_trials_says_where_they_are_written() {
    // The message whoever looks at this for the first time takes away. *0
    // trials* would leave them believing they are written from here, and they
    // are not: soma writes them from the machine running the study, under
    // this name.
    given!(at);

    let said = somatize_tree(at, &["trials", "HEAD"]);

    assert!(said.contains("0 trials"), "{said}");
    assert!(said.contains("somatize"), "{said}");
    assert!(said.contains("study="), "{said}");
}

#[test]
fn the_name_of_a_versions_study_comes_from_the_commit() {
    // The whole coupling with soma is this name, and whoever is going to run
    // the study has to be able to copy it from here.
    given!(at);
    let commit = String::from_utf8_lossy(
        &Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(at)
            .output()
            .expect("git")
            .stdout,
    )
    .trim()
    .to_string();

    let said = somatize_tree(at, &["trials", "HEAD"]);

    assert!(said.contains(&format!("/{commit}")), "{said}");
}

#[test]
fn a_goal_that_says_no_direction_is_refused_rather_than_ignored() {
    // A typo in `goal` would stop saying which was best with nothing saying
    // why, and that is worse than not starting at all.
    given!(at);
    std::fs::write(
        at.join("soma-tree.toml"),
        "build = \"experiments.encoder:build\"\ngoal = \"mas\"\n",
    )
    .expect("the config");

    let said = soma_tree_refusing(at, &["trials", "HEAD"]);

    assert!(said.contains("min"), "{said}");
    assert!(said.contains("max"), "{said}");
}

/// Abandons a commit's line, writing the decision the way the view would: an
/// attempt citing it, and a decision with that attempt as its scope.
fn abandoned(at: &Path, commit: &str) {
    let kept = somatize_store::Local::at(at.join("store")).expect("a store");
    // Read from the config and not from the directory's name: it is what
    // separates two investigations sharing a store, and writing under another
    // name leaves the moves where nobody reads them — with no error, which is
    // the worst part.
    let tree = somatize_tree::bench::Config::read(at)
        .expect("the config")
        .tree(at);
    let moves = somatize_tree::moves::Moves::of(tree, &kept);
    let a = moves
        .add(
            somatize_tree::moves::Kind::Attempt,
            "this way",
            "me",
            somatize_tree::moves::Scope::everything(),
            vec![somatize_tree::moves::Cited {
                what: "commit".into(),
                id: commit.into(),
            }],
            None,
        )
        .expect("the attempt");
    moves
        .add(
            somatize_tree::moves::Kind::Decision,
            "it leads nowhere",
            "me",
            somatize_tree::moves::Scope::of(vec![a]),
            Vec::new(),
            Some(somatize_tree::moves::Course::Abandon),
        )
        .expect("the decision");
}

#[test]
fn an_abandoned_line_folds_and_says_how_many_it_hides() {
    given!(at);
    let commit = revision(at, "HEAD~2");
    abandoned(at, &commit);

    let said = somatize_tree(at, &["log", "--store", &store(at), "HEAD~3..HEAD"]);

    // Not as a row of its own. Named in the step that came out of it, and said
    // to be pruned: a hash pointing at a row that is not drawn would be a dead
    // end for whoever reads.
    assert!(
        !said.contains(&format!("{}  ", &commit[..12])),
        "folded: {said}"
    );
    assert!(
        said.contains("(pruned)"),
        "and said, not left dangling: {said}"
    );
    // And never in silence: hiding by keeping quiet is the mistake this exists
    // not to make, so the missing row is counted out loud.
    assert!(said.contains("pruned lines"), "{said}");
    assert!(said.contains("--all-lines"), "{said}");
}

#[test]
fn nothing_is_deleted_by_pruning() {
    // Pruning is not drawing. A line that did not work is the most reusable
    // thing an investigation produces, and the only thing that stops it being
    // discovered again.
    given!(at);
    let commit = revision(at, "HEAD~2");
    abandoned(at, &commit);

    let said = somatize_tree(
        at,
        &["log", "--all-lines", "--store", &store(at), "HEAD~3..HEAD"],
    );

    assert!(said.contains(&commit[..12]), "{said}");
    assert!(
        said.contains("abandon"),
        "and it says why it was not drawn: {said}"
    );
}

#[test]
fn whoever_processes_the_answer_gets_all_of_it() {
    // Folding is for readability, and a program does not miss a row. Hiding a
    // commit from whoever is going to process the JSON would really hide it,
    // which is the opposite of what pruning does.
    given!(at);
    let commit = revision(at, "HEAD~2");
    abandoned(at, &commit);

    let said = somatize_tree(
        at,
        &["log", "--json", "--store", &store(at), "HEAD~3..HEAD"],
    );

    assert!(said.contains(&commit), "{said}");
    assert!(
        said.contains("\"pruned\": true"),
        "and said, not hidden: {said}"
    );
}

#[test]
fn a_commit_judged_wrong_does_not_fold_even_on_an_abandoned_line() {
    // The one that matters most. An `invalid` casts doubt on the measurement
    // the decision to abandon leaned on: hiding it would hide the very reason
    // to look at it again.
    given!(at);
    let commit = revision(at, "HEAD~2");
    abandoned(at, &commit);
    somatize_tree(
        at,
        &[
            "verdict",
            "invalid",
            "--store",
            &store(at),
            &commit,
            "-m",
            "the split was lying",
        ],
    );

    let said = somatize_tree(at, &["log", "--store", &store(at), "HEAD~3..HEAD"]);

    assert!(said.contains(&commit[..12]), "{said}");
}

/// The whole sha of a revspec, so it can be looked for in an output.
fn revision(at: &Path, rev: &str) -> String {
    String::from_utf8_lossy(
        &Command::new("git")
            .args(["rev-parse", rev])
            .current_dir(at)
            .output()
            .expect("git")
            .stdout,
    )
    .trim()
    .to_string()
}

fn store(at: &Path) -> String {
    at.join("store").to_string_lossy().into_owned()
}

#[test]
fn a_repository_with_nothing_to_build_reads_the_same() {
    // A finished paper, work from before soma: it has a history, a journal and
    // reasoning worth reading, and no graph to probe. Requiring `build` to read
    // them tied layer 2 to layer 1 in the wrong place — by the configuration
    // and not by the facts.
    given!(at);
    std::fs::write(at.join("soma-tree.toml"), "tree = \"terminada\"\n").expect("the config");

    let said = somatize_tree(at, &["log", "--store", &store(at), "HEAD~3..HEAD"]);

    assert!(
        said.contains("no probe"),
        "and said, not left blank: {said}"
    );
    assert!(
        said.matches("\n").count() > 3,
        "the stops are still there: {said}"
    );
}

#[test]
fn and_what_does_need_a_probe_says_that_it_is_missing() {
    given!(at);
    std::fs::write(at.join("soma-tree.toml"), "tree = \"terminada\"\n").expect("the config");

    let said = soma_tree_refusing(at, &["diff", "HEAD~1", "HEAD", "--store", &store(at)]);

    assert!(said.contains("build"), "{said}");
    assert!(
        said.contains("reasoning"),
        "and where it can be read: {said}"
    );
}

/// A network written across three modules and assembled in an `__init__`,
/// which is the case the example's fixture does not have: there everything
/// fits in one file, and `code` and `reaches` say the same thing.
fn spread_across_files() -> tempfile::TempDir {
    let at = tempfile::tempdir().expect("a temporary directory");
    let net = at.path().join("experiments");
    std::fs::create_dir_all(&net).expect("a package");
    std::fs::write(
        net.join("parts.py"),
        "WIDTH = 32\n\n\nclass Router:\n    def route(self, x):\n        return x[:WIDTH]\n",
    )
    .expect("a file");
    std::fs::write(
        net.join("head.py"),
        "from experiments.parts import Router\n\n\nclass Head:\n    def __init__(self):\n        \
         self.router = Router()\n\n    def run(self, x):\n        return self.router.route(x)\n",
    )
    .expect("a file");
    std::fs::write(
        net.join("encoder.py"),
        "from somatize import Graph, Node\n\nfrom experiments.head import Head\n\n\n\
         class Encoder(Node):\n    def __init__(self):\n        self.net = Head()\n\n    \
         def forward(self, x, ctx):\n        return self.net.run(x)\n\n\n\
         def build():\n    return Graph.somatize(Encoder().named(\"encoder\"))\n",
    )
    .expect("a file");
    at
}

fn probed_reaches(python: &Path, at: &Path) -> serde_json::Value {
    let said = Command::new(python)
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/soma_tree_probe.py"))
        .args(["--build", "experiments.encoder:build", "--commit", "HEAD"])
        .current_dir(at)
        .output()
        .expect("the probe runs");
    assert!(
        said.status.success(),
        "the probe refused: {}",
        String::from_utf8_lossy(&said.stderr)
    );
    let answer: serde_json::Value =
        serde_json::from_slice(&said.stdout).expect("the probe writes json");
    answer["reaches"].clone()
}

#[test]
fn a_node_says_every_file_its_network_is_written_across() {
    // What was missing. A node **is** its class, so `inspect.getsourcefile`
    // knows one of the three files and the panel showed that one; the other
    // two were nowhere in the answer, and had been inside the fingerprint from
    // the first day.
    given!(python, unused);
    let at = spread_across_files();

    let reaches = probed_reaches(&python, at.path());
    let files: Vec<&str> = reaches["encoder"]["files"]
        .as_array()
        .expect("the files")
        .iter()
        .map(|one| one["file"].as_str().expect("a path"))
        .collect();

    assert_eq!(
        files,
        [
            "experiments/encoder.py",
            "experiments/head.py",
            "experiments/parts.py"
        ],
        "all three, and relative to the checkout",
    );
}

#[test]
fn each_file_says_which_definitions_of_it_the_node_reaches() {
    // A file usually holds four classes and the node reaches one. The whole
    // file is shown — it is what gets edited — and which class arrived is said,
    // because otherwise the box says *this node depends on this file* and keeps
    // quiet about the half that matters.
    given!(python, unused);
    let at = spread_across_files();

    let reaches = probed_reaches(&python, at.path());
    let of = |file: &str| -> Vec<String> {
        reaches["encoder"]["files"]
            .as_array()
            .expect("the files")
            .iter()
            .find(|one| one["file"] == file)
            .unwrap_or_else(|| panic!("`{file}` is not there"))["defs"]
            .as_array()
            .expect("the definitions")
            .iter()
            .map(|one| one["called"].as_str().expect("a name").to_string())
            .collect()
    };

    assert_eq!(of("experiments/parts.py"), ["Router"]);
    assert_eq!(of("experiments/head.py"), ["Head"]);
    assert_eq!(of("experiments/encoder.py"), ["Encoder"]);
}

#[test]
fn a_file_that_stops_being_reached_leaves_the_answer() {
    // Not a second idea of what depends on what: it is the closure the
    // fingerprint already walked. So it stops naming a file exactly when it
    // stops being in the version, and not one edit later.
    given!(python, unused);
    let at = spread_across_files();
    let head = at.path().join("experiments/head.py");
    std::fs::write(
        &head,
        "class Head:\n    def run(self, x):\n        return x\n",
    )
    .expect("an edit");

    let reaches = probed_reaches(&python, at.path());
    let files: Vec<&str> = reaches["encoder"]["files"]
        .as_array()
        .expect("the files")
        .iter()
        .map(|one| one["file"].as_str().expect("a path"))
        .collect();

    assert!(
        !files.contains(&"experiments/parts.py"),
        "nobody reaches `parts.py` any more: {files:?}",
    );
}
