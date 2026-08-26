//! A version's trials, read as soma writes them.
//!
//! The records are written **by hand and with soma's names** rather than by
//! calling its `take`/`report`: what has to be defended is that this reader
//! understands that format, and calling the other library's writer would have
//! the two agree on any format at all, including one nobody has on disk. The
//! format is documented in `study/_run.py`, and this is that documentation
//! made executable.

use somatize_store::{Local, Store};
use somatize_tree::trials::{Goal, Trials};

fn somewhere() -> (tempfile::TempDir, Local) {
    let at = tempfile::tempdir().expect("a temporary directory");
    let kept = Local::at(at.path()).expect("a store inside");
    (at, kept)
}

/// A trial as soma binds it: `<study>/trial/<n>/<attempt>`, with the state and
/// the score in the **record** and the curve in the blob.
fn ran(
    kept: &Local,
    commit: &str,
    trial: u32,
    attempt: u32,
    state: &str,
    score: Option<f64>,
    reports: &[f64],
) {
    let blob = serde_json::json!({
        "point": "lr=0.001,batch=32",
        "reports": reports,
        "state": state,
        "because": if state == "pruned" { Some("it was not improving") } else { None },
        "took": 12.5,
    });
    let digest = kept.put(blob.to_string().as_bytes()).expect("the blob");
    let mut meta: Vec<(String, String)> = vec![
        ("state".into(), state.into()),
        ("point".into(), "lr=0.001,batch=32".into()),
        ("who".into(), "machine-3".into()),
    ];
    if let Some(score) = score {
        // `repr(float(score))`, which is what Python writes.
        meta.push(("score".into(), format!("{score:?}")));
    }
    kept.bind(
        &format!("exp/t/{commit}/trial/{trial}/{attempt}"),
        &digest,
        meta,
    )
    .expect("the record");
}

#[test]
fn a_commits_study_is_the_prefix_its_journal_already_lives_under() {
    // The whole coupling with soma is this name, and it is the one part of
    // this that cannot be changed later without moving somebody's directories.
    let (_at, kept) = somewhere();

    assert_eq!(Trials::of("t", &kept).study("abc123"), "exp/t/abc123");
}

#[test]
fn a_versions_trials_come_back_in_one_scan_with_no_fetches() {
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.83), &[0.4, 0.7, 0.83]);
    ran(&kept, "abc", 1, 0, "running", None, &[0.3]);

    let seen = Trials::of("t", &kept).of_commit("abc").unwrap();

    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].state.as_deref(), Some("done"));
    assert_eq!(seen[0].score, Some(0.83));
    assert_eq!(seen[0].who.as_deref(), Some("machine-3"));
    assert_eq!(seen[1].score, None, "still running, still no score");
}

#[test]
fn the_highest_attempt_wins_because_claiming_is_a_link() {
    // A trial whose machine died stays claimed for ever, and rescuing it by
    // writing over it would be a race. The rescue is claiming the next
    // attempt, and whoever reads keeps the highest.
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "running", None, &[0.2]);
    ran(&kept, "abc", 0, 1, "done", Some(0.91), &[0.2, 0.6, 0.91]);

    let seen = Trials::of("t", &kept).of_commit("abc").unwrap();

    assert_eq!(seen.len(), 1, "one trial, not two");
    assert_eq!(seen[0].attempt, 1);
    assert_eq!(seen[0].score, Some(0.91));
}

#[test]
fn another_versions_trials_are_not_this_ones() {
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.8), &[0.8]);
    ran(&kept, "def", 0, 0, "done", Some(0.9), &[0.9]);

    assert_eq!(Trials::of("t", &kept).of_commit("abc").unwrap().len(), 1);
}

#[test]
fn what_is_in_the_store_and_is_not_a_trial_is_asked_and_not_assumed() {
    // A store holds whatever anybody put in it: a cache, the journal, the reasoning.
    let (_at, kept) = somewhere();
    let digest = kept.put(b"something").unwrap();
    kept.bind("exp/t/abc/said/0", &digest, Vec::new()).unwrap();
    kept.bind("exp/t/move/3/said/0", &digest, Vec::new())
        .unwrap();
    kept.bind("other-thing/trial/0/0", &digest, Vec::new())
        .unwrap();
    ran(&kept, "abc", 0, 0, "done", Some(0.8), &[0.8]);

    assert_eq!(Trials::of("t", &kept).of_commit("abc").unwrap().len(), 1);
    assert_eq!(Trials::of("t", &kept).counted().unwrap().len(), 1);
}

#[test]
fn counting_forty_versions_costs_one_scan_and_not_forty() {
    // What the rail needs. Asking commit by commit would be forty walks of the
    // store to draw a list of forty rows.
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.83), &[0.83]);
    ran(&kept, "abc", 1, 0, "failed", None, &[]);
    ran(&kept, "abc", 2, 0, "running", None, &[0.1]);
    ran(&kept, "def", 0, 0, "pruned", Some(0.4), &[0.4]);

    let counted = Trials::of("t", &kept).counted().unwrap();

    assert_eq!(counted["abc"].trials, 3);
    assert_eq!(counted["abc"].done, 1);
    assert_eq!(counted["abc"].failed, 1);
    assert_eq!(counted["abc"].running, 1);
    assert_eq!(counted["def"].pruned, 1);
}

#[test]
fn a_rescued_trial_is_counted_once() {
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "running", None, &[0.2]);
    ran(&kept, "abc", 0, 1, "done", Some(0.9), &[0.9]);

    let counted = Trials::of("t", &kept).counted().unwrap();

    assert_eq!(counted["abc"].trials, 1);
    assert_eq!(counted["abc"].done, 1);
    assert_eq!(
        counted["abc"].running, 0,
        "the dead attempt is not still alive"
    );
}

#[test]
fn with_no_direction_declared_it_does_not_say_which_is_best() {
    // The one that matters most. Whether `0.0837` is good depends on whether
    // that metric is maximised or minimised, and that direction is in no
    // record: it lives in the `Goal` handed to the sampler. *Best* is the word
    // most often copied into a report unchecked, so either it is known or it
    // is not said.
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.83), &[0.83]);
    ran(&kept, "abc", 1, 0, "done", Some(0.21), &[0.21]);

    let counted = Trials::of("t", &kept).counted().unwrap();

    assert_eq!(counted["abc"].best, None);
    assert_eq!(counted["abc"].lowest, Some(0.21), "the range is true");
    assert_eq!(counted["abc"].highest, Some(0.83));
}

#[test]
fn with_the_direction_declared_it_does() {
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.83), &[0.83]);
    ran(&kept, "abc", 1, 0, "done", Some(0.21), &[0.21]);

    let maximizing = Trials::of("t", &kept).towards(Some(Goal::Max));
    let minimizing = Trials::of("t", &kept).towards(Some(Goal::Min));

    assert_eq!(maximizing.counted().unwrap()["abc"].best, Some(0.83));
    assert_eq!(minimizing.counted().unwrap()["abc"].best, Some(0.21));
}

#[test]
fn a_pruned_score_does_not_enter_the_range() {
    // It is real and it is not comparable: measured after fewer epochs.
    // Including it would make the range wider than anything anybody measured,
    // and a range that exaggerates is worse than no range.
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.83), &[0.83]);
    ran(&kept, "abc", 1, 0, "pruned", Some(0.05), &[0.05]);

    let counted = Trials::of("t", &kept).counted().unwrap();

    assert_eq!(counted["abc"].lowest, Some(0.83));
    assert_eq!(counted["abc"].pruned, 1, "counted, but out of the range");
}

#[test]
fn the_curve_is_paid_for_apart_and_says_why_it_stopped() {
    // The other side of the cost rule: the curve grows, so it lives in the
    // blob and only whoever asks to see it reads it.
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "pruned", Some(0.4), &[0.1, 0.3, 0.4]);
    let trials = Trials::of("t", &kept);
    let seen = trials.of_commit("abc").unwrap();

    let curve = trials.curve(&seen[0]).unwrap().expect("the curve");

    assert_eq!(curve.reports, vec![0.1, 0.3, 0.4]);
    assert_eq!(curve.because.as_deref(), Some("it was not improving"));
    assert_eq!(curve.took, Some(12.5));
}

#[test]
fn a_version_with_no_trials_is_not_an_error() {
    let (_at, kept) = somewhere();

    assert!(Trials::of("t", &kept).of_commit("abc").unwrap().is_empty());
    assert!(Trials::of("t", &kept).counted().unwrap().is_empty());
}

#[test]
fn only_a_done_is_comparable_with_another_done() {
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.8), &[0.8]);
    ran(&kept, "abc", 1, 0, "pruned", Some(0.4), &[0.4]);
    ran(&kept, "abc", 2, 0, "running", None, &[0.1]);
    let seen = Trials::of("t", &kept).of_commit("abc").unwrap();

    assert!(seen[0].comparable());
    assert!(!seen[1].comparable(), "it was measured after fewer epochs");
    assert!(!seen[2].comparable());
}
