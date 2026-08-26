//! What folds when a line is drawn, which is the only part of a walk that is a
//! rule rather than a walk.

use somatize_tree::journal::Verdict;
use somatize_tree::moves::Course;
use somatize_tree::walk::folds;

#[test]
fn what_somebody_decided_to_drop_folds() {
    assert!(folds(Some(Course::Abandon), None, false));
    assert!(folds(Some(Course::Superseded), None, false));
}

#[test]
fn what_nobody_decided_does_not_fold() {
    // Everything is drawn by default. Folding is the answer to a tree of forty
    // variants not reading, not to anything being spare.
    assert!(!folds(None, None, false));
    assert!(!folds(Some(Course::Pursue), None, false));
}

#[test]
fn what_somebody_judged_wrong_does_not_fold() {
    // The one that matters most. An `invalid` is what casts doubt on the
    // measurement the decision to abandon leaned on: hiding it would hide the
    // very reason to look at it again.
    assert!(!folds(Some(Course::Abandon), Some(Verdict::Invalid), false));
}

#[test]
fn nor_what_inherits_that_doubt() {
    // The same reason one level down, and what stops this being decidable from
    // what was written about this commit alone.
    assert!(!folds(Some(Course::Abandon), None, true));
}

#[test]
fn having_looked_and_found_nothing_does_not_unfold_it() {
    // `sound` says somebody looked and found nothing wrong, so there is no new
    // reason to come back: the decision to abandon stands.
    assert!(folds(Some(Course::Abandon), Some(Verdict::Sound), false));
}
