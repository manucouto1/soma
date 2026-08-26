//! The reasoning read back: names instead of ids, and what folds.
//!
//! Against a real `Local`, like `moves`: what is defended is a derivation over
//! what a store holds, and a double would be defending the double.

use somatize_store::Local;
use somatize_tree::moves::{Cited, Course, Kind, Moves, Said, Says, Scope, Standing, Writing};
use somatize_tree::reasoning::{Reasoning, outlined, reasoned};

fn somewhere() -> (tempfile::TempDir, Local) {
    let at = tempfile::tempdir().expect("a temporary directory");
    let kept = Local::at(at.path()).expect("a store inside");
    (at, kept)
}

fn named(moves: &Moves, kind: Kind, name: &str, prose: &str) -> u32 {
    moves
        .add(Writing::new(kind, name, prose, "me"))
        .expect("a move")
}

fn deciding(moves: &Moves, name: &str, course: Course, about: [u32; 1], why: &str) -> u32 {
    let mut writing = Writing::new(Kind::Decision, name, why, "me");
    writing.course = Some(course);
    writing.scope = Scope::of(about);
    moves.add(writing).expect("a decision")
}

fn seen<'a>(read: &'a Reasoning, name: &str) -> &'a somatize_tree::reasoning::Seen {
    read.went(name).expect("a move of that name")
}

#[test]
fn every_cross_reference_comes_back_as_a_name() {
    // The id stops identifying a move the moment nobody holds it in a variable,
    // and reading a store back is exactly that moment.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = named(
        &moves,
        Kind::Question,
        "why",
        "is the encoder the bottleneck?",
    );
    let a = named(&moves, Kind::Attempt, "wider", "twice the width");
    moves.hang(a, q).expect("hung");

    let read = reasoned("t", &kept).expect("read back");

    assert_eq!(seen(&read, "wider").under, ["why"]);
    assert_eq!(seen(&read, "why").id, q);
}

#[test]
fn a_standing_is_carried_by_the_move_it_is_about_and_nothing_else_has_one() {
    // `None` and not `open`: an attempt is not a question nobody answered.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = named(
        &moves,
        Kind::Hypothesis,
        "wider-helps",
        "width is the bottleneck",
    );
    let f = named(&moves, Kind::Finding, "it-did", "recall moved four points");
    named(&moves, Kind::Attempt, "wider", "twice the width");
    moves
        .say(Said {
            from: f,
            to: h,
            says: Says::Validates,
            scope: Scope::everything(),
            in_part: false,
        })
        .expect("said");

    let read = reasoned("t", &kept).expect("read back");

    assert_eq!(
        seen(&read, "wider-helps").standing,
        Some(Standing::Validated)
    );
    assert_eq!(seen(&read, "wider").standing, None);
    assert_eq!(seen(&read, "it-did").standing, None);
}

#[test]
fn an_attempt_nobody_ran_is_abandoned_although_it_cites_no_commit() {
    // The half `decided` cannot answer: it maps commits, and the move a
    // decision most needs to be able to abandon is one that never ran.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let a = named(
        &moves,
        Kind::Attempt,
        "per-length",
        "a threshold per length",
    );
    deciding(
        &moves,
        "drop-per-length",
        Course::Abandon,
        [a],
        "never ran: it is a second model",
    );

    let read = reasoned("t", &kept).expect("read back");

    assert!(seen(&read, "per-length").pruned);
    assert!(moves.decided().expect("decided").is_empty());
    let folded = &read.folded[0];
    assert_eq!(folded.root, "per-length");
    assert_eq!(folded.by, "drop-per-length");
    assert_eq!(folded.hides, ["per-length"]);
    assert!(folded.why.contains("second model"));
}

#[test]
fn what_folds_says_how_many_it_hides_and_the_reason_in_words() {
    // Pruning that does not say why is deletion with a nicer name.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = named(
        &moves,
        Kind::Question,
        "xattn",
        "does cross-attention help?",
    );
    for one in ["xattn-a", "xattn-b"] {
        let a = named(&moves, Kind::Attempt, one, "tried it");
        moves.hang(a, q).expect("hung");
    }
    deciding(
        &moves,
        "drop-xattn",
        Course::Abandon,
        [q],
        "it costs more than it gives",
    );

    let read = reasoned("t", &kept).expect("read back");

    let folded = &read.folded[0];
    assert_eq!(folded.hides, ["xattn", "xattn-a", "xattn-b"]);
    assert!(read.moves.iter().filter(|one| one.pruned).count() == 3);
}

#[test]
fn taking_a_line_up_again_unfolds_it_and_yesterdays_reason_is_still_written() {
    // Deciding again is how you change your mind, and the later decision is
    // the one that counts — without the earlier one being deleted.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let a = named(&moves, Kind::Attempt, "xattn", "cross-attention");
    deciding(&moves, "drop-it", Course::Abandon, [a], "too slow");
    deciding(
        &moves,
        "back-to-it",
        Course::Pursue,
        [a],
        "the profiler said otherwise",
    );

    let read = reasoned("t", &kept).expect("read back");

    assert!(!seen(&read, "xattn").pruned);
    assert!(read.folded.is_empty());
    assert!(seen(&read, "drop-it").prose.contains("too slow"));
}

#[test]
fn a_decision_that_hangs_nowhere_belongs_beside_what_it_abandons() {
    // Its scope is the only thing tying it to the line it ended, so a reader
    // with only `under` would draw it floating on its own.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let a = named(&moves, Kind::Attempt, "xattn", "cross-attention");
    deciding(&moves, "drop-it", Course::Abandon, [a], "too slow");

    let read = reasoned("t", &kept).expect("read back");

    assert_eq!(seen(&read, "drop-it").under, [] as [String; 0]);
    assert_eq!(seen(&read, "drop-it").about, ["xattn"]);
    assert_eq!(
        read.below("xattn")
            .iter()
            .map(|one| &one.name)
            .collect::<Vec<_>>(),
        ["drop-it"]
    );
}

#[test]
fn a_scope_covers_a_dag_and_not_a_subtree() {
    // `under` is multivalued, so the one move that hangs under two questions is
    // reached from both — which a reader walking a tree by hand would miss.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let one = named(
        &moves,
        Kind::Question,
        "capacity",
        "does more capacity help?",
    );
    let two = named(&moves, Kind::Question, "reading", "does it read better?");
    let both = named(&moves, Kind::Attempt, "both", "wider and deeper at once");
    moves.hang(both, one).expect("hung");
    moves.hang(both, two).expect("hung");

    let read = reasoned("t", &kept).expect("read back");

    assert_eq!(
        read.covers(&["capacity".into()]).unwrap(),
        ["capacity", "both"]
    );
    assert_eq!(
        read.covers(&["reading".into()]).unwrap(),
        ["reading", "both"]
    );
    // Which is what makes *do these two scopes touch* an intersection.
    assert_eq!(
        read.covers(&["capacity".into(), "reading".into()]).unwrap(),
        ["capacity", "reading", "both"]
    );
}

#[test]
fn a_move_nobody_hung_anywhere_is_still_in_the_outline() {
    // Work waiting for a place, not a move that hides.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    named(
        &moves,
        Kind::Question,
        "loose-end",
        "does the checkpoint matter?",
    );

    let read = reasoned("t", &kept).expect("read back");

    let lines = outlined(&read, None, false);
    assert!(
        lines
            .iter()
            .any(|line| line.starts_with("loose-end · question · open"))
    );
}

#[test]
fn an_outline_folds_an_abandoned_line_and_all_lines_opens_it() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = named(
        &moves,
        Kind::Question,
        "xattn",
        "does cross-attention help?",
    );
    let a = named(&moves, Kind::Attempt, "xattn-a", "tried it");
    moves.hang(a, q).expect("hung");
    deciding(
        &moves,
        "drop-xattn",
        Course::Abandon,
        [q],
        "it costs more than it gives",
    );

    let read = reasoned("t", &kept).expect("read back");

    let folded = outlined(&read, None, false);
    assert!(
        folded
            .iter()
            .any(|line| line.contains("⋯ 2 folded · abandon"))
    );
    assert!(!folded.iter().any(|line| line.contains("xattn-a")));
    // And nothing was deleted: a line that did not work is the most reusable
    // thing an investigation produces.
    let all = outlined(&read, None, true);
    assert!(all.iter().any(|line| line.contains("xattn-a")));
}

#[test]
fn a_move_under_two_parents_is_written_under_both_and_walked_once() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let one = named(
        &moves,
        Kind::Question,
        "capacity",
        "does more capacity help?",
    );
    let two = named(&moves, Kind::Question, "reading", "does it read better?");
    let both = named(&moves, Kind::Attempt, "both", "wider and deeper at once");
    let under = named(&moves, Kind::Finding, "cancelled", "together they cancel");
    moves.hang(both, one).expect("hung");
    moves.hang(both, two).expect("hung");
    moves.hang(under, both).expect("hung");

    let lines = outlined(&reasoned("t", &kept).expect("read back"), None, false);

    assert_eq!(
        lines
            .iter()
            .filter(|line| line.contains("both · attempt"))
            .count(),
        2
    );
    assert_eq!(
        lines.iter().filter(|line| line.contains("(again)")).count(),
        1
    );
    assert_eq!(
        lines
            .iter()
            .filter(|line| line.contains("cancelled"))
            .count(),
        1
    );
}

#[test]
fn what_an_attempt_cites_comes_back_whole_so_the_way_back_needs_no_index() {
    // Both halves: a commit is only half of what ran, and the same one under
    // two configurations is two experiments.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let mut writing = Writing::new(
        Kind::Attempt,
        "decorr-0.1",
        "the decorrelation weight at 0.1",
        "me",
    );
    writing.cites = vec![
        Cited {
            what: "commit".into(),
            id: "3847d0c1".into(),
        },
        Cited {
            what: "config".into(),
            id: "sha256_abc".into(),
        },
    ];
    moves.add(writing).expect("an attempt");

    let read = reasoned("t", &kept).expect("read back");

    let cites = &seen(&read, "decorr-0.1").cites;
    assert_eq!(cites[0].id, "3847d0c1");
    assert_eq!(cites[1].what, "config");
}
