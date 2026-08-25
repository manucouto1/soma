//! What somebody said, against a store in a temporary directory.
//!
//! A real `Local` and not a double: what is being defended is that nothing is
//! ever lost and that the last word wins, and both of those are claims about
//! how a store behaves under two writers. A double would agree with the belief
//! rather than with the store.

use soma_next_store::Local;
use soma_tree::journal::{Journal, Verdict};

fn somewhere() -> (tempfile::TempDir, Local) {
    let at = tempfile::tempdir().expect("a temporary directory");
    let kept = Local::at(at.path()).expect("a store in it");
    (at, kept)
}

#[test]
fn a_commit_nobody_judged_has_no_verdict() {
    let (_at, kept) = somewhere();
    let journal = Journal::of("t", &kept);

    assert!(journal.verdicts().expect("a scan").is_empty());
}

#[test]
fn what_a_verdict_is_means_the_last_one_anybody_claimed() {
    // The whole reason this is append-only. Changing your mind is saying
    // something else, not editing what you said.
    let (_at, kept) = somewhere();
    let journal = Journal::of("t", &kept);

    journal
        .say("abc", Some(Verdict::Invalid), "me", "the split lied")
        .unwrap();
    journal
        .say(
            "abc",
            Some(Verdict::Sound),
            "me",
            "the split was fine, I misread",
        )
        .unwrap();

    assert_eq!(
        journal.verdicts().unwrap().get("abc"),
        Some(&Verdict::Sound)
    );
}

#[test]
fn and_the_one_before_it_is_still_there() {
    // An `invalid` that turned out to be a misreading is the most instructive
    // thing in an investigation. Overwriting it would throw that away.
    let (_at, kept) = somewhere();
    let journal = Journal::of("t", &kept);
    journal
        .say("abc", Some(Verdict::Invalid), "me", "recall is impossible")
        .unwrap();
    journal
        .say(
            "abc",
            Some(Verdict::Sound),
            "me",
            "I was reading the wrong split",
        )
        .unwrap();

    let said = journal.all().unwrap();

    assert_eq!(said.len(), 2);
    assert_eq!(said[0].verdict, Some(Verdict::Invalid));
    assert_eq!(journal.read(&said[0]).unwrap(), "recall is impossible");
}

#[test]
fn a_note_does_not_overwrite_a_verdict() {
    // Writing down what you saw is not thereby changing your mind.
    let (_at, kept) = somewhere();
    let journal = Journal::of("t", &kept);
    journal
        .say("abc", Some(Verdict::Invalid), "me", "no")
        .unwrap();

    journal.say("abc", None, "me", "recall was 0.61").unwrap();

    assert_eq!(
        journal.verdicts().unwrap().get("abc"),
        Some(&Verdict::Invalid)
    );
}

#[test]
fn nobody_saying_something_at_the_same_time_loses_it() {
    // The property the store's `claim` is for, and the reason this is not a
    // row somebody updates: two machines on one NFS mount both get heard.
    let (_at, kept) = somewhere();
    let kept = &kept;

    std::thread::scope(|scope| {
        for which in 0..8 {
            scope.spawn(move || {
                Journal::of("t", kept)
                    .say("abc", None, "me", &format!("saw {which}"))
                    .unwrap();
            });
        }
    });

    let journal = Journal::of("t", kept);
    let mut seen: Vec<String> = journal
        .all()
        .unwrap()
        .iter()
        .map(|saying| journal.read(saying).unwrap())
        .collect();
    seen.sort();

    assert_eq!(seen.len(), 8, "eight writers, eight slots, none lost");
    assert_eq!(seen[0], "saw 0");
}

#[test]
fn two_investigations_in_one_store_do_not_see_each_other() {
    // A store holds whatever anybody put in it, which is why `tree` is in the
    // name and why reading is a question rather than an assumption.
    let (_at, kept) = somewhere();
    Journal::of("one", &kept)
        .say("abc", Some(Verdict::Invalid), "me", "mine")
        .unwrap();

    assert!(Journal::of("other", &kept).verdicts().unwrap().is_empty());
}

#[test]
fn only_being_wrong_reaches_down() {
    // Having looked and found nothing does not put anybody's numbers in doubt.
    assert!(Verdict::Invalid.reaches_down());
    assert!(!Verdict::Sound.reaches_down());
}

#[test]
fn a_verdict_nobody_defined_is_refused_rather_than_guessed_at() {
    assert_eq!(Verdict::read("invalid"), Some(Verdict::Invalid));
    assert_eq!(Verdict::read("invalidd"), None);
    assert_eq!(Verdict::read(""), None);
}

#[test]
fn una_palabra_retirada_deja_la_prosa_y_pierde_sólo_el_veredicto() {
    // La migración entera. Un registro viejo que dice `verdict=dead-end` ya no
    // se lee como veredicto, así que vuelve como nota —con lo escrito intacto,
    // que era siempre la parte que valía— en vez de romper la lectura o, peor,
    // colarse como si aún significara algo.
    for retirada in ["promising", "dead-end", "superseded"] {
        assert_eq!(
            Verdict::read(retirada),
            None,
            "{retirada} se fue a la capa 2"
        );
    }
}
