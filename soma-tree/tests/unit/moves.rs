//! The reasoning, against a store in a temporary directory.
//!
//! A real `Local` and not a double: what is defended here is that two people
//! writing at once get two moves, and that is a claim about how a store
//! behaves under two writers.

use somatize_store::Local;
use somatize_tree::moves::{
    Cited, Course, Kind, Move, Moves, Said, Says, Scope, Standing, Trouble, Writing,
};

fn somewhere() -> (tempfile::TempDir, Local) {
    let at = tempfile::tempdir().expect("a temporary directory");
    let kept = Local::at(at.path()).expect("a store inside");
    (at, kept)
}

/// Adds a move covering everything and citing nothing, the ordinary case.
///
/// The name is made up here rather than taken from the prose: two moves in one
/// test often say nearly the same thing, and a slug of that would collide and
/// report as a bug in naming rather than in the test.
fn plain(moves: &Moves, kind: Kind, prose: &str) -> u32 {
    named(moves, kind, &a_name(), prose)
}

/// A name nobody else in this binary will take.
///
/// Made up rather than slugged from the prose: two moves in one test often say
/// nearly the same thing, and a collision there would report as a bug in
/// naming rather than in the test that wrote it.
fn a_name() -> String {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    format!(
        "m{}",
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

/// The same, with a name somebody chose.
fn named(moves: &Moves, kind: Kind, name: &str, prose: &str) -> u32 {
    moves
        .add(Writing::new(kind, name, prose, "me"))
        .expect("a move")
}

#[test]
fn a_question_nobody_tried_is_a_move_like_any_other() {
    // The only kind that can exist with nothing under it. A question nobody
    // has attacked has nowhere to live otherwise, and that is outstanding work
    // going missing.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);

    let q = plain(&moves, Kind::Question, "is the encoder the bottleneck?");

    assert_eq!(moves.all().unwrap()[&q].kind, Kind::Question);
    assert_eq!(moves.standing().unwrap()[&q], Standing::Open);
}

#[test]
fn a_validates_pointing_at_an_attempt_means_nothing_and_is_refused() {
    // Accepting it would store a sentence nobody can read afterwards.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let f = plain(&moves, Kind::Finding, "recall does not move");
    let a = plain(&moves, Kind::Attempt, "tried it at scale 2.0");

    let said = moves.say(Said {
        from: f,
        to: a,
        says: Says::Validates,
        scope: Scope::everything(),
        in_part: false,
    });

    assert!(said.is_err(), "{said:?}");
}

#[test]
fn answering_and_validating_are_not_the_same_verb() {
    // Folding hypothesis into question erased exactly this: one gets answered,
    // the other gets validated or refuted.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "why does recall fall?");
    let h = plain(&moves, Kind::Hypothesis, "the tokenizer splits badly");
    let f = plain(
        &moves,
        Kind::Finding,
        "with another tokenizer it is the same",
    );

    assert!(
        moves
            .say(Said {
                from: f,
                to: h,
                says: Says::Answers,
                scope: Scope::everything(),
                in_part: false
            })
            .is_err(),
        "a hypothesis does not get answered",
    );
    assert!(
        moves
            .say(Said {
                from: f,
                to: q,
                says: Says::Refutes,
                scope: Scope::everything(),
                in_part: false
            })
            .is_err(),
        "a question does not get refuted",
    );
}

#[test]
fn three_partial_answers_push_a_question_without_closing_it() {
    // *Does more capacity help?* is not settled at once: three attempts get
    // generated and each answers part. Neither open nor closed.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(
        &moves,
        Kind::Question,
        "if I add capacity, does it improve?",
    );

    for said in ["×2 yes", "×4 yes", "×8 stalls"] {
        let f = plain(&moves, Kind::Finding, said);
        moves
            .say(Said {
                from: f,
                to: q,
                says: Says::Answers,
                scope: Scope::everything(),
                in_part: true,
            })
            .expect("a partial answer");
    }

    assert_eq!(moves.standing().unwrap()[&q], Standing::Partly);
}

#[test]
fn one_that_settles_it_is_enough_to_call_it_answered() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "does it improve?");
    let a = plain(&moves, Kind::Finding, "in part");
    let b = plain(&moves, Kind::Finding, "fully, and for this reason");

    for (from, in_part) in [(a, true), (b, false)] {
        moves
            .say(Said {
                from,
                to: q,
                says: Says::Answers,
                scope: Scope::everything(),
                in_part,
            })
            .expect("an answer");
    }

    assert_eq!(moves.standing().unwrap()[&q], Standing::Answered);
}

#[test]
fn validating_and_refuting_over_different_situations_is_no_contradiction() {
    // The combination case: A alone worked, A+B cancel out. Two facts about
    // two situations. Counting them as a conflict would call the most
    // instructive thing in an investigation a contradiction.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "more capacity improves it");
    let a = plain(&moves, Kind::Attempt, "variant A");
    let ab = plain(&moves, Kind::Attempt, "variant A + B");
    let alone = plain(&moves, Kind::Finding, "A only: it improves");
    let together = plain(&moves, Kind::Finding, "A+B: they cancel out");

    moves
        .say(Said {
            from: alone,
            to: h,
            says: Says::Validates,
            scope: Scope::of([a]),
            in_part: false,
        })
        .unwrap();
    moves
        .say(Said {
            from: together,
            to: h,
            says: Says::Refutes,
            scope: Scope::of([ab]),
            in_part: false,
        })
        .unwrap();

    assert_eq!(
        moves.standing().unwrap()[&h],
        Standing::Depends,
        "not a dispute and not half a validation: the answer depends on where you look",
    );
}

#[test]
fn depends_is_not_the_same_as_in_part() {
    // Came out of running the whole case: both used one word for two things.
    // A half-answered question is pushed along; a hypothesis that holds here
    // and not there has a conditional answer, which is the most informative
    // outcome an investigation gives.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "it improves");
    let a = plain(&moves, Kind::Attempt, "A");
    let f = plain(&moves, Kind::Finding, "in A, and only in part");

    moves
        .say(Said {
            from: f,
            to: h,
            says: Says::Validates,
            scope: Scope::of([a]),
            in_part: true,
        })
        .unwrap();

    assert_eq!(
        moves.standing().unwrap()[&h],
        Standing::PartlyValidated,
        "one sign in part is half a validation, not a conditional one",
    );
}

#[test]
fn and_over_the_same_situation_it_is() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "more capacity improves it");
    let a = plain(&moves, Kind::Attempt, "variant A");
    let (yes, no) = (
        plain(&moves, Kind::Finding, "it improves"),
        plain(&moves, Kind::Finding, "it does not improve"),
    );

    for (from, says) in [(yes, Says::Validates), (no, Says::Refutes)] {
        moves
            .say(Said {
                from,
                to: h,
                says,
                scope: Scope::of([a]),
                in_part: false,
            })
            .unwrap();
    }

    assert_eq!(moves.standing().unwrap()[&h], Standing::Disputed);
}

#[test]
fn a_scope_of_everything_touches_any_other() {
    // *This is false in general* does contradict *this holds for A*.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "more capacity improves it");
    let a = plain(&moves, Kind::Attempt, "variant A");
    let (yes, no) = (
        plain(&moves, Kind::Finding, "in A it improves"),
        plain(&moves, Kind::Finding, "it improves nowhere"),
    );

    moves
        .say(Said {
            from: yes,
            to: h,
            says: Says::Validates,
            scope: Scope::of([a]),
            in_part: false,
        })
        .unwrap();
    moves
        .say(Said {
            from: no,
            to: h,
            says: Says::Refutes,
            scope: Scope::everything(),
            in_part: false,
        })
        .unwrap();

    assert_eq!(moves.standing().unwrap()[&h], Standing::Disputed);
}

#[test]
fn a_scope_drags_along_whatever_hangs_under_its_root() {
    // *The whole encoder branch* is a root, not an enumeration. That is what
    // makes asking whether two scopes touch affordable.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let branch = plain(&moves, Kind::Attempt, "the encoder branch");
    let inside = plain(&moves, Kind::Attempt, "a step of that branch");
    moves.hang(inside, branch).unwrap();

    let h = plain(&moves, Kind::Hypothesis, "the encoder is the problem");
    let (yes, no) = (
        plain(&moves, Kind::Finding, "across the whole branch"),
        plain(&moves, Kind::Finding, "in that particular step"),
    );
    moves
        .say(Said {
            from: yes,
            to: h,
            says: Says::Validates,
            scope: Scope::of([branch]),
            in_part: false,
        })
        .unwrap();
    moves
        .say(Said {
            from: no,
            to: h,
            says: Says::Refutes,
            scope: Scope::of([inside]),
            in_part: false,
        })
        .unwrap();

    assert_eq!(
        moves.standing().unwrap()[&h],
        Standing::Disputed,
        "the step is inside the branch, so the scopes touch",
    );
}

#[test]
fn one_move_hangs_under_two_questions_at_once() {
    // The case that forces the DAG: the combination is about two answers
    // interacting and fits under neither question.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q1 = plain(&moves, Kind::Question, "does it improve interpretability?");
    let q2 = plain(&moves, Kind::Question, "does it improve performance?");
    let ab = plain(&moves, Kind::Attempt, "A + B");

    moves.hang(ab, q1).unwrap();
    moves.hang(ab, q2).unwrap();

    let mut of = moves.under().unwrap().parents_of(ab);
    of.sort();
    assert_eq!(of, vec![q1, q2]);
}

#[test]
fn combines_is_an_edge_from_attempt_to_attempt_and_is_not_hanging() {
    // It says this attempt **is** the composition of those, which is what lets
    // *each worked alone, together they cancel* be read.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let (a, b) = (
        plain(&moves, Kind::Attempt, "variant A"),
        plain(&moves, Kind::Attempt, "variant B"),
    );
    let ab = plain(&moves, Kind::Attempt, "A + B");

    for one in [a, b] {
        moves
            .say(Said {
                from: ab,
                to: one,
                says: Says::Combines,
                scope: Scope::everything(),
                in_part: false,
            })
            .expect("a combination");
    }

    let says = moves.says().unwrap();
    assert_eq!(
        says.iter().filter(|one| one.says == Says::Combines).count(),
        2
    );
    assert!(
        moves.under().unwrap().parents_of(ab).is_empty(),
        "combining is not hanging",
    );
}

#[test]
fn a_cycle_is_refused_when_written_and_not_when_walked() {
    // With `under` multivalued the shape can no longer be trusted, and a cycle
    // hangs every later walk — including the one that would draw it.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let (a, b, c) = (
        plain(&moves, Kind::Question, "a"),
        plain(&moves, Kind::Question, "b"),
        plain(&moves, Kind::Question, "c"),
    );
    moves.hang(b, a).unwrap();
    moves.hang(c, b).unwrap();

    assert!(moves.hang(a, c).is_err(), "a → b → c → a");
    assert!(moves.hang(a, a).is_err(), "not even with itself");
}

#[test]
fn an_attempt_cites_layer_one() {
    // The only kind that touches it, and what ties the reasoning to something
    // that can be run again.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);

    let a = moves
        .add(Writing {
            cites: vec![
                Cited {
                    what: "commit".into(),
                    id: "4910005c".into(),
                },
                Cited {
                    what: "trial".into(),
                    id: "exp/t/4910005c/trial/0/0".into(),
                },
            ],
            ..Writing::new(Kind::Attempt, &a_name(), "three scales", "me")
        })
        .unwrap();

    let body: Move = moves.all().unwrap().remove(&a).unwrap();
    assert_eq!(body.cites.len(), 2);
    assert_eq!(body.cites[0].id, "4910005c");
}

#[test]
fn rewording_the_prose_does_not_erase_what_came_before() {
    // As in the journal: the last wins, and the one before is still there.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let f = plain(&moves, Kind::Finding, "it does not improve");

    moves
        .reword(
            f,
            Some("it does not improve between 1.0 and 3.0"),
            None,
            None,
            "me",
        )
        .unwrap();

    assert_eq!(
        moves.all().unwrap()[&f].prose,
        "it does not improve between 1.0 and 3.0"
    );
}

#[test]
fn nobody_writing_at_the_same_time_loses_their_move() {
    // The property `claim` exists for, and why this is not a row somebody
    // updates: two machines over one NFS mount are both heard.
    let (_at, kept) = somewhere();
    let kept = &kept;

    std::thread::scope(|scope| {
        for which in 0..8 {
            scope.spawn(move || {
                Moves::of("t", kept)
                    .add(Writing::new(
                        Kind::Finding,
                        &format!("saw-{which}"),
                        &format!("I saw {which}"),
                        "me",
                    ))
                    .unwrap();
            });
        }
    });

    assert_eq!(Moves::of("t", kept).all().unwrap().len(), 8);
}

#[test]
fn two_investigations_in_one_store_do_not_see_each_other() {
    let (_at, kept) = somewhere();
    plain(&Moves::of("one", &kept), Kind::Question, "mine");

    assert!(Moves::of("another", &kept).all().unwrap().is_empty());
}

#[test]
fn saying_it_again_corrects_the_scope_instead_of_duplicating_it() {
    // Changing your mind about **where** a finding holds is the ordinary case:
    // believed general, turned out to be one branch's. If both edges survived,
    // widening a scope would count as saying it twice.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "more capacity improves it");
    let a = plain(&moves, Kind::Attempt, "variant A");
    let f = plain(&moves, Kind::Finding, "it improves");

    let mut said = Said {
        from: f,
        to: h,
        says: Says::Validates,
        scope: Scope::of(vec![a]),
        in_part: true,
    };
    moves.say(said.clone()).unwrap();
    said.scope = Scope::everything();
    said.in_part = false;
    moves.say(said).unwrap();

    let says = moves.says().unwrap();
    assert_eq!(
        says.len(),
        1,
        "the earlier one is still kept, but does not count"
    );
    assert!(says[0].scope.is_everything());
    assert!(!says[0].in_part);
}

#[test]
fn correcting_the_scope_leaves_what_was_said_with_another_verb() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "more capacity improves it");
    let f = plain(&moves, Kind::Finding, "depending on where");

    for says in [Says::Validates, Says::Refutes, Says::Validates] {
        moves
            .say(Said {
                from: f,
                to: h,
                says,
                scope: Scope::everything(),
                in_part: true,
            })
            .unwrap();
    }

    assert_eq!(
        moves.says().unwrap().len(),
        2,
        "validates and refutes, one of each"
    );
}

/// An attempt citing a commit, hung wherever it is told.
fn tried(moves: &Moves, prose: &str, commit: &str, under: &[u32]) -> u32 {
    let id = moves
        .add(Writing {
            cites: vec![Cited {
                what: "commit".into(),
                id: commit.into(),
            }],
            ..Writing::new(Kind::Attempt, &a_name(), prose, "me")
        })
        .unwrap();
    for parent in under {
        moves.hang(id, *parent).unwrap();
    }
    id
}

#[test]
fn abandoning_a_line_reaches_the_commits_its_attempts_cite() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "more capacity?");
    tried(&moves, "x2", "aaa", &[q]);

    let d = moves
        .add(Writing {
            scope: Scope::of(vec![q]),
            course: Some(Course::Abandon),
            ..Writing::new(Kind::Decision, &a_name(), "not this way", "me")
        })
        .unwrap();
    moves.hang(d, q).unwrap();

    assert_eq!(moves.decided().unwrap().get("aaa"), Some(&Course::Abandon));
}

#[test]
fn an_attempt_hung_after_the_decision_is_born_abandoned() {
    // The case that justifies deriving it rather than storing it: nobody goes
    // back to mark anything, and the whole line still reads dead.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "more capacity?");
    let d = moves
        .add(Writing {
            scope: Scope::of(vec![q]),
            course: Some(Course::Abandon),
            ..Writing::new(Kind::Decision, &a_name(), "not this way", "me")
        })
        .unwrap();
    moves.hang(d, q).unwrap();

    tried(&moves, "x4, to try it", "bbb", &[q]);

    assert_eq!(moves.decided().unwrap().get("bbb"), Some(&Course::Abandon));
}

#[test]
fn forking_off_an_abandoned_attempt_starts_clean() {
    // And this is the opposite, on purpose. Trying something else **because**
    // that did not work is the move you make at a dead end: inheriting the
    // abandonment down git ancestry would mark it as more of the same.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "more capacity?");
    let a = tried(&moves, "x2", "aaa", &[q]);
    let d = moves
        .add(Writing {
            scope: Scope::of(vec![a]),
            course: Some(Course::Abandon),
            ..Writing::new(Kind::Decision, &a_name(), "x2 leads nowhere", "me")
        })
        .unwrap();
    moves.hang(d, a).unwrap();

    // How `placed` hangs a fork: off the parents of whatever cited the
    // starting commit, not off that attempt itself.
    tried(&moves, "and what if depth instead of capacity", "bbb", &[q]);

    let decided = moves.decided().unwrap();
    assert_eq!(decided.get("aaa"), Some(&Course::Abandon));
    assert_eq!(decided.get("bbb"), None, "it is a sibling, not a child");
}

#[test]
fn a_decision_with_no_scope_is_about_where_it_hangs_not_the_tree() {
    // Without this, writing *this line is dead* while looking at one attempt
    // would mark the whole investigation, quietly. For a question, no scope
    // means about everything; for a decision it would be a trap.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let one = plain(&moves, Kind::Question, "capacity?");
    let another = plain(&moves, Kind::Question, "depth?");
    let a = tried(&moves, "x2", "aaa", &[one]);
    tried(&moves, "more layers", "bbb", &[another]);

    let d = moves
        .add(Writing {
            course: Some(Course::Abandon),
            ..Writing::new(Kind::Decision, &a_name(), "not this way", "me")
        })
        .unwrap();
    moves.hang(d, a).unwrap();

    let decided = moves.decided().unwrap();
    assert_eq!(decided.get("aaa"), Some(&Course::Abandon));
    assert_eq!(
        decided.get("bbb"),
        None,
        "the other question was none of its business"
    );
}

#[test]
fn a_decision_hung_off_nothing_with_no_scope_colours_nothing() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "capacity?");
    tried(&moves, "x2", "aaa", &[q]);
    moves
        .add(Writing {
            course: Some(Course::Abandon),
            ..Writing::new(Kind::Decision, &a_name(), "it has to be dropped", "me")
        })
        .unwrap();

    assert!(moves.decided().unwrap().is_empty());
}

#[test]
fn changing_your_mind_is_deciding_again_and_the_last_wins() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "capacity?");
    tried(&moves, "x2", "aaa", &[q]);
    for course in [Course::Abandon, Course::Pursue] {
        let d = moves
            .add(Writing {
                scope: Scope::of(vec![q]),
                course: Some(course),
                ..Writing::new(Kind::Decision, &a_name(), "…", "me")
            })
            .unwrap();
        moves.hang(d, q).unwrap();
    }

    assert_eq!(moves.decided().unwrap().get("aaa"), Some(&Course::Pursue));
}

#[test]
fn a_course_on_something_that_is_not_a_decision_is_refused() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);

    assert!(
        moves
            .add(Writing {
                course: Some(Course::Abandon),
                ..Writing::new(Kind::Finding, &a_name(), "I saw that it does not", "me")
            })
            .is_err()
    );
}

#[test]
fn correcting_a_decisions_scope_makes_it_reach_the_commits() {
    // The bug this closes came out of running it: a decision written while
    // looking at a finding reached the finding, and a finding is not a line —
    // nothing hangs off it and it cites no commit — so the decision reached
    // nowhere and the line went on reading alive. Without a correctable scope
    // there would be no fix but writing it again.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "capacity?");
    let a = tried(&moves, "x2", "aaa", &[q]);
    let f = plain(&moves, Kind::Finding, "latency goes up");
    moves.hang(f, a).unwrap();
    let d = moves
        .add(Writing {
            scope: Scope::of(vec![f]),
            course: Some(Course::Abandon),
            ..Writing::new(Kind::Decision, &a_name(), "we are not carrying on", "me")
        })
        .unwrap();
    moves.hang(d, f).unwrap();
    assert!(moves.decided().unwrap().is_empty(), "it gets nowhere");

    moves
        .reword(d, None, Some(Scope::of(vec![a])), None, "me")
        .unwrap();

    assert_eq!(moves.decided().unwrap().get("aaa"), Some(&Course::Abandon));
}

#[test]
fn correcting_the_scope_touches_neither_the_prose_nor_the_course() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "capacity?");
    let d = moves
        .add(Writing {
            course: Some(Course::Abandon),
            ..Writing::new(
                Kind::Decision,
                &a_name(),
                "we are not carrying on, nobody pays that latency",
                "me",
            )
        })
        .unwrap();

    moves
        .reword(d, None, Some(Scope::of(vec![q])), None, "me")
        .unwrap();

    let body = moves.all().unwrap().remove(&d).unwrap();
    assert_eq!(
        body.prose,
        "we are not carrying on, nobody pays that latency"
    );
    assert_eq!(body.course, Some(Course::Abandon));
    assert_eq!(body.scope, Scope::of(vec![q]));
}

#[test]
fn changing_the_course_touches_neither_the_prose_nor_the_scope() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "capacity?");
    tried(&moves, "x2", "aaa", &[q]);
    let d = moves
        .add(Writing {
            scope: Scope::of(vec![q]),
            course: Some(Course::Abandon),
            ..Writing::new(Kind::Decision, &a_name(), "we are not carrying on", "me")
        })
        .unwrap();

    moves
        .reword(d, None, None, Some(Course::Pursue), "me")
        .unwrap();

    assert_eq!(moves.decided().unwrap().get("aaa"), Some(&Course::Pursue));
    assert_eq!(moves.all().unwrap()[&d].scope, Scope::of(vec![q]));
}

#[test]
fn a_course_in_a_rewording_of_something_that_decides_nothing_is_refused() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let f = plain(&moves, Kind::Finding, "I saw that it does not");

    assert!(
        moves
            .reword(f, None, None, Some(Course::Pursue), "me")
            .is_err()
    );
}

#[test]
fn an_attempt_can_cite_a_trial_after_it_was_written() {
    // The trials run after the attempt is noted, so the evidence is added
    // afterwards. If it could only travel at creation, an attempt could never
    // point at what was run with it.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let a = tried(&moves, "x2", "aaa", &[]);

    moves
        .cite(
            a,
            Cited {
                what: "trial".into(),
                id: "exp/t/aaa/trial/3/0".into(),
            },
            "me",
        )
        .unwrap();

    let body = moves.all().unwrap().remove(&a).unwrap();
    assert_eq!(
        body.cites.len(),
        2,
        "the commit it already had, and the trial"
    );
    assert_eq!(body.cites[1].id, "exp/t/aaa/trial/3/0");
}

#[test]
fn citing_the_same_thing_twice_does_not_duplicate_it() {
    // Two people looking at one screen would ask for it, and a list with the
    // same trial twice says nothing a list with it once does not.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let a = tried(&moves, "x2", "aaa", &[]);
    let cited = Cited {
        what: "trial".into(),
        id: "exp/t/aaa/trial/3/0".into(),
    };

    moves.cite(a, cited.clone(), "me").unwrap();
    moves.cite(a, cited, "other").unwrap();

    assert_eq!(moves.all().unwrap()[&a].cites.len(), 2);
}

#[test]
fn a_question_cites_neither_commits_nor_trials() {
    // It is about moves, not layer-1 pieces. Letting it cite would let it
    // point at a commit with nobody knowing what that means.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "capacity?");

    assert!(
        moves
            .cite(
                q,
                Cited {
                    what: "commit".into(),
                    id: "aaa".into()
                },
                "me"
            )
            .is_err()
    );
}

#[test]
fn a_finding_does_cite_the_trial_it_was_seen_in() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let f = plain(&moves, Kind::Finding, "latency goes up");

    moves
        .cite(
            f,
            Cited {
                what: "trial".into(),
                id: "exp/t/aaa/trial/1/0".into(),
            },
            "me",
        )
        .unwrap();

    assert_eq!(moves.all().unwrap()[&f].cites.len(), 1);
}

#[test]
fn citing_touches_neither_the_prose_nor_the_scope_nor_the_course() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "capacity?");
    let a = tried(&moves, "x2, the good one", "aaa", &[q]);

    moves
        .cite(
            a,
            Cited {
                what: "artifact".into(),
                id: "report.pdf".into(),
            },
            "me",
        )
        .unwrap();

    let body = moves.all().unwrap().remove(&a).unwrap();
    assert_eq!(body.prose, "x2, the good one");
    assert_eq!(moves.under().unwrap().parents_of(a), vec![q]);
}

#[test]
fn a_move_is_reached_by_the_name_its_author_chose() {
    // The whole reason a name exists: the id works while somebody is holding
    // it, and this is the process that never saw it created.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = named(
        &moves,
        Kind::Question,
        "capacity-vs-interpretability",
        "does more capacity help?",
    );

    let again = Moves::of("t", &kept);

    assert_eq!(again.went("capacity-vs-interpretability").unwrap(), q);
}

#[test]
fn finding_a_name_costs_one_lookup_and_not_a_walk_of_every_move() {
    // Said as a test because it is the reason the name has a record of its
    // own: `go` and every `--under` ask this, and answering it by reading the
    // whole reasoning would make the cheapest question the most expensive one.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    for n in 0..20 {
        named(&moves, Kind::Question, &format!("q{n}"), "one of many");
    }

    let counted = crate::counting::Counting::over(kept);
    let moves = Moves::of("t", &counted);
    counted.forget();
    moves.went("q17").expect("the move");

    let (resolves, scans, fetches) = counted.seen();
    assert_eq!(scans, 0, "a name is resolved, never scanned for");
    assert_eq!(
        (resolves, fetches),
        (1, 1),
        "one lookup and the id behind it"
    );
}

#[test]
fn two_moves_cannot_answer_to_one_name() {
    // Not a nicety: a name is how a move is found again, so a second one
    // taking it is a move nobody can reach.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let first = named(&moves, Kind::Question, "the-same", "asked first");

    let again = moves.add(Writing::new(
        Kind::Hypothesis,
        "the-same",
        "asked second",
        "me",
    ));

    assert!(
        matches!(again, Err(Trouble::NameTaken { by, .. }) if by == first),
        "it says who has it: {again:?}",
    );
}

#[test]
fn the_move_refused_for_its_name_was_never_written() {
    // A refusal that half-wrote would leave a move with no name, which is a
    // move nobody can reach — the very thing the refusal is for.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    named(&moves, Kind::Question, "taken", "asked first");
    let before = moves.all().unwrap().len();

    let _ = moves.add(Writing::new(
        Kind::Hypothesis,
        "taken",
        "asked second",
        "me",
    ));

    assert_eq!(moves.all().unwrap().len(), before);
}

#[test]
fn a_name_nobody_took_says_so_rather_than_answering_about_another_move() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    named(&moves, Kind::Question, "one", "the only one");

    let went = moves.went("another");

    assert!(matches!(went, Err(Trouble::NoSuchName { .. })), "{went:?}",);
}

#[test]
fn one_tree_does_not_see_another_trees_names() {
    // Two investigations share a store on purpose, and `tree` is what keeps
    // them apart. A name is scoped by it like everything else.
    let (_at, kept) = somewhere();
    named(&Moves::of("one", &kept), Kind::Question, "shared", "mine");

    let other = Moves::of("another", &kept);

    assert!(matches!(
        other.went("shared"),
        Err(Trouble::NoSuchName { .. })
    ));
    assert!(
        other
            .add(Writing::new(Kind::Question, "shared", "also mine", "me"))
            .is_ok(),
        "the same word is free in another tree",
    );
}
