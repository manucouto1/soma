//! Guided by what already worked.

use soma_next_study::{Dimension, Goal, Point, Random, Setting, Space, Tpe};

fn space() -> Space {
    Space::new()
        .with(
            "lr",
            Dimension::Real {
                low: 1e-5,
                high: 1e-1,
                log: true,
            },
        )
        .unwrap()
        .with("opt", Dimension::Choice(vec!["adam".into(), "sgd".into()]))
        .unwrap()
}

fn at(lr: f64, opt: &str) -> Point {
    Point::of(vec![
        ("lr".into(), Setting::Real(lr)),
        ("opt".into(), Setting::Choice(opt.into())),
    ])
}

fn lr_of(point: &Point) -> f64 {
    match point.get("lr") {
        Some(Setting::Real(value)) => *value,
        other => panic!("a real was expected, and {other:?} arrived"),
    }
}

/// Ten finished trials where the small learning rates did well and `adam` did
/// better than `sgd`, scored as a loss.
fn history() -> Vec<(Point, f64)> {
    vec![
        (at(1.1e-5, "adam"), 0.10),
        (at(2.0e-5, "adam"), 0.12),
        (at(3.0e-5, "adam"), 0.15),
        (at(9.0e-3, "sgd"), 5.00),
        (at(1.5e-2, "sgd"), 6.00),
        (at(3.0e-2, "adam"), 7.00),
        (at(5.0e-2, "sgd"), 8.00),
        (at(7.0e-2, "sgd"), 9.00),
        (at(8.0e-2, "adam"), 9.50),
        (at(9.5e-2, "sgd"), 9.90),
    ]
}

fn tpe(startup: usize) -> Tpe {
    Tpe {
        goal: Goal::Minimize,
        startup,
        candidates: 24,
        quantile: 0.3,
        seed: 11,
    }
}

#[test]
fn before_it_has_anything_to_learn_from_it_is_the_random_one() {
    // And exactly it, seed for seed: there is no second-best sampler hiding in
    // the startup phase.
    let space = space();

    assert_eq!(
        tpe(5).ask(&space, 2, &history()[..3]),
        Random { seed: 11 }.ask(&space, 2, &[])
    );
}

#[test]
fn two_finished_trials_is_the_floor_whatever_startup_says() {
    // Below two there is nothing to split into good and bad.
    let space = space();
    let one = &history()[..1];

    assert_eq!(
        tpe(0).ask(&space, 0, one),
        Random { seed: 11 }.ask(&space, 0, &[])
    );
}

#[test]
fn it_concentrates_where_the_good_trials_were() {
    // The point of the whole scheme. The best three sat below 1e-4; a uniform
    // draw would put a quarter of its proposals down there, and this one puts
    // most of them.
    let space = space();
    let history = history();

    let low = (0..60)
        .filter(|&trial| lr_of(&tpe(4).ask(&space, trial, &history).unwrap()) < 1e-4)
        .count();

    assert!(
        low > 40,
        "{low} of 60 below 1e-4, which is not concentrating"
    );
}

#[test]
fn it_prefers_the_option_the_good_trials_chose() {
    let space = space();
    let history = history();

    let adam = (0..60)
        .filter(|&trial| {
            tpe(4).ask(&space, trial, &history).unwrap().get("opt")
                == Some(&Setting::Choice("adam".into()))
        })
        .count();

    assert!(adam > 40, "{adam} of 60 chose adam");
}

#[test]
fn an_option_nobody_tried_stays_reachable() {
    // One imaginary observation of each keeps it unlikely rather than
    // impossible: a search that can never revisit a discarded option is a
    // search that cannot recover from three unlucky trials.
    let space = space();
    let history: Vec<(Point, f64)> = (0..8)
        .map(|i| (at(1e-4 * (i + 1) as f64, "adam"), i as f64))
        .collect();

    let sgd = (0..80)
        .filter(|&trial| {
            tpe(4).ask(&space, trial, &history).unwrap().get("opt")
                == Some(&Setting::Choice("sgd".into()))
        })
        .count();

    assert!(sgd > 0, "sgd became unreachable after nobody tried it");
}

#[test]
fn maximizing_looks_at_the_other_end_of_the_scores() {
    // Same history, read as an accuracy: now the big learning rates are the
    // good ones, and it goes there instead.
    let space = space();
    let history = history();
    let up = Tpe {
        goal: Goal::Maximize,
        ..tpe(4)
    };

    let high = (0..60)
        .filter(|&trial| lr_of(&up.ask(&space, trial, &history).unwrap()) > 1e-3)
        .count();

    assert!(high > 40, "{high} of 60 above 1e-3");
}

#[test]
fn a_trial_that_scored_nothing_comparable_is_dropped_and_not_counted_as_terrible() {
    let space = space();
    let clean = history();
    let mut history = clean.clone();
    history.push((at(2.0e-5, "adam"), f64::NAN));

    // The NaN sat where the good trials are; counted as a score it would drag
    // the split about. Dropped, the answer does not move at all.
    assert_eq!(
        tpe(4).ask(&space, 3, &history),
        tpe(4).ask(&space, 3, &clean),
    );
}

#[test]
fn the_same_seed_index_and_history_give_the_same_proposal() {
    let space = space();

    assert_eq!(
        tpe(4).ask(&space, 5, &history()),
        tpe(4).ask(&space, 5, &history())
    );
}

#[test]
fn everything_it_proposes_is_still_inside_the_knob() {
    // A bell drawn around an observation near the edge lands outside it; what
    // falls off is pulled back on rather than handed over.
    let space = space();
    let history = history();

    for trial in 0..100 {
        let point = tpe(4).ask(&space, trial, &history).unwrap();
        assert!((1e-5..=1e-1).contains(&lr_of(&point)), "{point}");
    }
}
