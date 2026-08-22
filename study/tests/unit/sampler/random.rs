//! Uniform, looking at nothing — and reproducible from the index alone.

use soma_next_study::{Dimension, Point, Random, Setting, Space};

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
        .with("batch", Dimension::Int { low: 16, high: 128 })
        .unwrap()
        .with("opt", Dimension::Choice(vec!["adam".into(), "sgd".into()]))
        .unwrap()
}

fn lr(point: &Point) -> f64 {
    match point.get("lr") {
        Some(Setting::Real(value)) => *value,
        other => panic!("a real was expected, and {other:?} arrived"),
    }
}

#[test]
fn the_same_seed_and_index_give_the_same_point_however_it_is_asked_for() {
    // The property that lets a machine which claimed trial 7 out of a shared
    // folder derive it without replaying the first six — no coordinator, the
    // same as CU15's federated round.
    let how = Random { seed: 42 };
    let space = space();

    assert_eq!(how.ask(&space, 7, &[]), how.ask(&space, 7, &[]));
    // Asked for out of order, and after everything else: still the same point.
    let straight: Vec<_> = (0..8).map(|t| how.ask(&space, t, &[])).collect();
    assert_eq!(how.ask(&space, 7, &[]), straight[7]);
}

#[test]
fn a_different_seed_searches_somewhere_else() {
    let space = space();

    assert_ne!(
        Random { seed: 1 }.ask(&space, 0, &[]),
        Random { seed: 2 }.ask(&space, 0, &[])
    );
}

#[test]
fn everything_it_draws_is_inside_the_knob() {
    let how = Random { seed: 7 };
    let space = space();

    for trial in 0..200 {
        let point = how.ask(&space, trial, &[]).unwrap();
        assert!((1e-5..=1e-1).contains(&lr(&point)), "{point}");
        match point.get("batch") {
            Some(Setting::Int(value)) => assert!((16..=128).contains(value)),
            other => panic!("an int was expected, and {other:?} arrived"),
        }
        match point.get("opt") {
            Some(Setting::Choice(option)) => assert!(option == "adam" || option == "sgd"),
            other => panic!("a choice was expected, and {other:?} arrived"),
        }
    }
}

#[test]
fn a_logarithmic_knob_spreads_over_the_decades_and_not_over_the_line() {
    // The whole reason `log` exists: drawn linearly, four fifths of `1e-5..1e-1`
    // sits above 0.02 and a search never sees a small learning rate at all.
    let how = Random { seed: 3 };
    let space = space();
    let middle = (1e-5f64 * 1e-1).sqrt(); // 1e-3, the geometric midpoint

    let below = (0..400)
        .filter(|&t| lr(&how.ask(&space, t, &[]).unwrap()) < middle)
        .count();

    assert!((150..=250).contains(&below), "{below} of 400 below 1e-3");
}

#[test]
fn it_never_runs_out_and_never_looks_at_the_finished_trials() {
    let how = Random { seed: 1 };
    let space = space();
    let finished = vec![(how.ask(&space, 0, &[]).unwrap(), Some(0.5))];

    assert!(how.ask(&space, 10_000, &[]).is_some());
    assert_eq!(how.ask(&space, 5, &[]), how.ask(&space, 5, &finished));
    assert_eq!(how.ask(&Space::new(), 0, &[]), None);
}
