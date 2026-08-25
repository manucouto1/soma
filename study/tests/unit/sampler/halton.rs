//! Spread on purpose: what it promises that a uniform draw only promises on
//! average.

use somatize_study::{Dimension, Halton, Point, Random, Setting, Space};
use std::collections::HashSet;

/// Three knobs on the unit interval, so that a value *is* its place in the
/// space and a cell is arithmetic rather than a conversion.
fn flat() -> Space {
    ["a", "b", "c"]
        .into_iter()
        .fold(Space::new(), |space, name| {
            space
                .with(
                    name,
                    Dimension::Real {
                        low: 0.0,
                        high: 1.0,
                        log: false,
                    },
                )
                .unwrap()
        })
}

fn unit(point: &Point, name: &str) -> f64 {
    match point.get(name) {
        Some(Setting::Real(value)) => *value,
        other => panic!("a real was expected, and {other:?} arrived"),
    }
}

fn cell(value: f64, of: usize) -> usize {
    ((value * of as f64) as usize).min(of - 1)
}

#[test]
fn the_same_seed_and_index_give_the_same_point_however_it_is_asked_for() {
    // The property the whole crate is built on, and the reason this is a
    // sampler and not a generator: a machine that claimed trial 7 out of a
    // shared folder derives it without replaying the first six.
    let how = Halton { seed: 42 };
    let space = flat();

    let straight: Vec<_> = (0..8).map(|t| how.ask(&space, t, &[])).collect();
    assert_eq!(how.ask(&space, 7, &[]), straight[7]);
    assert_eq!(how.ask(&space, 0, &[]), straight[0]);
}

#[test]
fn of_the_first_base_squared_trials_exactly_one_lands_in_each_cell() {
    // The promise, and it is an equality and not a tendency. Knob `d` is read in
    // base the `d`-th prime, so each one has its own count of trials at which it
    // comes out exact.
    //
    // This is also what catches the scramble going wrong: permuting only the
    // digits an index actually has puts a one-digit index and a two-digit one on
    // two different grids, and then this is off by a handful of cells while
    // everything else still looks fine.
    let how = Halton { seed: 11 };
    let space = flat();

    for (knob, base) in [("a", 2usize), ("b", 3), ("c", 5)] {
        let cells = base * base;
        let mut taken = vec![0usize; cells];
        for trial in 0..cells {
            taken[cell(unit(&how.ask(&space, trial, &[]).unwrap(), knob), cells)] += 1;
        }
        assert!(taken.iter().all(|&once| once == 1), "{knob}: {taken:?}");
    }
}

#[test]
fn trial_zero_is_a_point_and_not_the_corner_of_the_space() {
    // Unscrambled, index zero has no digits and so is the bottom of every knob
    // at once — the smallest learning rate with the smallest batch, which is not
    // a first trial, it is a limiting case in disguise. The offsets settle it.
    let space = flat();

    for seed in 0..8 {
        let first = Halton { seed }.ask(&space, 0, &[]).unwrap();
        assert!(
            ["a", "b", "c"].iter().any(|knob| unit(&first, knob) > 0.05),
            "seed {seed} starts in the corner: {first}"
        );
    }
}

#[test]
fn where_a_uniform_draw_leaves_holes_it_does_not() {
    // What it is for, said as a comparison: over the same budget and the same
    // seed, a draw clumps and leaves cells nobody ever tried. This is the shape
    // of "two machines that never spoke do not land on the same configuration".
    let space = flat();
    let filled = |sixty_four: fn(u64, usize, &Space) -> Point| {
        let seen: HashSet<(usize, usize)> = (0..64)
            .map(|trial| {
                let point = sixty_four(4, trial, &space);
                (cell(unit(&point, "a"), 8), cell(unit(&point, "b"), 8))
            })
            .collect();
        seen.len()
    };

    let spread = filled(|seed, trial, space| Halton { seed }.ask(space, trial, &[]).unwrap());
    let drawn = filled(|seed, trial, space| Random { seed }.ask(space, trial, &[]).unwrap());

    assert!(spread > drawn, "spread {spread}, drawn {drawn} of 64");
    assert!(spread >= 48, "only {spread} of 64 cells");
}

#[test]
fn a_different_seed_searches_somewhere_else() {
    let space = flat();

    assert_ne!(
        Halton { seed: 1 }.ask(&space, 0, &[]),
        Halton { seed: 2 }.ask(&space, 0, &[])
    );
}

#[test]
fn everything_it_draws_is_inside_the_knob() {
    // A radical inverse is a sum of digits over powers, so the way it fails is
    // by reaching one exactly — and a `Choice` indexed by one is out of bounds.
    let space = Space::new()
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
        .unwrap();
    let how = Halton { seed: 7 };

    for trial in 0..500 {
        let point = how.ask(&space, trial, &[]).unwrap();
        assert!((1e-5..=1e-1).contains(&unit(&point, "lr")), "{point}");
        match point.get("batch") {
            Some(Setting::Int(value)) => assert!((16..=128).contains(value), "{point}"),
            other => panic!("an int was expected, and {other:?} arrived"),
        }
        match point.get("opt") {
            Some(Setting::Choice(option)) => {
                assert!(option == "adam" || option == "sgd", "{point}")
            }
            other => panic!("a choice was expected, and {other:?} arrived"),
        }
    }
}

#[test]
fn it_never_runs_out_and_never_looks_at_the_finished_trials() {
    let how = Halton { seed: 1 };
    let space = flat();
    let finished = vec![(how.ask(&space, 0, &[]).unwrap(), Some(0.5))];

    assert!(how.ask(&space, 100_000, &[]).is_some());
    assert_eq!(how.ask(&space, 5, &[]), how.ask(&space, 5, &finished));
    assert_eq!(how.ask(&Space::new(), 0, &[]), None);
}
