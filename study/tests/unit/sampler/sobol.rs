//! Spread on purpose without a seam — and a table that has to be right.

use soma_next_study::{Dimension, KNOBS, Point, Random, Setting, Sobol, Space};
use std::collections::HashSet;

/// `knobs` knobs on the unit interval, so a value *is* its place in the space.
fn flat(knobs: usize) -> Space {
    (0..knobs).fold(Space::new(), |space, which| {
        space
            .with(
                format!("k{which}"),
                Dimension::Real {
                    low: 0.0,
                    high: 1.0,
                    log: false,
                },
            )
            .unwrap()
    })
}

fn unit(point: &Point, which: usize) -> f64 {
    match point.get(&format!("k{which}")) {
        Some(Setting::Real(value)) => *value,
        other => panic!("a real was expected, and {other:?} arrived"),
    }
}

fn cell(value: f64, of: usize) -> usize {
    ((value * of as f64) as usize).min(of - 1)
}

#[test]
fn the_same_seed_and_index_give_the_same_point_however_it_is_asked_for() {
    let how = Sobol { seed: 42 };
    let space = flat(4);

    let straight: Vec<_> = (0..8).map(|t| how.ask(&space, t, &[])).collect();
    assert_eq!(how.ask(&space, 7, &[]), straight[7]);
    assert_eq!(how.ask(&space, 0, &[]), straight[0]);
}

#[test]
fn of_the_first_two_to_the_k_trials_exactly_one_lands_in_each_cell_of_every_knob() {
    // The defining property, held by **every** knob the table reaches — which is
    // how the table is checked. Direction numbers that are subtly wrong do not
    // make a Sobol sequence fail; they make it cover worse, and nobody notices.
    // They cannot survive this: the equality is exact and it is per knob.
    let how = Sobol { seed: 7 };
    let space = flat(KNOBS);

    for cells in [16usize, 256] {
        for knob in 0..KNOBS {
            let mut taken = vec![0usize; cells];
            for trial in 0..cells {
                taken[cell(unit(&how.ask(&space, trial, &[]).unwrap(), knob), cells)] += 1;
            }
            assert!(
                taken.iter().all(|&once| once == 1),
                "knob {knob}: {taken:?}"
            );
        }
    }
}

#[test]
fn the_first_two_knobs_fill_every_rectangle_and_not_just_the_square() {
    // Sixty-four trials land one per cell of an 8x8 grid — and of a 2x32, and of
    // a 64x1, and of every split in between. That is what a table of direction
    // numbers is *for*: any one sequence covers its own line, and the work is
    // making two of them cover a plane together.
    //
    // Two knobs and not any two: further out the guarantee weakens to two points
    // per cell, and asserting otherwise would be asserting something false.
    let space = flat(8);

    for seed in 0..4 {
        let points: Vec<Point> = (0..64)
            .map(|trial| Sobol { seed }.ask(&space, trial, &[]).unwrap())
            .collect();
        for split in 0..=6u32 {
            let (across, down) = (1usize << split, 1usize << (6 - split));
            let filled: HashSet<(usize, usize)> = points
                .iter()
                .map(|p| (cell(unit(p, 0), across), cell(unit(p, 1), down)))
                .collect();
            assert_eq!(filled.len(), 64, "seed {seed}, {across}x{down}");
        }
    }
}

#[test]
fn where_a_uniform_draw_leaves_holes_it_leaves_none() {
    // The same comparison the spread schemes are here to make, and this one gets
    // to state it as an equality rather than as "better".
    let space = flat(2);
    let filled = |point: fn(u64, usize, &Space) -> Point| {
        let seen: HashSet<(usize, usize)> = (0..64)
            .map(|trial| {
                let p = point(4, trial, &space);
                (cell(unit(&p, 0), 8), cell(unit(&p, 1), 8))
            })
            .collect();
        seen.len()
    };

    assert_eq!(
        filled(|seed, trial, space| Sobol { seed }.ask(space, trial, &[]).unwrap()),
        64
    );
    assert!(filled(|seed, trial, space| Random { seed }.ask(space, trial, &[]).unwrap()) < 64);
}

#[test]
fn more_knobs_than_the_table_reaches_is_answered_with_nothing_from_the_first_trial() {
    // Its one ceiling, and it is loud: you get nothing at all, at once, rather
    // than a search that quietly stops covering somewhere in the middle.
    // `Halton` has no such ceiling, which is the other half of why there are two.
    let space = flat(KNOBS + 1);
    let how = Sobol { seed: 0 };

    assert_eq!(how.ask(&space, 0, &[]), None);
    assert_eq!(how.ask(&space, 9, &[]), None);
    assert!(how.ask(&flat(KNOBS), 0, &[]).is_some());
}

#[test]
fn a_different_seed_searches_somewhere_else() {
    let space = flat(3);

    assert_ne!(
        Sobol { seed: 1 }.ask(&space, 0, &[]),
        Sobol { seed: 2 }.ask(&space, 0, &[])
    );
}

#[test]
fn everything_it_draws_is_inside_the_knob() {
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
    let how = Sobol { seed: 7 };

    for trial in 0..500 {
        let point = how.ask(&space, trial, &[]).unwrap();
        match point.get("lr") {
            Some(Setting::Real(value)) => assert!((1e-5..=1e-1).contains(value), "{point}"),
            other => panic!("a real was expected, and {other:?} arrived"),
        }
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
    let how = Sobol { seed: 1 };
    let space = flat(3);
    let finished = vec![(how.ask(&space, 0, &[]).unwrap(), 0.5)];

    assert!(how.ask(&space, 100_000, &[]).is_some());
    assert_eq!(how.ask(&space, 5, &[]), how.ask(&space, 5, &finished));
    assert_eq!(how.ask(&Space::new(), 0, &[]), None);
}
