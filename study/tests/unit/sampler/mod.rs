//! The family: what the enum adds over calling a scheme directly.

mod grid;
mod random;
mod tpe;

use soma_next_study::{Dimension, Goal, Grid, Point, Random, Sampler, Setting, Space, Tpe};

fn space() -> Space {
    Space::new()
        .with(
            "lr",
            Dimension::Real {
                low: 0.001,
                high: 0.1,
                log: true,
            },
        )
        .unwrap()
        .with("opt", Dimension::Choice(vec!["adam".into(), "sgd".into()]))
        .unwrap()
}

fn history() -> Vec<(Point, f64)> {
    (0..6)
        .map(|i| {
            (
                Point::of(vec![
                    ("lr".into(), Setting::Real(0.001 * (i + 1) as f64)),
                    ("opt".into(), Setting::Choice("adam".into())),
                ]),
                i as f64,
            )
        })
        .collect()
}

fn one_of_each() -> Vec<Sampler> {
    vec![
        Grid { steps: 4 }.into(),
        Random { seed: 0 }.into(),
        Tpe {
            goal: Goal::Minimize,
            startup: 4,
            candidates: 24,
            quantile: 0.25,
            seed: 0,
        }
        .into(),
    ]
}

#[test]
fn going_through_the_enum_asks_exactly_the_same_as_not_going_through_it() {
    let space = space();
    let history = history();

    let how = Grid { steps: 4 };
    assert_eq!(
        Sampler::from(how.clone()).ask(&space, 3, &history),
        how.ask(&space, 3, &history)
    );

    let how = Random { seed: 9 };
    assert_eq!(
        Sampler::from(how.clone()).ask(&space, 3, &history),
        how.ask(&space, 3, &history)
    );

    let how = Tpe {
        goal: Goal::Minimize,
        startup: 4,
        candidates: 24,
        quantile: 0.25,
        seed: 9,
    };
    assert_eq!(
        Sampler::from(how.clone()).ask(&space, 3, &history),
        how.ask(&space, 3, &history)
    );
}

#[test]
fn only_the_grid_runs_out_and_it_does_so_through_the_enum_too() {
    let space = space();
    let grid: Sampler = Grid { steps: 4 }.into();

    assert!(grid.ask(&space, 7, &[]).is_some());
    assert_eq!(grid.ask(&space, 8, &[]), None);
    let endless = Sampler::from(Random { seed: 0 });
    assert!(endless.ask(&space, 10_000, &[]).is_some());
}

#[test]
fn two_of_the_three_ignore_what_finished_and_that_is_why_there_are_three() {
    let space = space();
    let history = history();

    for how in [
        Sampler::from(Grid { steps: 4 }),
        Sampler::from(Random { seed: 0 }),
    ] {
        assert_eq!(how.ask(&space, 2, &[]), how.ask(&space, 2, &history));
    }

    let guided = Sampler::from(Tpe {
        goal: Goal::Minimize,
        startup: 4,
        candidates: 24,
        quantile: 0.25,
        seed: 0,
    });
    assert_ne!(guided.ask(&space, 2, &[]), guided.ask(&space, 2, &history));
}

#[test]
fn nothing_to_search_is_nowhere_to_look() {
    for how in one_of_each() {
        assert_eq!(how.ask(&Space::new(), 0, &[]), None);
    }
}

#[test]
fn it_writes_itself_down_for_the_record_of_a_run() {
    assert_eq!(Sampler::from(Grid { steps: 4 }).to_string(), "grid:4");
    assert_eq!(Sampler::from(Random { seed: 7 }).to_string(), "random:7");
    assert_eq!(
        Sampler::from(Tpe {
            goal: Goal::Minimize,
            startup: 10,
            candidates: 24,
            quantile: 0.25,
            seed: 7,
        })
        .to_string(),
        "tpe:min:startup:10:candidates:24:quantile:0.25:seed:7"
    );
}

#[test]
fn two_samplers_that_differ_are_written_down_differently() {
    let mut all = one_of_each();
    all.push(Grid { steps: 5 }.into());
    all.push(Random { seed: 1 }.into());
    all.push(
        Tpe {
            goal: Goal::Maximize,
            startup: 4,
            candidates: 24,
            quantile: 0.25,
            seed: 0,
        }
        .into(),
    );

    let names: std::collections::BTreeSet<String> = all.iter().map(|s| s.to_string()).collect();
    assert_eq!(names.len(), all.len());
}
