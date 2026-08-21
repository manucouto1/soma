//! The knobs and what each one may be.

use soma_next_study::{Dimension, Space, SpaceError};

fn real(low: f64, high: f64) -> Dimension {
    Dimension::Real {
        low,
        high,
        log: false,
    }
}

#[test]
fn the_knobs_come_out_in_the_order_they_were_declared() {
    // A grid enumerates in this order and a point writes itself down in it, so
    // two machines that never spoke have to agree on it.
    let space = Space::new()
        .with("lr", real(0.0, 1.0))
        .unwrap()
        .with("batch", Dimension::Int { low: 16, high: 128 })
        .unwrap();

    assert_eq!(space.len(), 2);
    assert_eq!(
        space
            .dimensions()
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["lr", "batch"]
    );
}

#[test]
fn two_knobs_by_the_same_name_are_refused() {
    let space = Space::new().with("lr", real(0.0, 1.0)).unwrap();

    assert_eq!(
        space.with("lr", real(0.0, 2.0)),
        Err(SpaceError::Taken("lr".into()))
    );
}

#[test]
fn a_knob_with_nothing_to_draw_from_is_refused_where_it_was_written() {
    // Not as a search that quietly only ever tried one value.
    let empty = |name, dimension| Space::new().with(name, dimension).unwrap_err();

    assert!(matches!(
        empty("lr", real(1.0, 0.0)),
        SpaceError::Empty(_, _)
    ));
    assert!(matches!(
        empty("opt", Dimension::Choice(vec![])),
        SpaceError::Empty(_, _)
    ));
    assert!(matches!(
        empty("n", Dimension::Int { low: 5, high: 5 }),
        SpaceError::Empty(_, _)
    ));
}

#[test]
fn a_logarithmic_range_has_to_start_above_zero() {
    // There is no logarithm of zero, and a learning rate of zero is not a
    // learning rate. Caught here rather than as a `-inf` inside a draw.
    assert!(matches!(
        Space::new()
            .with(
                "lr",
                Dimension::Real {
                    low: 0.0,
                    high: 0.1,
                    log: true
                }
            )
            .unwrap_err(),
        SpaceError::Empty(_, _)
    ));
}

#[test]
fn a_grid_takes_a_narrow_int_whole_and_cuts_what_is_continuous() {
    assert_eq!(Dimension::Int { low: 1, high: 5 }.grid_of(10), 5);
    assert_eq!(Dimension::Int { low: 1, high: 100 }.grid_of(4), 4);
    assert_eq!(real(0.0, 1.0).grid_of(4), 4);
    assert_eq!(
        Dimension::Choice(vec!["a".into(), "b".into()]).grid_of(9),
        2
    );
}

#[test]
fn it_writes_itself_down_and_says_which_ranges_are_logarithmic() {
    let space = Space::new()
        .with(
            "lr",
            Dimension::Real {
                low: 1e-5,
                high: 0.1,
                log: true,
            },
        )
        .unwrap()
        .with("opt", Dimension::Choice(vec!["adam".into(), "sgd".into()]))
        .unwrap();

    assert_eq!(
        space.to_string(),
        "lr=logreal(0.00001,0.1),opt=choice(adam|sgd)"
    );
}
