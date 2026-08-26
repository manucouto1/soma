//! The knobs and what each one may be.

use somatize_study::{Dimension, Goal, Grid, ReadError, Sampler, Space, SpaceError, Tpe};

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

fn searched() -> Space {
    Space::new()
        .with("lr", real(1e-5, 1e-1))
        .unwrap()
        .with("batch", Dimension::Int { low: 16, high: 128 })
        .unwrap()
        .with("opt", Dimension::Choice(vec!["adam".into(), "sgd".into()]))
        .unwrap()
}

#[test]
fn every_point_the_space_can_produce_survives_being_written_down() {
    // The property the whole record rests on: a trial keeps its configuration as
    // text next to its score, and the history comes back in one scan of the
    // folder rather than one fetch per trial. If the round trip is not exact,
    // that scan quietly reads points nobody ever tried.
    let space = searched();
    let how = Sampler::from(Grid { steps: 5 });

    for trial in 0.. {
        let Some(point) = how.ask(&space, trial, &[]) else {
            break;
        };
        assert_eq!(space.read(&point.to_string()), Ok(point));
    }
}

#[test]
fn the_space_is_what_says_whether_sixty_four_is_a_number_or_a_word() {
    // Why this is a method of `Space` and not of `Point`: the text on its own is
    // ambiguous, and nothing in it can settle the ambiguity.
    let counted = Space::new()
        .with("batch", Dimension::Int { low: 16, high: 128 })
        .unwrap();
    let named = Space::new()
        .with("batch", Dimension::Choice(vec!["64".into(), "128".into()]))
        .unwrap();

    assert_eq!(
        counted.read("batch=64").unwrap().get("batch"),
        Some(&somatize_study::Setting::Int(64))
    );
    assert_eq!(
        named.read("batch=64").unwrap().get("batch"),
        Some(&somatize_study::Setting::Choice("64".into()))
    );
}

#[test]
fn a_record_written_against_another_space_is_refused_and_not_half_read() {
    let space = searched();

    assert_eq!(
        space.read("lr=0.001,opt=adam"),
        Err(ReadError::Missing("batch".into()))
    );
    assert_eq!(
        space.read("lr=0.001,batch=32,opt=adam,dropout=0.5"),
        Err(ReadError::Stranger("dropout".into()))
    );
    assert_eq!(
        space.read("lr=0.001,batch,opt=adam"),
        Err(ReadError::Shapeless("batch".into()))
    );
}

#[test]
fn a_value_the_knob_could_never_take_is_not_one_of_its_values() {
    let space = searched();

    // The wrong kind, an option nobody declared, and outside the range.
    assert!(matches!(
        space.read("lr=fast,batch=32,opt=adam"),
        Err(ReadError::NotIn(name, _, _)) if name == "lr"
    ));
    assert!(matches!(
        space.read("lr=0.001,batch=32,opt=lion"),
        Err(ReadError::NotIn(name, _, _)) if name == "opt"
    ));
    assert!(matches!(
        space.read("lr=0.001,batch=9000,opt=adam"),
        Err(ReadError::NotIn(name, _, _)) if name == "batch"
    ));
}

#[test]
fn what_could_not_be_read_back_is_refused_where_it_was_typed() {
    // Not when the record is read — by then which knob was meant is gone. A
    // point is `name=value,name=value`, so those two characters cannot appear
    // inside either half.
    let commad = Space::new().with("a,b", real(0.0, 1.0));
    assert!(matches!(commad, Err(SpaceError::Unreadable(_, _))));

    let equalled = Space::new().with("lr=x", real(0.0, 1.0));
    assert!(matches!(equalled, Err(SpaceError::Unreadable(_, _))));

    let inside = Space::new().with("opt", Dimension::Choice(vec!["adam".into(), "a,b".into()]));
    assert!(matches!(inside, Err(SpaceError::Unreadable(name, text))
        if name == "opt" && text == "a,b"));
}

#[test]
fn it_is_what_hands_a_guided_sampler_a_history_it_never_saw_made() {
    // What reading is for. The scan of a shared folder gives text and scores; a
    // machine that ran none of those trials rebuilds `finished` from them and
    // asks where to look next.
    let space = searched();
    let scanned = [
        ("lr=0.01,batch=32,opt=adam", 0.9),
        ("lr=0.02,batch=64,opt=adam", 0.8),
        ("lr=0.09,batch=16,opt=sgd", 0.4),
        ("lr=0.08,batch=128,opt=sgd", 0.3),
    ];

    let finished: Vec<_> = scanned
        .iter()
        .map(|(said, score)| (space.read(said).unwrap(), Some(*score)))
        .collect();
    let guided = Tpe {
        goal: Goal::Maximize,
        startup: 2,
        candidates: 24,
        quantile: 0.5,
        seed: 3,
    };

    assert!(guided.ask(&space, 4, &finished).is_some());
    // And it is guided by them: with nothing scanned it proposes elsewhere.
    assert_ne!(guided.ask(&space, 4, &finished), guided.ask(&space, 4, &[]));
}
