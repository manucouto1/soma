//! One configuration, which is also a trial's name.

use somatize_study::{Point, Setting};

fn point() -> Point {
    Point::of(vec![
        ("lr".into(), Setting::Real(0.001)),
        ("batch".into(), Setting::Int(32)),
        ("opt".into(), Setting::Choice("adam".into())),
    ])
}

#[test]
fn every_knob_is_there_and_in_the_space_order() {
    let point = point();

    assert_eq!(point.len(), 3);
    assert_eq!(point.get("lr"), Some(&Setting::Real(0.001)));
    assert_eq!(point.get("batch"), Some(&Setting::Int(32)));
    assert_eq!(point.get("nothing"), None);
}

#[test]
fn it_writes_itself_down_because_that_is_the_trial_name() {
    // Derived from the values in the space's order, so two machines that never
    // spoke file the same configuration under the same name.
    assert_eq!(point().to_string(), "lr=0.001,batch=32,opt=adam");
}

#[test]
fn two_configurations_that_differ_are_written_down_differently() {
    let other = Point::of(vec![
        ("lr".into(), Setting::Real(0.002)),
        ("batch".into(), Setting::Int(32)),
        ("opt".into(), Setting::Choice("adam".into())),
    ]);

    assert_ne!(point().to_string(), other.to_string());
}
