//! The coordinate: what a graph is handed instead of the rows.

use soma_next_core::Value;
use soma_next_data::Span;

#[test]
fn a_span_comes_back_from_the_value_it_went_out_as() {
    let span = Span::new(4096, 64);

    assert_eq!(Span::of(&span.value()).unwrap(), span);
}

#[test]
fn it_says_what_it_is_asking_for_by_name() {
    // A map and not a pair of positions: a record shows what was asked for, and
    // `[4096, 64]` needs the reader to remember which is which.
    let value = Span::new(4096, 64).value();

    assert_eq!(value.get("at"), Some(&Value::number(4096.0)));
    assert_eq!(value.get("take"), Some(&Value::number(64.0)));
}

#[test]
fn asking_for_no_rows_is_a_question_and_not_a_mistake() {
    assert_eq!(
        Span::of(&Span::new(10, 0).value()).unwrap(),
        Span::new(10, 0)
    );
}

#[test]
fn something_that_is_not_a_span_says_so_with_what_it_takes() {
    let why = Span::of(&Value::number(4096.0)).unwrap_err();

    assert!(why.message().contains("\"at\""), "{why}");
    assert!(why.message().contains("\"take\""), "{why}");
}

#[test]
fn and_so_does_one_with_half_of_it_missing() {
    let half = Value::map(vec![("at".to_string(), Value::number(0.0))]);

    assert!(Span::of(&half).is_err());
}

#[test]
fn a_count_of_rows_is_whole_and_not_negative() {
    for wrong in [-1.0, 2.5] {
        let span = Value::map(vec![
            ("at".to_string(), Value::number(0.0)),
            ("take".to_string(), Value::number(wrong)),
        ]);

        let why = Span::of(&span).unwrap_err();
        assert!(why.message().contains("count of rows"), "{why}");
    }
}
