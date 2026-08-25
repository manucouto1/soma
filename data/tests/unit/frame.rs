//! What a source answers with, and how it crosses an edge.

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use somatize_core::Value;
use somatize_data::Frame;
use std::sync::Arc;

/// Two columns and three rows, which is enough to have a shape.
pub fn batch() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("sms", DataType::Utf8, false),
        Field::new("label", DataType::Int64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(vec![
                "free entry",
                "see you at eight",
                "win cash",
            ])),
            Arc::new(Int64Array::from(vec![1, 0, 1])),
        ],
    )
    .unwrap()
}

#[test]
fn a_frame_crosses_an_edge_and_is_the_same_frame_on_the_other_side() {
    let value = Frame::new(batch()).value();

    let landed = Frame::of(&value).expect("a frame went in");
    assert_eq!(landed.rows(), 3);
    assert_eq!(landed.batch(), &batch());
}

#[test]
fn what_the_columns_are_called_is_there_without_reading_a_value() {
    let frame = Frame::new(batch());

    let names: Vec<&str> = frame
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(names, ["sms", "label"]);
}

#[test]
fn something_that_is_not_a_frame_is_not_one() {
    assert!(Frame::of(&Value::number(3.0)).is_none());
    assert!(Frame::of(&Value::opaque("a string is not a frame".to_string())).is_none());
}

#[test]
fn a_frame_does_not_travel_on_its_own() {
    // It is an opaque, so the frontier is the same one a tensor meets: what
    // crosses a wire or reaches a store does so through a codec, and until
    // there is one this says no rather than writing something wrong.
    assert!(!Frame::new(batch()).value().travels());
}
