//! Which rows, and where from.

use somatize_core::Value;
use std::fmt;

/// The rows a source is being asked for: `take` of them, starting at `at`.
///
/// This is what a graph reading from a source is handed as its **input**, and
/// the input is the one value a cache hashes by content. Two numbers, so naming
/// it is free; the rows are named by the source's version instead.
///
/// And it is what makes a stream cacheable. A span is a **position**, and a
/// position is repeatable: rows 400..500 are the same rows tomorrow, whatever
/// has arrived since. What moves is not the source's state, it is which spans
/// exist. A source answering *whatever is newest* is the other thing, and the
/// engine already refuses to cache under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    /// The first row, counting from zero.
    pub at: u64,
    /// How many, at most: the last span of a dataset is short and that is not an
    /// error.
    pub take: u64,
}

impl Span {
    /// `take` rows starting at `at`.
    pub fn new(at: u64, take: u64) -> Self {
        Self { at, take }
    }

    /// The span this value is, if it is one.
    ///
    /// A `Map` of two numbers and not a pair of positions in a list: what a
    /// record shows is what was asked for, and `{"at": 4096, "take": 64}` says
    /// it where `[4096, 64]` needs the reader to remember the order.
    pub fn of(value: &Value) -> Result<Self, SpanError> {
        let (Some(at), Some(take)) = (value.get("at"), value.get("take")) else {
            return Err(SpanError(format!(
                "a source is asked for rows, and what arrived was a {}. It takes \
                 `{{\"at\": <first row>, \"take\": <how many>}}`",
                value.type_name()
            )));
        };
        Ok(Self::new(whole(at, "at")?, whole(take, "take")?))
    }

    /// As a value, which is how it is handed to a graph.
    pub fn value(&self) -> Value {
        Value::map(vec![
            ("at".to_string(), Value::number(self.at as f64)),
            ("take".to_string(), Value::number(self.take as f64)),
        ])
    }
}

/// A number that is a count: whole, and not negative.
fn whole(value: &Value, field: &str) -> Result<u64, SpanError> {
    let Value::Number(x) = value else {
        return Err(SpanError(format!(
            "`{field}` is a number of rows, and this is a {}",
            value.type_name()
        )));
    };
    if *x < 0.0 || x.fract() != 0.0 {
        return Err(SpanError(format!(
            "`{field}` is a count of rows, and it is {x}"
        )));
    }
    Ok(*x as u64)
}

/// Why that was not a span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanError(String);

impl SpanError {
    /// The message.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SpanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SpanError {}
