//! Typed values flowing between filters in a pipeline.
//!
//! [`Value`] variants: Tensor (f64 array with shape), Text, JSON, Bytes,
//! Object, Empty.
//! Values are serializable and content-addressable via [`crate::cache::CacheKey`].

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Typed values flowing between filters in a pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data")]
#[non_exhaustive]
pub enum Value {
    /// Numeric tensor data (shape + flat data).
    /// `values` is wrapped in [`Arc`] so that cloning a `Value` is O(1).
    Tensor {
        /// Flat data in row-major order; `Arc`-shared, never mutated in place.
        values: Arc<Vec<f64>>,
        /// Dimension sizes; the product must equal `values.len()`.
        shape: Vec<usize>,
    },

    /// UTF-8 text (Arc-wrapped for cheap cloning).
    ///
    /// Distinct from `Json(String)`: a prompt or completion is text, not a
    /// JSON document that happens to be a string. Keeping them apart means
    /// no round-trip through quoting/escaping on every hop, and lets a
    /// schema say "this edge carries text" — see [`crate::schema::DataType`].
    Text(Arc<str>),

    /// Structured JSON data (Arc-wrapped for cheap cloning).
    Json(Arc<serde_json::Value>),

    /// Raw bytes (Arc-wrapped for cheap cloning).
    Bytes(Arc<Vec<u8>>),

    /// Opaque serialized object (e.g. Python pickle).
    /// Soma passes it through without interpreting the contents.
    /// Used for efficient inter-filter data transfer when the producing
    /// and consuming runtimes share a serialization format.
    Object(Arc<Vec<u8>>),

    /// Empty / void value
    Empty,
}

impl Value {
    /// Create a tensor from flat row-major data and a shape.
    pub fn tensor(values: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::Tensor {
            values: Arc::new(values),
            shape,
        }
    }

    /// Create a text value.
    pub fn text(s: impl AsRef<str>) -> Self {
        Self::Text(Arc::from(s.as_ref()))
    }

    /// Create a JSON value.
    pub fn json(val: serde_json::Value) -> Self {
        Self::Json(Arc::new(val))
    }

    /// Create a raw bytes value.
    pub fn bytes(data: Vec<u8>) -> Self {
        Self::Bytes(Arc::new(data))
    }

    /// Create an opaque serialized object (e.g. a Python pickle).
    pub fn object(data: Vec<u8>) -> Self {
        Self::Object(Arc::new(data))
    }

    /// Is this the [`Value::Empty`] variant?
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Try to extract tensor data.
    pub fn as_tensor(&self) -> Option<(&[f64], &[usize])> {
        match self {
            Self::Tensor { values, shape } => Some((values, shape)),
            _ => None,
        }
    }

    /// Try to extract text.
    ///
    /// A `Json` string counts: a filter that returns `"hello"` as JSON and one
    /// that returns it as text should both satisfy a consumer wanting text.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            Self::Json(v) => v.as_str(),
            _ => None,
        }
    }

    /// Try to extract JSON value.
    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Json(v) => Some(v),
            _ => None,
        }
    }

    /// Natural JSON form for user-facing fan-in: tensors become
    /// (nested) number arrays, Json unwraps, Empty is null. This is what
    /// a multi-predecessor node receives per upstream branch — never the
    /// internal serde-tagged encoding.
    pub fn to_plain_json(&self) -> serde_json::Value {
        fn nest(values: &[f64], shape: &[usize]) -> serde_json::Value {
            if shape.len() <= 1 {
                return serde_json::Value::Array(
                    values.iter().map(|v| serde_json::json!(v)).collect(),
                );
            }
            let rows = shape[0];
            let row_len: usize = shape[1..].iter().product::<usize>().max(1);
            serde_json::Value::Array(
                (0..rows)
                    .map(|r| {
                        let start = r * row_len;
                        let end = (start + row_len).min(values.len());
                        nest(&values[start..end.max(start)], &shape[1..])
                    })
                    .collect(),
            )
        }
        match self {
            Self::Tensor { values, shape } => nest(values, shape),
            Self::Text(s) => serde_json::Value::String(s.to_string()),
            Self::Json(v) => (**v).clone(),
            Self::Empty => serde_json::Value::Null,
            other => serde_json::to_value(other).unwrap_or(serde_json::Value::Null),
        }
    }

    /// Short name of the variant, for error messages.
    ///
    /// Unlike `Display` this never renders the payload, so it is safe to put
    /// in an error that may reach a log — a JSON value can hold a prompt.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Tensor { .. } => "Tensor",
            Self::Text(_) => "Text",
            Self::Json(_) => "Json",
            Self::Bytes(_) => "Bytes",
            Self::Object(_) => "Object",
            Self::Empty => "Empty",
        }
    }

    /// Number of elements (for tensors) or bytes.
    pub fn size(&self) -> usize {
        match self {
            Self::Tensor { values, .. } => values.len(),
            Self::Text(s) => s.len(),
            Self::Json(v) => v.to_string().len(),
            Self::Bytes(b) | Self::Object(b) => b.len(),
            Self::Empty => 0,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor { shape, values } => {
                write!(f, "Tensor(shape={shape:?}, len={})", values.len())
            }
            Self::Text(s) => write!(f, "Text(len={})", s.len()),
            Self::Json(v) => write!(f, "Json({v})"),
            Self::Bytes(b) => write!(f, "Bytes(len={})", b.len()),
            Self::Object(b) => write!(f, "Object(len={})", b.len()),
            Self::Empty => write!(f, "Empty"),
        }
    }
}

impl From<Vec<f64>> for Value {
    fn from(values: Vec<f64>) -> Self {
        let len = values.len();
        Self::Tensor {
            values: Arc::new(values),
            shape: vec![len],
        }
    }
}

impl From<serde_json::Value> for Value {
    fn from(v: serde_json::Value) -> Self {
        Self::Json(Arc::new(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tensor_creation_and_access() {
        let v = Value::tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let (data, shape) = v.as_tensor().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, &[2, 2]);
    }

    #[test]
    fn json_value() {
        let v = Value::json(json!({"key": "value"}));
        let j = v.as_json().unwrap();
        assert_eq!(j["key"], "value");
    }

    #[test]
    fn empty_value() {
        let v = Value::Empty;
        assert!(v.is_empty());
        assert_eq!(v.size(), 0);
    }

    #[test]
    fn from_vec_f64() {
        let v: Value = vec![1.0, 2.0, 3.0].into();
        let (data, shape) = v.as_tensor().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0]);
        assert_eq!(shape, &[3]);
    }

    #[test]
    fn display_formatting() {
        let t = Value::tensor(vec![1.0, 2.0], vec![2]);
        assert_eq!(t.to_string(), "Tensor(shape=[2], len=2)");

        let e = Value::Empty;
        assert_eq!(e.to_string(), "Empty");
    }

    #[test]
    fn serde_roundtrip() {
        let values = vec![
            Value::tensor(vec![1.0, 2.0, 3.0], vec![3]),
            Value::json(json!({"a": 1})),
            Value::bytes(vec![0xDE, 0xAD]),
            Value::Empty,
        ];

        for v in values {
            let serialized = serde_json::to_string(&v).unwrap();
            let deserialized: Value = serde_json::from_str(&serialized).unwrap();
            assert_eq!(v, deserialized);
        }
    }

    #[test]
    fn size_returns_correct_values() {
        assert_eq!(Value::tensor(vec![1.0; 100], vec![10, 10]).size(), 100);
        assert_eq!(Value::bytes(vec![0; 50]).size(), 50);
        assert!(Value::json(json!({"key": "val"})).size() > 0);
    }
}
