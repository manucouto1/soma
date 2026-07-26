//! Typed values flowing between filters in a pipeline.
//!
//! [`Value`] variants: Tensor (f64 array with shape), JSON, Bytes, Empty.
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
        values: Arc<Vec<f64>>,
        shape: Vec<usize>,
    },

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
    pub fn tensor(values: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::Tensor {
            values: Arc::new(values),
            shape,
        }
    }

    pub fn json(val: serde_json::Value) -> Self {
        Self::Json(Arc::new(val))
    }

    pub fn bytes(data: Vec<u8>) -> Self {
        Self::Bytes(Arc::new(data))
    }

    pub fn object(data: Vec<u8>) -> Self {
        Self::Object(Arc::new(data))
    }

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

    /// Try to extract JSON value.
    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Json(v) => Some(v),
            _ => None,
        }
    }

    /// Number of elements (for tensors) or bytes.
    pub fn size(&self) -> usize {
        match self {
            Self::Tensor { values, .. } => values.len(),
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
