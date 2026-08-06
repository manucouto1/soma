//! Schema — dtype and shape for compile-time type checking between filters.
//!
//! The compiler validates that connected filters have compatible schemas
//! before execution begins, catching shape/type mismatches early.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Primitive data types that Soma values can contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DataType {
    /// 64-bit floating point.
    Float64,
    /// 32-bit floating point.
    Float32,
    /// 64-bit signed integer.
    Int64,
    /// Boolean.
    Bool,
    /// UTF-8 string.
    Utf8,
    /// Raw bytes.
    Bytes,
    /// Structured JSON (any shape).
    Json,
    /// A conversation: a list of [`crate::agentic::message::Message`].
    ///
    /// Distinct from `Json` so the compiler can reject an edge that hands a
    /// tensor, or an arbitrary document, to a node expecting a conversation.
    Messages,
}

impl DataType {
    /// Could a value of this type ever be read as `target`?
    ///
    /// This is coarser than equality on purpose. `is_compatible_with`
    /// demands an exact dtype match, which makes every mismatch equally
    /// suspicious — `f32` meeting `f64` and a conversation meeting a tensor
    /// both come out as "not compatible", so neither can be more than a
    /// warning without breaking the first case.
    ///
    /// This answers the stronger question: is there *any* reading under
    /// which this connection makes sense? Numeric widths differ but describe
    /// the same thing; a tensor and a conversation do not. The second kind
    /// is what fails a multi-agent handoff, and it is worth refusing to
    /// compile rather than warning about.
    ///
    /// The permitted coercions are exactly the ones the runtime performs:
    /// - anything → `Json` (every `Value` has a `to_plain_json`)
    /// - `Json` → anything (it is the dynamic type; the reader checks)
    /// - `Utf8` → `Messages` (a bare prompt becomes a user turn)
    /// - `Messages` → `Utf8` (a conversation's prose)
    /// - numeric ↔ numeric (widths differ; meaning does not)
    pub fn can_coerce_to(&self, target: &DataType) -> bool {
        use DataType::*;

        if self == target {
            return true;
        }
        // Json is the dynamic type: it absorbs and yields anything.
        if matches!(self, Json) || matches!(target, Json) {
            return true;
        }
        if self.is_numeric() && target.is_numeric() {
            return true;
        }
        matches!((self, target), (Utf8, Messages) | (Messages, Utf8))
    }

    /// Numeric in the sense [`Self::can_coerce_to`] uses: `Float64`,
    /// `Float32`, `Int64` — and `Bool`, which tensors carry as 0/1.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::Float64 | Self::Float32 | Self::Int64 | Self::Bool
        )
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float64 => write!(f, "f64"),
            Self::Float32 => write!(f, "f32"),
            Self::Int64 => write!(f, "i64"),
            Self::Bool => write!(f, "bool"),
            Self::Utf8 => write!(f, "str"),
            Self::Bytes => write!(f, "bytes"),
            Self::Json => write!(f, "json"),
            Self::Messages => write!(f, "messages"),
        }
    }
}

/// Describes the shape and type of a Value, without holding the actual data.
///
/// Used by:
/// - Filters: declare what they accept (input) and produce (output)
/// - Compiler: validate type compatibility between connected filters
/// - VirtualValue: know schema without materializing
/// - Cache metadata: describe stored entries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schema {
    /// The primitive data type.
    pub dtype: DataType,

    /// Shape dimensions. Empty for scalars, `[n]` for vectors, `[r,c]` for matrices, etc.
    /// `None` means shape is dynamic/unknown.
    pub shape: Option<Vec<Dimension>>,
}

/// A single dimension in a tensor shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dimension {
    /// Fixed size (e.g., 128 features).
    Fixed(usize),
    /// Dynamic size (e.g., batch dimension). Named for documentation.
    Dynamic(String),
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed(n) => write!(f, "{n}"),
            Self::Dynamic(name) => write!(f, "{name}"),
        }
    }
}

impl Schema {
    /// Create a schema for a 1D tensor (vector) of known length.
    pub fn vector(dtype: DataType, len: usize) -> Self {
        Self {
            dtype,
            shape: Some(vec![Dimension::Fixed(len)]),
        }
    }

    /// Create a schema for a 2D tensor (matrix) with known dimensions.
    pub fn matrix(dtype: DataType, rows: usize, cols: usize) -> Self {
        Self {
            dtype,
            shape: Some(vec![Dimension::Fixed(rows), Dimension::Fixed(cols)]),
        }
    }

    /// Create a schema for a tensor with a dynamic batch dimension.
    pub fn batched(dtype: DataType, feature_dims: &[usize]) -> Self {
        let mut dims = vec![Dimension::Dynamic("batch".into())];
        dims.extend(feature_dims.iter().map(|&d| Dimension::Fixed(d)));
        Self {
            dtype,
            shape: Some(dims),
        }
    }

    /// Create a schema for a scalar value.
    pub fn scalar(dtype: DataType) -> Self {
        Self {
            dtype,
            shape: Some(vec![]),
        }
    }

    /// Create a schema for JSON data (shape is irrelevant).
    pub fn json() -> Self {
        Self {
            dtype: DataType::Json,
            shape: None,
        }
    }

    /// Create a schema for UTF-8 text (shape is irrelevant).
    ///
    /// This is what a prompt or a completion carries. An edge typed `text`
    /// will not accept a tensor, which is how a mis-wired handoff between two
    /// agent nodes becomes a compile error rather than a runtime surprise.
    pub fn text() -> Self {
        Self {
            dtype: DataType::Utf8,
            shape: None,
        }
    }

    /// Create a schema for a conversation.
    pub fn messages() -> Self {
        Self {
            dtype: DataType::Messages,
            shape: None,
        }
    }

    /// Create a schema for raw bytes.
    pub fn bytes() -> Self {
        Self {
            dtype: DataType::Bytes,
            shape: None,
        }
    }

    /// Create a schema with fully dynamic (unknown) shape.
    pub fn dynamic(dtype: DataType) -> Self {
        Self { dtype, shape: None }
    }

    /// Is connecting these two definitely a mistake?
    ///
    /// True when no reading of `self` could satisfy `other` — a tensor
    /// arriving where a conversation is expected, say. The compiler refuses
    /// to build such a graph, rather than warning and letting it fail
    /// mid-run once tokens have been spent.
    pub fn is_incompatible_with(&self, other: &Schema) -> bool {
        !self.dtype.can_coerce_to(&other.dtype)
    }

    /// Check if this schema is compatible with another (can be connected in a pipeline).
    ///
    /// Compatibility rules:
    /// - Same dtype required (no implicit coercion)
    /// - If both shapes are known, fixed dimensions must match
    /// - Dynamic dimensions are compatible with any size
    /// - Unknown shape (None) is compatible with anything of the same dtype
    pub fn is_compatible_with(&self, other: &Schema) -> bool {
        if self.dtype != other.dtype {
            return false;
        }

        match (&self.shape, &other.shape) {
            (None, _) | (_, None) => true, // unknown shape is flexible
            (Some(a), Some(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                a.iter().zip(b.iter()).all(|(da, db)| match (da, db) {
                    (Dimension::Fixed(x), Dimension::Fixed(y)) => x == y,
                    _ => true, // dynamic is compatible with anything
                })
            }
        }
    }

    /// Number of known dimensions (rank).
    pub fn rank(&self) -> Option<usize> {
        self.shape.as_ref().map(|s| s.len())
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.dtype)?;
        if let Some(shape) = &self.shape {
            if shape.is_empty() {
                write!(f, " (scalar)")?;
            } else {
                let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                write!(f, "[{}]", dims.join(", "))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_display() {
        assert_eq!(
            Schema::scalar(DataType::Float64).to_string(),
            "f64 (scalar)"
        );
        assert_eq!(
            Schema::vector(DataType::Float64, 128).to_string(),
            "f64[128]"
        );
        assert_eq!(
            Schema::matrix(DataType::Float64, 100, 50).to_string(),
            "f64[100, 50]"
        );
        assert_eq!(
            Schema::batched(DataType::Float32, &[128]).to_string(),
            "f32[batch, 128]"
        );
        assert_eq!(Schema::json().to_string(), "json");
    }

    #[test]
    fn compatible_same_schema() {
        let s = Schema::vector(DataType::Float64, 128);
        assert!(s.is_compatible_with(&s));
    }

    #[test]
    fn compatible_dynamic_with_fixed() {
        let dynamic = Schema::batched(DataType::Float64, &[128]);
        let fixed = Schema::matrix(DataType::Float64, 32, 128);
        assert!(dynamic.is_compatible_with(&fixed));
        assert!(fixed.is_compatible_with(&dynamic));
    }

    #[test]
    fn compatible_unknown_shape() {
        let unknown = Schema::dynamic(DataType::Float64);
        let known = Schema::vector(DataType::Float64, 128);
        assert!(unknown.is_compatible_with(&known));
        assert!(known.is_compatible_with(&unknown));
    }

    #[test]
    fn incompatible_different_dtype() {
        let f64_schema = Schema::vector(DataType::Float64, 128);
        let i64_schema = Schema::vector(DataType::Int64, 128);
        assert!(!f64_schema.is_compatible_with(&i64_schema));
    }

    #[test]
    fn incompatible_different_fixed_dims() {
        let a = Schema::vector(DataType::Float64, 128);
        let b = Schema::vector(DataType::Float64, 256);
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn incompatible_different_rank() {
        let vec = Schema::vector(DataType::Float64, 128);
        let mat = Schema::matrix(DataType::Float64, 128, 64);
        assert!(!vec.is_compatible_with(&mat));
    }

    #[test]
    fn json_compatible_with_json() {
        assert!(Schema::json().is_compatible_with(&Schema::json()));
    }

    #[test]
    fn json_incompatible_with_tensor() {
        assert!(!Schema::json().is_compatible_with(&Schema::vector(DataType::Float64, 10)));
    }

    #[test]
    fn serde_roundtrip() {
        let schemas = vec![
            Schema::scalar(DataType::Float64),
            Schema::vector(DataType::Float32, 100),
            Schema::batched(DataType::Float64, &[128, 64]),
            Schema::json(),
            Schema::dynamic(DataType::Int64),
        ];
        for s in schemas {
            let json = serde_json::to_string(&s).unwrap();
            let deserialized: Schema = serde_json::from_str(&json).unwrap();
            assert_eq!(s, deserialized);
        }
    }

    #[test]
    fn rank() {
        assert_eq!(Schema::scalar(DataType::Float64).rank(), Some(0));
        assert_eq!(Schema::vector(DataType::Float64, 10).rank(), Some(1));
        assert_eq!(Schema::matrix(DataType::Float64, 10, 5).rank(), Some(2));
        assert_eq!(Schema::json().rank(), None);
    }
}
