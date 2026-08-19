//! What travels along an edge.
//!
//! The price of the core doing the executing: if the engine is in Rust, the data
//! has to have a shape Rust understands. Five variants are data the core
//! understands and can compare — no `Json`, which would pull in `serde_json`,
//! and no shaped `Tensor`, which nobody produces.
//!
//! The sixth, [`Value::Opaque`], is of another nature: it carries something the
//! core **does not look at**. It exists because some values cannot be converted
//! without being destroyed — a torch tensor mid-autograd-graph round-tripped
//! through numbers comes back without its `grad_fn`.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// A datum crossing from one node to the next.
///
/// `Arc` wherever the data is heavy, because a value is cloned on every edge
/// and cloning must not copy.
#[derive(Clone)]
pub enum Value {
    /// Nothing. What a root node receives when you pass it no input.
    Null,
    /// A number.
    Number(f64),
    /// UTF-8 text.
    Text(Arc<str>),
    /// Uninterpreted bytes.
    Bytes(Arc<Vec<u8>>),
    /// Several values in order.
    List(Arc<Vec<Value>>),
    /// Something the core carries without looking at it, as `dyn Any` so the
    /// core need not depend on PyO3.
    ///
    /// It **only exists in this process and in this run**, and everything else
    /// follows: no content hash, no serialization, and no comparison but
    /// identity.
    Opaque(Arc<dyn Any + Send + Sync>),
    /// Several named values: what a node with several incoming edges receives,
    /// keyed by the node that produced each one.
    ///
    /// **Ordered**, in the edges' declaration order: a `HashMap` iterates
    /// differently in each process, so a content hash would be useless.
    Map(Arc<Vec<(String, Value)>>),
}

impl Value {
    /// A number.
    pub fn number(x: f64) -> Self {
        Self::Number(x)
    }

    /// Text.
    pub fn text(s: impl AsRef<str>) -> Self {
        Self::Text(Arc::from(s.as_ref()))
    }

    /// A list.
    pub fn list(values: impl Into<Vec<Value>>) -> Self {
        Self::List(Arc::new(values.into()))
    }

    /// A map, in the order you pass the pairs.
    pub fn map(pairs: impl Into<Vec<(String, Value)>>) -> Self {
        Self::Map(Arc::new(pairs.into()))
    }

    /// The value stored under that key, if this is a map and has it.
    pub fn get(&self, key: &str) -> Option<&Value> {
        let Self::Map(pairs) = self else {
            return None;
        };
        pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// A map's values, in order — flattening a map to a list is this.
    pub fn values(&self) -> Option<Vec<&Value>> {
        let Self::Map(pairs) = self else {
            return None;
        };
        Some(pairs.iter().map(|(_, v)| v).collect())
    }

    /// Wraps something so it crosses the graph untouched.
    pub fn opaque(x: impl Any + Send + Sync) -> Self {
        Self::Opaque(Arc::new(x))
    }

    /// What is inside an opaque, if it is of this type.
    pub fn downcast<T: Any + Send + Sync>(&self) -> Option<&T> {
        let Self::Opaque(inner) = self else {
            return None;
        };
        inner.downcast_ref::<T>()
    }

    /// Whether this value, and everything inside it, can leave this process.
    ///
    /// `false` for an [`Opaque`](Self::Opaque) at any depth: what it carries
    /// only exists here. Whoever is about to send it asks first, so the refusal
    /// names the reason instead of coming out of a serializer.
    pub fn travels(&self) -> bool {
        match self {
            Self::Opaque(_) => false,
            Self::List(items) => items.iter().all(Self::travels),
            Self::Map(pairs) => pairs.iter().all(|(_, value)| value.travels()),
            Self::Null | Self::Number(_) | Self::Text(_) | Self::Bytes(_) => true,
        }
    }

    /// What to call this variant in an error message.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Number(_) => "number",
            Self::Text(_) => "text",
            Self::Bytes(_) => "bytes",
            Self::List(_) => "list",
            Self::Map(_) => "map",
            Self::Opaque(_) => "opaque",
        }
    }
}

impl PartialEq for Value {
    /// Two opaques are equal only if they are **the same one**: the core cannot
    /// compare contents it does not look at.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Number(a), Self::Number(b)) => a == b,
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Bytes(a), Self::Bytes(b)) => a == b,
            (Self::List(a), Self::List(b)) => a == b,
            (Self::Map(a), Self::Map(b)) => a == b,
            (Self::Opaque(a), Self::Opaque(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => f.write_str("Null"),
            Self::Number(x) => write!(f, "Number({x})"),
            Self::Text(s) => write!(f, "Text({s:?})"),
            Self::Bytes(b) => write!(f, "Bytes({} bytes)", b.len()),
            Self::List(items) => f.debug_tuple("List").field(items).finish(),
            Self::Map(pairs) => f.debug_tuple("Map").field(pairs).finish(),
            Self::Opaque(_) => f.write_str("Opaque(..)"),
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Self::text(s)
    }
}

impl From<f64> for Value {
    fn from(x: f64) -> Self {
        Self::Number(x)
    }
}

/// What a value looks like once it leaves this process.
///
/// A shadow of [`Value`] and not `Value` itself for one reason worth the fifty
/// lines: **it has no opaque variant**. That what only exists here cannot be
/// sent stops being a check somebody has to remember and becomes a type that
/// cannot be built. The borrowed halves are so that sending copies nothing.
#[cfg(feature = "serde")]
mod shadow {
    use super::Value;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::borrow::Cow;
    use std::sync::Arc;

    #[derive(Serialize, Deserialize)]
    pub(super) enum Shadow<'a> {
        Null,
        Number(f64),
        Text(Cow<'a, str>),
        Bytes(Cow<'a, [u8]>),
        List(Vec<Shadow<'a>>),
        Map(Vec<(Cow<'a, str>, Shadow<'a>)>),
    }

    /// The one thing that cannot become a [`Shadow`].
    pub(super) struct Opaque;

    impl<'a> TryFrom<&'a Value> for Shadow<'a> {
        type Error = Opaque;

        fn try_from(value: &'a Value) -> Result<Self, Opaque> {
            Ok(match value {
                Value::Null => Shadow::Null,
                Value::Number(x) => Shadow::Number(*x),
                Value::Text(s) => Shadow::Text(Cow::Borrowed(s)),
                Value::Bytes(bytes) => Shadow::Bytes(Cow::Borrowed(bytes)),
                Value::List(items) => Shadow::List(
                    items
                        .iter()
                        .map(Shadow::try_from)
                        .collect::<Result<_, _>>()?,
                ),
                Value::Map(pairs) => Shadow::Map(
                    pairs
                        .iter()
                        .map(|(key, value)| Ok((Cow::Borrowed(key.as_str()), value.try_into()?)))
                        .collect::<Result<_, Opaque>>()?,
                ),
                Value::Opaque(_) => return Err(Opaque),
            })
        }
    }

    impl From<Shadow<'_>> for Value {
        fn from(shadow: Shadow<'_>) -> Self {
            match shadow {
                Shadow::Null => Value::Null,
                Shadow::Number(x) => Value::Number(x),
                Shadow::Text(s) => Value::text(s),
                Shadow::Bytes(bytes) => Value::Bytes(Arc::new(bytes.into_owned())),
                Shadow::List(items) => {
                    Value::list(items.into_iter().map(Value::from).collect::<Vec<_>>())
                }
                Shadow::Map(pairs) => Value::map(
                    pairs
                        .into_iter()
                        .map(|(key, value)| (key.into_owned(), value.into()))
                        .collect::<Vec<_>>(),
                ),
            }
        }
    }

    impl Serialize for Value {
        /// # Errors
        /// If anything inside only exists in this process. Ask
        /// [`travels`](Value::travels) first and the refusal reads better.
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            Shadow::try_from(self)
                .map_err(|Opaque| {
                    serde::ser::Error::custom(
                        "an opaque value does not leave this process: what it carries \
                         only exists here",
                    )
                })?
                .serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Value {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            Shadow::deserialize(deserializer).map(Value::from)
        }
    }
}
