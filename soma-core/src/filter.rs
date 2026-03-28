use crate::cache::CacheKey;
use crate::error::Result;
use crate::value::Value;
use serde::{Deserialize, Serialize};

/// Classification of filter behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FilterKind {
    /// No state needed. `fit()` is a no-op.
    /// Example: activation function, fixed projection.
    Stateless,

    /// Learns state in `fit()`, uses it in `forward()`.
    /// Example: scaler, PCA, classifier.
    Trainable,

    /// Not differentiable. Breaks gradient flow.
    /// Example: decision tree, SQL query, file I/O.
    Opaque,
}

/// How a filter behaves in streaming mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StreamMode {
    /// State is fixed (pre-trained). Each chunk processed independently.
    FixedState,

    /// State evolves with each chunk (online learning).
    Evolving { checkpoint_every: usize },

    /// Must see all data before producing output. Forces materialization.
    Barrier,
}

/// Where a filter should execute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Distribution {
    /// Execute in the local process (default).
    Local,
    /// Execute on a specific remote target.
    Remote(RemoteTarget),
    /// Execute anywhere (scheduler decides).
    Any,
}

/// Target for remote execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemoteTarget {
    /// A specific worker by ID.
    WorkerId(String),
    /// Any worker matching a tag (e.g. "gpu", "high-memory").
    Tag(String),
}

/// Metadata about a filter, used by the compiler for optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterMeta {
    /// The name/type identifier of this filter.
    pub name: String,

    /// Classification of behavior.
    pub kind: FilterKind,

    /// Whether outputs can be cached.
    pub cacheable: bool,

    /// Whether `forward()` maintains a differentiable computational graph.
    pub differentiable: bool,

    /// Behavior in streaming mode.
    pub stream_mode: StreamMode,

    /// Where this filter should execute.
    pub distribution: Distribution,
}

/// The fundamental computation unit in Soma.
///
/// A Filter has two phases:
/// - `fit(x, y)`: Learn state from training data
/// - `forward(x, state)`: Transform data using learned state
///
/// Each phase is independently cacheable:
/// - State cache: `hash(config + training_data)`
/// - Output cache: `hash(config + state + input_data)`
pub trait Filter: Send + Sync {
    /// Compute a hash of this filter's configuration.
    /// Same config must always produce the same hash.
    /// Only public constructor parameters contribute.
    fn config_hash(&self) -> CacheKey;

    /// Learn state from training data.
    /// Returns serialized state as a Value.
    fn fit(&self, x: &Value, y: Option<&Value>) -> Result<Value>;

    /// Transform data using learned state.
    /// This is the potentially differentiable operation.
    fn forward(&self, x: &Value, state: &Value) -> Result<Value>;

    /// Metadata for the compiler.
    fn meta(&self) -> FilterMeta;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheKey;
    use crate::error::SomaError;
    use crate::value::Value;

    /// A test filter: multiplies tensor by a scale factor.
    struct TestScaler {
        scale: f64,
    }

    impl Filter for TestScaler {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[
                b"TestScaler",
                &self.scale.to_le_bytes(),
            ])
        }

        fn fit(&self, x: &Value, _y: Option<&Value>) -> Result<Value> {
            let (data, _shape) = x.as_tensor().ok_or(SomaError::Other(
                "expected tensor".into(),
            ))?;
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            Ok(Value::json(serde_json::json!({ "mean": mean })))
        }

        fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
            let (data, shape) = x.as_tensor().ok_or(SomaError::Other(
                "expected tensor".into(),
            ))?;
            let mean = state
                .as_json()
                .and_then(|j| j["mean"].as_f64())
                .ok_or(SomaError::Other("expected mean in state".into()))?;
            let scaled: Vec<f64> = data.iter().map(|v| (v - mean) * self.scale).collect();
            Ok(Value::tensor(scaled, shape.to_vec()))
        }

        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "TestScaler".into(),
                kind: FilterKind::Trainable,
                cacheable: true,
                differentiable: true,
                stream_mode: StreamMode::FixedState,
                distribution: Distribution::Local,
            }
        }
    }

    #[test]
    fn filter_config_hash_deterministic() {
        let f = TestScaler { scale: 2.0 };
        assert_eq!(f.config_hash(), f.config_hash());
    }

    #[test]
    fn filter_config_hash_sensitive() {
        let f1 = TestScaler { scale: 1.0 };
        let f2 = TestScaler { scale: 2.0 };
        assert_ne!(f1.config_hash(), f2.config_hash());
    }

    #[test]
    fn filter_fit_computes_state() {
        let f = TestScaler { scale: 1.0 };
        let x = Value::tensor(vec![2.0, 4.0, 6.0], vec![3]);
        let state = f.fit(&x, None).unwrap();
        let mean = state.as_json().unwrap()["mean"].as_f64().unwrap();
        assert!((mean - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn filter_forward_uses_state() {
        let f = TestScaler { scale: 2.0 };
        let x = Value::tensor(vec![5.0, 10.0], vec![2]);
        let state = Value::json(serde_json::json!({"mean": 5.0}));
        let result = f.forward(&x, &state).unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[0.0, 10.0]); // (5-5)*2=0, (10-5)*2=10
    }

    #[test]
    fn filter_meta_values() {
        let f = TestScaler { scale: 1.0 };
        let meta = f.meta();
        assert_eq!(meta.kind, FilterKind::Trainable);
        assert!(meta.cacheable);
        assert!(meta.differentiable);
        assert_eq!(meta.stream_mode, StreamMode::FixedState);
    }

    #[test]
    fn filter_fit_requires_tensor() {
        let f = TestScaler { scale: 1.0 };
        let result = f.fit(&Value::Empty, None);
        assert!(result.is_err());
    }

    /// A stateless test filter (identity).
    struct IdentityFilter;

    impl Filter for IdentityFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Identity"])
        }

        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }

        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            Ok(x.clone())
        }

        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Identity".into(),
                kind: FilterKind::Stateless,
                cacheable: true,
                differentiable: true,
                stream_mode: StreamMode::FixedState,
                distribution: Distribution::Local,
            }
        }
    }

    #[test]
    fn stateless_filter_passthrough() {
        let f = IdentityFilter;
        let x = Value::tensor(vec![1.0, 2.0], vec![2]);
        let state = f.fit(&x, None).unwrap();
        assert!(state.is_empty());
        let out = f.forward(&x, &state).unwrap();
        assert_eq!(out, x);
    }
}
