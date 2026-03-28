use crate::cache::CacheKey;
use crate::schema::Schema;
use crate::value::Value;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A lazy reference to a value that can be materialized on demand.
///
/// This is the core of Soma's data virtualization. Instead of computing
/// and storing every intermediate result, values are represented as
/// references that carry enough information to:
///
/// - Inspect schema without loading data
/// - Check whether the data is already available
/// - Compute it when needed
///
/// Like Denodo's data virtualization, but for computation rather than SQL.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum VirtualValue {
    /// Already computed and in memory. Ready to use.
    Materialized { value: Value, schema: Schema },

    /// Stored in cache (K/V store). Can be loaded on demand.
    Cached { key: CacheKey, schema: Schema },

    /// Not computed yet. Carries the "recipe" to produce it:
    /// which node produces it, and what its cache key would be.
    Deferred {
        producer_node_id: String,
        cache_key: CacheKey,
        schema: Schema,
    },

    /// A stream that materializes chunk by chunk.
    Stream { source_id: String, schema: Schema },
}

/// Status of a VirtualValue without inspecting the actual data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueStatus {
    /// In memory, ready to use.
    InMemory,
    /// On disk/cache, needs loading.
    OnDisk,
    /// Not computed, needs execution.
    NotComputed,
    /// Streaming, partial data.
    Streaming,
}

impl VirtualValue {
    /// Create a materialized value with auto-inferred schema.
    pub fn materialized(value: Value) -> Self {
        let schema = Self::infer_schema(&value);
        Self::Materialized { value, schema }
    }

    /// Create a materialized value with explicit schema.
    pub fn materialized_with_schema(value: Value, schema: Schema) -> Self {
        Self::Materialized { value, schema }
    }

    /// Create a cached reference.
    pub fn cached(key: CacheKey, schema: Schema) -> Self {
        Self::Cached { key, schema }
    }

    /// Create a deferred (not yet computed) reference.
    pub fn deferred(
        producer_node_id: impl Into<String>,
        cache_key: CacheKey,
        schema: Schema,
    ) -> Self {
        Self::Deferred {
            producer_node_id: producer_node_id.into(),
            cache_key,
            schema,
        }
    }

    /// Get the schema without materializing the value.
    pub fn schema(&self) -> &Schema {
        match self {
            Self::Materialized { schema, .. }
            | Self::Cached { schema, .. }
            | Self::Deferred { schema, .. }
            | Self::Stream { schema, .. } => schema,
        }
    }

    /// Get the current status.
    pub fn status(&self) -> ValueStatus {
        match self {
            Self::Materialized { .. } => ValueStatus::InMemory,
            Self::Cached { .. } => ValueStatus::OnDisk,
            Self::Deferred { .. } => ValueStatus::NotComputed,
            Self::Stream { .. } => ValueStatus::Streaming,
        }
    }

    /// Get the materialized value if already in memory.
    pub fn as_value(&self) -> Option<&Value> {
        match self {
            Self::Materialized { value, .. } => Some(value),
            _ => None,
        }
    }

    /// Get the cache key (if this value is cached or deferred).
    pub fn cache_key(&self) -> Option<&CacheKey> {
        match self {
            Self::Cached { key, .. } | Self::Deferred { cache_key: key, .. } => Some(key),
            _ => None,
        }
    }

    /// Materialize from a cache store. Returns the value if found, None if not cached.
    pub fn try_load(
        &self,
        cache: &dyn crate::cache::CacheStore,
    ) -> crate::error::Result<Option<Value>> {
        match self {
            Self::Materialized { value, .. } => Ok(Some(value.clone())),
            Self::Cached { key, .. } => cache.get(key),
            Self::Deferred { cache_key, .. } => cache.get(cache_key),
            _ => Ok(None),
        }
    }

    /// Upgrade this reference: if Deferred, check cache; if Cached, load.
    /// Returns a new VirtualValue that may be closer to Materialized.
    pub fn resolve(&self, cache: &dyn crate::cache::CacheStore) -> crate::error::Result<Self> {
        match self {
            Self::Materialized { .. } => Ok(self.clone()),
            Self::Cached { key, schema } => {
                if let Some(value) = cache.get(key)? {
                    Ok(Self::Materialized {
                        value,
                        schema: schema.clone(),
                    })
                } else {
                    Ok(self.clone()) // still cached but value missing
                }
            }
            Self::Deferred {
                cache_key, schema, ..
            } => {
                if let Some(value) = cache.get(cache_key)? {
                    Ok(Self::Materialized {
                        value,
                        schema: schema.clone(),
                    })
                } else {
                    Ok(self.clone()) // still deferred
                }
            }
            _ => Ok(self.clone()),
        }
    }

    /// Infer schema from a concrete Value.
    fn infer_schema(value: &Value) -> Schema {
        match value {
            Value::Tensor { values: _, shape } => Schema {
                dtype: crate::schema::DataType::Float64,
                shape: Some(
                    shape
                        .iter()
                        .map(|&d| crate::schema::Dimension::Fixed(d))
                        .collect(),
                ),
            },
            Value::Json(_) => Schema::json(),
            Value::Bytes(_) => Schema::bytes(),
            Value::Empty => Schema::dynamic(crate::schema::DataType::Float64),
        }
    }
}

impl fmt::Display for VirtualValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Materialized { schema, value } => {
                write!(f, "Materialized({schema}, size={})", value.size())
            }
            Self::Cached { key, schema } => {
                write!(f, "Cached({schema}, key={key})")
            }
            Self::Deferred {
                producer_node_id,
                schema,
                ..
            } => {
                write!(f, "Deferred({schema}, producer={producer_node_id})")
            }
            Self::Stream { source_id, schema } => {
                write!(f, "Stream({schema}, source={source_id})")
            }
        }
    }
}

impl From<Value> for VirtualValue {
    fn from(value: Value) -> Self {
        Self::materialized(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheKey;
    use crate::schema::{DataType, Schema};

    #[test]
    fn materialized_from_value() {
        let val = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);
        let vv = VirtualValue::materialized(val.clone());

        assert_eq!(vv.status(), ValueStatus::InMemory);
        assert_eq!(vv.as_value(), Some(&val));
        assert_eq!(vv.schema().dtype, DataType::Float64);
        assert_eq!(vv.schema().rank(), Some(1));
    }

    #[test]
    fn cached_reference() {
        let key = CacheKey::hash_data(b"test");
        let schema = Schema::vector(DataType::Float64, 100);
        let vv = VirtualValue::cached(key.clone(), schema.clone());

        assert_eq!(vv.status(), ValueStatus::OnDisk);
        assert_eq!(vv.cache_key(), Some(&key));
        assert_eq!(vv.schema(), &schema);
        assert!(vv.as_value().is_none());
    }

    #[test]
    fn deferred_reference() {
        let key = CacheKey::hash_data(b"deferred");
        let schema = Schema::batched(DataType::Float64, &[128]);
        let vv = VirtualValue::deferred("my_node", key.clone(), schema.clone());

        assert_eq!(vv.status(), ValueStatus::NotComputed);
        assert_eq!(vv.cache_key(), Some(&key));
        assert_eq!(vv.schema(), &schema);
    }

    #[test]
    fn schema_inferred_from_tensor() {
        let val = Value::tensor(vec![0.0; 12], vec![3, 4]);
        let vv = VirtualValue::materialized(val);
        assert_eq!(vv.schema().dtype, DataType::Float64);
        assert_eq!(vv.schema().rank(), Some(2));
    }

    #[test]
    fn schema_inferred_from_json() {
        let val = Value::json(serde_json::json!({"a": 1}));
        let vv = VirtualValue::materialized(val);
        assert_eq!(vv.schema().dtype, DataType::Json);
    }

    #[test]
    fn display_formatting() {
        let vv = VirtualValue::materialized(Value::tensor(vec![1.0], vec![1]));
        assert!(vv.to_string().contains("Materialized"));

        let vv = VirtualValue::cached(CacheKey::hash_data(b"k"), Schema::json());
        assert!(vv.to_string().contains("Cached"));

        let vv = VirtualValue::deferred("node_1", CacheKey::hash_data(b"k"), Schema::json());
        assert!(vv.to_string().contains("Deferred"));
    }

    #[test]
    fn from_value_conversion() {
        let val = Value::tensor(vec![1.0, 2.0], vec![2]);
        let vv: VirtualValue = val.clone().into();
        assert_eq!(vv.status(), ValueStatus::InMemory);
        assert_eq!(vv.as_value(), Some(&val));
    }

    #[test]
    fn resolve_materialized_stays_materialized() {
        use crate::cache::CacheStore;
        use std::collections::HashMap;
        use std::sync::Mutex;

        // Simple mock cache
        struct EmptyCache;
        impl CacheStore for EmptyCache {
            fn get(&self, _: &CacheKey) -> crate::error::Result<Option<Value>> {
                Ok(None)
            }
            fn put(&self, _: &CacheKey, _: &Value) -> crate::error::Result<()> {
                Ok(())
            }
            fn exists(&self, _: &CacheKey) -> crate::error::Result<bool> {
                Ok(false)
            }
            fn remove(&self, _: &CacheKey) -> crate::error::Result<()> {
                Ok(())
            }
            fn metadata(
                &self,
                _: &CacheKey,
            ) -> crate::error::Result<Option<crate::cache::EntryMeta>> {
                Ok(None)
            }
        }

        let val = Value::tensor(vec![1.0], vec![1]);
        let vv = VirtualValue::materialized(val);
        let resolved = vv.resolve(&EmptyCache).unwrap();
        assert_eq!(resolved.status(), ValueStatus::InMemory);
    }

    #[test]
    fn resolve_deferred_checks_cache() {
        use crate::cache::{CacheStore, EntryMeta};
        use std::collections::HashMap;
        use std::sync::Mutex;

        struct TestCache {
            store: Mutex<HashMap<CacheKey, Value>>,
        }
        impl TestCache {
            fn with(key: CacheKey, value: Value) -> Self {
                let mut store = HashMap::new();
                store.insert(key, value);
                Self {
                    store: Mutex::new(store),
                }
            }
        }
        impl CacheStore for TestCache {
            fn get(&self, key: &CacheKey) -> crate::error::Result<Option<Value>> {
                Ok(self.store.lock().unwrap().get(key).cloned())
            }
            fn put(&self, _: &CacheKey, _: &Value) -> crate::error::Result<()> {
                Ok(())
            }
            fn exists(&self, key: &CacheKey) -> crate::error::Result<bool> {
                Ok(self.store.lock().unwrap().contains_key(key))
            }
            fn remove(&self, _: &CacheKey) -> crate::error::Result<()> {
                Ok(())
            }
            fn metadata(&self, _: &CacheKey) -> crate::error::Result<Option<EntryMeta>> {
                Ok(None)
            }
        }

        let key = CacheKey::hash_data(b"cached_value");
        let expected = Value::tensor(vec![42.0], vec![1]);
        let cache = TestCache::with(key.clone(), expected.clone());

        // Deferred → resolve → Materialized (if found in cache)
        let vv = VirtualValue::deferred("producer", key, Schema::vector(DataType::Float64, 1));
        assert_eq!(vv.status(), ValueStatus::NotComputed);

        let resolved = vv.resolve(&cache).unwrap();
        assert_eq!(resolved.status(), ValueStatus::InMemory);
        assert_eq!(resolved.as_value(), Some(&expected));
    }

    #[test]
    fn serde_roundtrip() {
        let values = vec![
            VirtualValue::materialized(Value::tensor(vec![1.0], vec![1])),
            VirtualValue::cached(CacheKey::hash_data(b"k"), Schema::json()),
            VirtualValue::deferred(
                "n",
                CacheKey::hash_data(b"d"),
                Schema::vector(DataType::Float64, 10),
            ),
        ];
        for vv in values {
            let json = serde_json::to_string(&vv).unwrap();
            let deserialized: VirtualValue = serde_json::from_str(&json).unwrap();
            assert_eq!(vv.status(), deserialized.status());
            assert_eq!(vv.schema(), deserialized.schema());
        }
    }
}
