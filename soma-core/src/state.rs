//! Trained-state storage — authoritative data produced by `fit()`.
//!
//! States are distinct from [`CacheStore`](crate::CacheStore) entries:
//! - Cache entries are **discardable** — the system can recompute them.
//! - States are **authoritative** — they are the product of training and
//!   belong to the Graph that produced them. They must not be evicted
//!   arbitrarily.
//!
//! [`StateStore`] is the trait; implementations may keep states in memory,
//! on local disk, or in object storage. States are returned as
//! `Arc<Value>` so the hot forward path can borrow them (`&*arc`) without
//! cloning potentially-large tensors.

use crate::error::Result;
use crate::value::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Storage for trained filter states, keyed by node id.
///
/// Implementations must be `Send + Sync` and use interior mutability so
/// the store can be shared (via `Arc`) across the executor and the
/// graph session.
pub trait StateStore: Send + Sync {
    /// Fetch the state for `node_id`, if present.
    fn get(&self, node_id: &str) -> Result<Option<Arc<Value>>>;

    /// Store `state` under `node_id`, replacing any previous value.
    fn set(&self, node_id: &str, state: Value) -> Result<()>;

    /// Remove the state for `node_id`, if present.
    fn remove(&self, node_id: &str) -> Result<()>;

    /// Drop all stored states.
    fn clear(&self) -> Result<()>;

    /// List all node ids that currently have a stored state.
    fn keys(&self) -> Result<Vec<String>>;
}

/// In-memory [`StateStore`] — the default backend.
///
/// States live as `Arc<Value>` so reads are zero-copy (just `Arc::clone`)
/// and multiple consumers can hold references concurrently.
#[derive(Default)]
pub struct MemoryStateStore {
    inner: Mutex<HashMap<String, Arc<Value>>>,
}

impl MemoryStateStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Lock the map, tolerating poisoning.
    ///
    /// The runtime catches panics from user code and keeps going, so a
    /// recovered panic must not leave the store permanently unusable —
    /// which is exactly what propagating the poison would do. The map's
    /// invariants do not span a lock acquisition, so the data behind a
    /// poisoned lock is still sound. Same policy as the LRU cache.
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Arc<Value>>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl StateStore for MemoryStateStore {
    fn get(&self, node_id: &str) -> Result<Option<Arc<Value>>> {
        Ok(self.lock().get(node_id).cloned())
    }

    fn set(&self, node_id: &str, state: Value) -> Result<()> {
        self.lock().insert(node_id.to_string(), Arc::new(state));
        Ok(())
    }

    fn remove(&self, node_id: &str) -> Result<()> {
        self.lock().remove(node_id);
        Ok(())
    }

    fn clear(&self) -> Result<()> {
        self.lock().clear();
        Ok(())
    }

    fn keys(&self) -> Result<Vec<String>> {
        Ok(self.lock().keys().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_store_roundtrip() {
        let store = MemoryStateStore::new();
        assert!(store.get("a").unwrap().is_none());

        store
            .set("a", Value::json(serde_json::json!({"mean": 5.0})))
            .unwrap();
        let state = store.get("a").unwrap().unwrap();
        assert_eq!(state.as_json().unwrap()["mean"], 5.0);

        // Same Arc returned on subsequent reads
        let s1 = store.get("a").unwrap().unwrap();
        let s2 = store.get("a").unwrap().unwrap();
        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn memory_store_remove_and_clear() {
        let store = MemoryStateStore::new();
        store.set("a", Value::Empty).unwrap();
        store.set("b", Value::Empty).unwrap();
        assert_eq!(store.keys().unwrap().len(), 2);

        store.remove("a").unwrap();
        assert!(store.get("a").unwrap().is_none());
        assert!(store.get("b").unwrap().is_some());

        store.clear().unwrap();
        assert!(store.keys().unwrap().is_empty());
    }

    #[test]
    fn memory_store_overwrites() {
        let store = MemoryStateStore::new();
        store
            .set("a", Value::json(serde_json::json!({"v": 1})))
            .unwrap();
        store
            .set("a", Value::json(serde_json::json!({"v": 2})))
            .unwrap();
        let state = store.get("a").unwrap().unwrap();
        assert_eq!(state.as_json().unwrap()["v"], 2);
    }
}
