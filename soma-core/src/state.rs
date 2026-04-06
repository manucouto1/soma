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
    pub fn new() -> Self {
        Self::default()
    }
}

impl StateStore for MemoryStateStore {
    fn get(&self, node_id: &str) -> Result<Option<Arc<Value>>> {
        let guard = self.inner.lock().expect("MemoryStateStore poisoned");
        Ok(guard.get(node_id).cloned())
    }

    fn set(&self, node_id: &str, state: Value) -> Result<()> {
        let mut guard = self.inner.lock().expect("MemoryStateStore poisoned");
        guard.insert(node_id.to_string(), Arc::new(state));
        Ok(())
    }

    fn remove(&self, node_id: &str) -> Result<()> {
        let mut guard = self.inner.lock().expect("MemoryStateStore poisoned");
        guard.remove(node_id);
        Ok(())
    }

    fn clear(&self) -> Result<()> {
        let mut guard = self.inner.lock().expect("MemoryStateStore poisoned");
        guard.clear();
        Ok(())
    }

    fn keys(&self) -> Result<Vec<String>> {
        let guard = self.inner.lock().expect("MemoryStateStore poisoned");
        Ok(guard.keys().cloned().collect())
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
