//! Coordinator — lightweight gateway that manages worker registration,
//! routing, and health monitoring.
//!
//! Can run as:
//! - **Standalone binary**: `soma-coordinator --token sk-xxx --port 9090`
//! - **Embedded**: `Coordinator::new().start_local()` for development
//!
//! The coordinator does NOT execute plans. It:
//! 1. Accepts worker registrations (with capabilities + heartbeats)
//! 2. Authenticates connections via bearer token
//! 3. Routes client plan submissions to appropriate workers
//! 4. Forwards worker events back to the client

use crate::protocol::{Capabilities, LoadMetrics, WorkerId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Status of a registered worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub id: WorkerId,
    pub address: String,
    pub capabilities: Capabilities,
    pub load: Option<LoadMetrics>,
    pub active_plans: Vec<String>,
    pub last_heartbeat: DateTime<Utc>,
    pub connected: bool,
}

impl WorkerStatus {
    /// Whether the worker has capacity for more work.
    pub fn has_capacity(&self, max_concurrent: usize) -> bool {
        self.connected && self.active_plans.len() < max_concurrent
    }

    /// Whether the worker matches a set of required tags.
    pub fn matches_tags(&self, required: &[String]) -> bool {
        required
            .iter()
            .all(|tag| self.capabilities.tags.contains(tag))
    }

    /// Whether the worker is considered alive (heartbeat within timeout).
    pub fn is_alive(&self, timeout_secs: i64) -> bool {
        self.connected && (Utc::now() - self.last_heartbeat).num_seconds() < timeout_secs
    }
}

/// The worker registry — tracks all known workers and their status.
#[derive(Debug, Clone)]
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<WorkerId, WorkerStatus>>>,
    heartbeat_timeout_secs: i64,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            heartbeat_timeout_secs: 30,
        }
    }

    pub fn with_heartbeat_timeout(mut self, secs: i64) -> Self {
        self.heartbeat_timeout_secs = secs;
        self
    }

    /// Register a new worker or update an existing one.
    pub fn register(
        &self,
        id: impl Into<String>,
        address: impl Into<String>,
        capabilities: Capabilities,
    ) {
        let id = id.into();
        let mut workers = self.workers.write().unwrap();
        workers.insert(
            id.clone(),
            WorkerStatus {
                id,
                address: address.into(),
                capabilities,
                load: None,
                active_plans: vec![],
                last_heartbeat: Utc::now(),
                connected: true,
            },
        );
    }

    /// Update a worker's heartbeat and load metrics.
    pub fn heartbeat(&self, worker_id: &str, load: LoadMetrics) {
        let mut workers = self.workers.write().unwrap();
        if let Some(w) = workers.get_mut(worker_id) {
            w.load = Some(load);
            w.last_heartbeat = Utc::now();
        }
    }

    /// Mark a worker as disconnected.
    pub fn disconnect(&self, worker_id: &str) {
        let mut workers = self.workers.write().unwrap();
        if let Some(w) = workers.get_mut(worker_id) {
            w.connected = false;
        }
    }

    /// Remove a worker entirely.
    pub fn remove(&self, worker_id: &str) {
        let mut workers = self.workers.write().unwrap();
        workers.remove(worker_id);
    }

    /// Get all alive, connected workers.
    pub fn active_workers(&self) -> Vec<WorkerStatus> {
        let workers = self.workers.read().unwrap();
        workers
            .values()
            .filter(|w| w.is_alive(self.heartbeat_timeout_secs))
            .cloned()
            .collect()
    }

    /// Get a specific worker by ID.
    pub fn get(&self, worker_id: &str) -> Option<WorkerStatus> {
        let workers = self.workers.read().unwrap();
        workers.get(worker_id).cloned()
    }

    /// Find workers matching required tags with available capacity.
    pub fn find_workers(&self, tags: &[String], max_concurrent: usize) -> Vec<WorkerStatus> {
        self.active_workers()
            .into_iter()
            .filter(|w| w.matches_tags(tags) && w.has_capacity(max_concurrent))
            .collect()
    }

    /// Total number of registered workers (including disconnected).
    pub fn total_count(&self) -> usize {
        self.workers.read().unwrap().len()
    }

    /// Number of alive, connected workers.
    pub fn active_count(&self) -> usize {
        self.active_workers().len()
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let workers = self.active_workers();
        let total_cpus: usize = workers.iter().map(|w| w.capabilities.cpu_cores).sum();
        let total_gpus: usize = workers.iter().map(|w| w.capabilities.gpus.len()).sum();
        let total_ram: u64 = workers.iter().map(|w| w.capabilities.ram_bytes).sum();
        format!(
            "{} workers ({} CPUs, {} GPUs, {:.1} GB RAM)",
            workers.len(),
            total_cpus,
            total_gpus,
            total_ram as f64 / (1024.0 * 1024.0 * 1024.0),
        )
    }

    /// Prune workers that haven't sent a heartbeat within the timeout.
    pub fn prune_stale(&self) {
        let mut workers = self.workers.write().unwrap();
        let timeout = self.heartbeat_timeout_secs;
        workers.retain(|_, w| w.is_alive(timeout) || w.connected);
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::GpuInfo;

    fn test_caps(tags: Vec<String>) -> Capabilities {
        Capabilities {
            cpu_cores: 4,
            ram_bytes: 8_000_000_000,
            gpus: vec![],
            python_envs: vec![],
            tags,
        }
    }

    fn gpu_caps() -> Capabilities {
        Capabilities {
            cpu_cores: 8,
            ram_bytes: 32_000_000_000,
            gpus: vec![GpuInfo {
                name: "A100".into(),
                memory_bytes: 80_000_000_000,
            }],
            python_envs: vec![],
            tags: vec!["gpu".into(), "training".into()],
        }
    }

    #[test]
    fn register_and_query() {
        let registry = WorkerRegistry::new();
        registry.register("w1", "ws://host1:8080", test_caps(vec!["cpu".into()]));
        registry.register("w2", "ws://host2:8080", gpu_caps());

        assert_eq!(registry.total_count(), 2);
        assert_eq!(registry.active_count(), 2);

        let w1 = registry.get("w1").unwrap();
        assert_eq!(w1.address, "ws://host1:8080");
        assert!(w1.connected);
    }

    #[test]
    fn find_by_tags() {
        let registry = WorkerRegistry::new();
        registry.register("cpu1", "ws://c1:8080", test_caps(vec!["cpu".into()]));
        registry.register("gpu1", "ws://g1:8080", gpu_caps());

        let gpu_workers = registry.find_workers(&["gpu".into()], 10);
        assert_eq!(gpu_workers.len(), 1);
        assert_eq!(gpu_workers[0].id, "gpu1");

        let cpu_workers = registry.find_workers(&["cpu".into()], 10);
        assert_eq!(cpu_workers.len(), 1);
    }

    #[test]
    fn disconnect_and_reconnect() {
        let registry = WorkerRegistry::new();
        registry.register("w1", "ws://host1:8080", test_caps(vec![]));
        assert_eq!(registry.active_count(), 1);

        registry.disconnect("w1");
        assert_eq!(registry.active_count(), 0);

        // Re-register = reconnect
        registry.register("w1", "ws://host1:8080", test_caps(vec![]));
        assert_eq!(registry.active_count(), 1);
    }

    #[test]
    fn summary_format() {
        let registry = WorkerRegistry::new();
        registry.register("w1", "ws://h1:8080", test_caps(vec![]));
        registry.register("w2", "ws://h2:8080", gpu_caps());

        let s = registry.summary();
        assert!(s.contains("2 workers"));
        assert!(s.contains("12 CPUs")); // 4 + 8
        assert!(s.contains("1 GPUs"));
    }

    #[test]
    fn capacity_check() {
        let registry = WorkerRegistry::new();
        registry.register("w1", "ws://h1:8080", test_caps(vec![]));

        // With max_concurrent=0, no one has capacity
        let workers = registry.find_workers(&[], 0);
        assert!(workers.is_empty());

        // With max_concurrent=1, worker with 0 active plans has capacity
        let workers = registry.find_workers(&[], 1);
        assert_eq!(workers.len(), 1);
    }
}
