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

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use somatize_worker::protocol::{Capabilities, LoadMetrics, WorkerId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Status of a registered worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    /// The worker's unique id, as it registered itself.
    pub id: WorkerId,
    /// Where clients reach the worker (e.g. `ws://host:8080`). The
    /// coordinator hands this out on `/submit` and steps aside — the
    /// plan and its tensor payloads travel client→worker direct, never
    /// through the coordinator.
    pub address: String,
    /// What the worker offers: CPUs, RAM, GPUs, Python envs, tags.
    /// Placement matches required tags against these via
    /// [`matches_tags`](Self::matches_tags).
    pub capabilities: Capabilities,
    /// Load reported with the latest heartbeat; `None` until the first
    /// one arrives after registration.
    pub load: Option<LoadMetrics>,
    /// The plans currently leased to this worker. `/submit` takes a
    /// lease ([`WorkerRegistry::claim`]), `/complete` releases it
    /// ([`WorkerRegistry::release`]) — this list is what makes
    /// [`has_capacity`](Self::has_capacity) and the least-loaded
    /// tie-break mean anything.
    pub active_plans: Vec<String>,
    /// When the worker last beat (workers beat every 10s). What
    /// [`is_alive`](Self::is_alive) and the reaper compare against.
    pub last_heartbeat: DateTime<Utc>,
    /// False after an explicit [`WorkerRegistry::disconnect`];
    /// re-registering sets it back. A disconnected worker is never
    /// alive, however fresh its heartbeat.
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
    /// Read the registry, tolerating poisoning.
    ///
    /// Every access used `.unwrap()`, so one handler panicking anywhere
    /// left the whole coordinator unable to answer about any worker for
    /// the rest of the process. A `HashMap` of statuses has no invariant
    /// that spans a lock acquisition, so the data behind a poisoned lock
    /// is still sound. Same policy the worker already uses.
    fn read(&self) -> std::sync::RwLockReadGuard<'_, HashMap<WorkerId, WorkerStatus>> {
        self.workers.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<WorkerId, WorkerStatus>> {
        self.workers.write().unwrap_or_else(|e| e.into_inner())
    }

    /// An empty registry with a 30-second heartbeat timeout — three
    /// missed beats at the workers' 10-second cadence before a worker
    /// counts as dead.
    pub fn new() -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            heartbeat_timeout_secs: 30,
        }
    }

    /// Override the heartbeat timeout (builder-style). Tests use 0 to
    /// make everything instantly stale and 3600 to make nothing stale.
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
        let mut workers = self.write();
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
        let mut workers = self.write();
        if let Some(w) = workers.get_mut(worker_id) {
            w.load = Some(load);
            w.last_heartbeat = Utc::now();
        }
    }

    /// Record that `plan_id` has been placed on `worker_id`.
    ///
    /// `active_plans` was initialised to `vec![]` and never touched again,
    /// so `has_capacity` and the "least loaded" tie-break both compared
    /// zeroes: placement picked an arbitrary worker and called it balanced.
    /// Returns false if the worker is unknown.
    pub fn claim(&self, worker_id: &str, plan_id: impl Into<String>) -> bool {
        let mut workers = self.write();
        match workers.get_mut(worker_id) {
            Some(w) => {
                let plan_id = plan_id.into();
                if !w.active_plans.contains(&plan_id) {
                    w.active_plans.push(plan_id);
                }
                true
            }
            None => false,
        }
    }

    /// Release a plan, whether it finished or failed.
    pub fn release(&self, worker_id: &str, plan_id: &str) -> bool {
        let mut workers = self.write();
        match workers.get_mut(worker_id) {
            Some(w) => {
                w.active_plans.retain(|p| p != plan_id);
                true
            }
            None => false,
        }
    }

    /// Mark a worker as disconnected.
    pub fn disconnect(&self, worker_id: &str) {
        let mut workers = self.write();
        if let Some(w) = workers.get_mut(worker_id) {
            w.connected = false;
        }
    }

    /// Remove a worker entirely.
    pub fn remove(&self, worker_id: &str) {
        let mut workers = self.write();
        workers.remove(worker_id);
    }

    /// Get all alive, connected workers.
    pub fn active_workers(&self) -> Vec<WorkerStatus> {
        let workers = self.read();
        workers
            .values()
            .filter(|w| w.is_alive(self.heartbeat_timeout_secs))
            .cloned()
            .collect()
    }

    /// Get a specific worker by ID.
    pub fn get(&self, worker_id: &str) -> Option<WorkerStatus> {
        let workers = self.read();
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
        self.read().len()
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

    /// Drop workers that have stopped sending heartbeats.
    ///
    /// The predicate was `is_alive(timeout) || w.connected`, and
    /// `is_alive` already requires `connected` — so it reduced to
    /// `w.connected` and pruned nothing that was still marked connected,
    /// however long ago it had last been heard from. Which is the only
    /// case worth pruning. It also had no callers.
    ///
    /// Returns the ids that were dropped, so a caller can log them.
    pub fn prune_stale(&self) -> Vec<WorkerId> {
        let timeout = self.heartbeat_timeout_secs;
        let mut workers = self.write();
        let stale: Vec<WorkerId> = workers
            .iter()
            .filter(|(_, w)| !w.is_alive(timeout))
            .map(|(id, _)| id.clone())
            .collect();
        for id in &stale {
            workers.remove(id);
        }
        stale
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
    use somatize_worker::protocol::GpuInfo;

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

    /// A worker that stops beating is dropped, and one that keeps beating
    /// is not.
    ///
    /// `prune_stale` had no callers and a predicate that reduced to
    /// `w.connected`, so it removed nothing that mattered: a worker whose
    /// process had died stayed in the registry forever.
    #[test]
    fn a_silent_worker_is_reaped_and_a_beating_one_is_not() {
        // Zero-second timeout: everything registered in the past is stale.
        let registry = WorkerRegistry::new().with_heartbeat_timeout(0);
        registry.register("gone", "ws://h1:8080", test_caps(vec![]));
        assert_eq!(registry.total_count(), 1);

        let reaped = registry.prune_stale();
        assert_eq!(reaped, vec!["gone".to_string()]);
        assert_eq!(registry.total_count(), 0, "the dead worker is gone");

        // A generous window: a freshly registered worker survives.
        let registry = WorkerRegistry::new().with_heartbeat_timeout(3600);
        registry.register("here", "ws://h1:8080", test_caps(vec![]));
        assert!(registry.prune_stale().is_empty());
        assert_eq!(registry.total_count(), 1);
    }

    /// Placement is only balanced if placements are recorded.
    ///
    /// `active_plans` was initialised empty and never touched, so
    /// `has_capacity` and the least-loaded tie-break both compared zeroes.
    #[test]
    fn a_placed_plan_counts_against_the_worker_that_took_it() {
        let registry = WorkerRegistry::new();
        registry.register("w1", "ws://h1:8080", test_caps(vec![]));

        assert!(registry.claim("w1", "plan-1"));
        assert_eq!(registry.get("w1").unwrap().active_plans, vec!["plan-1"]);

        // At a cap of one, the worker is now full.
        assert!(registry.find_workers(&[], 1).is_empty());

        // Claiming the same plan twice is not two plans.
        registry.claim("w1", "plan-1");
        assert_eq!(registry.get("w1").unwrap().active_plans.len(), 1);

        assert!(registry.release("w1", "plan-1"));
        assert_eq!(registry.find_workers(&[], 1).len(), 1, "capacity is back");

        // An unknown worker is reported, not silently accepted.
        assert!(!registry.claim("nobody", "plan-2"));
        assert!(!registry.release("nobody", "plan-2"));
    }

    /// The least-loaded worker is the one with fewest placements.
    #[test]
    fn placement_prefers_the_less_loaded_worker() {
        let registry = WorkerRegistry::new();
        registry.register("busy", "ws://h1:8080", test_caps(vec![]));
        registry.register("idle", "ws://h2:8080", test_caps(vec![]));
        registry.claim("busy", "plan-1");
        registry.claim("busy", "plan-2");

        let best = registry
            .find_workers(&[], 4)
            .into_iter()
            .min_by_key(|w| w.active_plans.len())
            .expect("a candidate");
        assert_eq!(best.id, "idle");
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
