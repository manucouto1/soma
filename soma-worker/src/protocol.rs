use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use soma_compiler::ExecutionPlan;
use soma_core::event::Event;
use soma_core::value::Value;

/// Unique worker identifier.
pub type WorkerId = String;

/// Unique plan execution identifier.
pub type PlanId = String;

/// Hardware and software capabilities of a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    /// Number of CPU cores.
    pub cpu_cores: usize,
    /// Total RAM in bytes.
    pub ram_bytes: u64,
    /// GPU information.
    pub gpus: Vec<GpuInfo>,
    /// Available Python environments.
    pub python_envs: Vec<String>,
    /// User-defined tags for routing (e.g. "gpu", "training", "inference").
    pub tags: Vec<String>,
}

/// GPU hardware info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub memory_bytes: u64,
}

/// Current load metrics reported by a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub gpu_usage: Vec<f32>,
    pub active_plans: usize,
    pub queue_depth: usize,
    pub timestamp: DateTime<Utc>,
}

/// A serialized plan ready for remote execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedPlan {
    pub plan_id: PlanId,
    pub plan: ExecutionPlan,
    pub input_data: Option<Value>,
    pub metadata: serde_json::Value,
}

/// Messages from Worker → Coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerToCoordinator {
    /// Worker announces itself.
    Register {
        worker_id: WorkerId,
        capabilities: Capabilities,
    },

    /// Periodic health check.
    Heartbeat {
        worker_id: WorkerId,
        load: LoadMetrics,
    },

    /// Execution event streamed back in real-time.
    Event {
        worker_id: WorkerId,
        plan_id: PlanId,
        event: Event,
    },

    /// Plan execution completed.
    PlanResult {
        worker_id: WorkerId,
        plan_id: PlanId,
        result: PlanResult,
    },
}

/// Messages from Coordinator → Worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CoordinatorToWorker {
    /// Accept worker registration.
    Registered { worker_id: WorkerId },

    /// Assign a plan for execution.
    AssignPlan { plan: SerializedPlan },

    /// Cancel a running plan.
    CancelPlan { plan_id: PlanId },

    /// Request current status.
    StatusRequest,
}

/// Result of a plan execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum PlanResult {
    Success { output: Value, duration_ms: u64 },
    Failed { error: String, duration_ms: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use soma_core::event::PlanSummary;

    #[test]
    fn capabilities_serde() {
        let caps = Capabilities {
            cpu_cores: 8,
            ram_bytes: 32 * 1024 * 1024 * 1024,
            gpus: vec![GpuInfo {
                name: "A100".into(),
                memory_bytes: 80 * 1024 * 1024 * 1024,
            }],
            python_envs: vec!["py310".into(), "py311".into()],
            tags: vec!["gpu".into(), "training".into()],
        };
        let json = serde_json::to_string(&caps).unwrap();
        let deserialized: Capabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.cpu_cores, 8);
        assert_eq!(deserialized.gpus.len(), 1);
        assert_eq!(deserialized.tags, vec!["gpu", "training"]);
    }

    #[test]
    fn worker_message_serde() {
        let msg = WorkerToCoordinator::Register {
            worker_id: "worker_01".into(),
            capabilities: Capabilities {
                cpu_cores: 4,
                ram_bytes: 16_000_000_000,
                gpus: vec![],
                python_envs: vec![],
                tags: vec!["cpu".into()],
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("Register"));
        let deserialized: WorkerToCoordinator = serde_json::from_str(&json).unwrap();
        if let WorkerToCoordinator::Register { worker_id, .. } = deserialized {
            assert_eq!(worker_id, "worker_01");
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn coordinator_message_serde() {
        let msg = CoordinatorToWorker::AssignPlan {
            plan: SerializedPlan {
                plan_id: "plan_001".into(),
                plan: ExecutionPlan::Execute {
                    node_id: "train".into(),
                },
                input_data: Some(Value::tensor(vec![1.0, 2.0], vec![2])),
                metadata: serde_json::json!({"experiment": "test"}),
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: CoordinatorToWorker = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            deserialized,
            CoordinatorToWorker::AssignPlan { .. }
        ));
    }

    #[test]
    fn plan_result_serde() {
        let success = PlanResult::Success {
            output: Value::tensor(vec![0.95], vec![1]),
            duration_ms: 1234,
        };
        let json = serde_json::to_string(&success).unwrap();
        let deserialized: PlanResult = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, PlanResult::Success { .. }));

        let failed = PlanResult::Failed {
            error: "OOM".into(),
            duration_ms: 500,
        };
        let json = serde_json::to_string(&failed).unwrap();
        let deserialized: PlanResult = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, PlanResult::Failed { .. }));
    }

    #[test]
    fn event_message_serde() {
        let msg = WorkerToCoordinator::Event {
            worker_id: "w1".into(),
            plan_id: "p1".into(),
            event: Event::RunStarted {
                run_id: "r1".into(),
                plan_summary: PlanSummary {
                    total_nodes: 3,
                    cached_nodes: 1,
                    parallel_branches: 0,
                },
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WorkerToCoordinator = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, WorkerToCoordinator::Event { .. }));
    }

    #[test]
    fn heartbeat_serde() {
        let msg = WorkerToCoordinator::Heartbeat {
            worker_id: "w1".into(),
            load: LoadMetrics {
                cpu_usage: 0.45,
                memory_usage: 0.72,
                gpu_usage: vec![0.88],
                active_plans: 2,
                queue_depth: 5,
                timestamp: Utc::now(),
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WorkerToCoordinator = serde_json::from_str(&json).unwrap();
        if let WorkerToCoordinator::Heartbeat { load, .. } = deserialized {
            assert!(load.cpu_usage > 0.0);
            assert_eq!(load.active_plans, 2);
        }
    }
}
