//! Wire protocol for coordinator ↔ worker communication.
//!
//! Defines message types for plan assignment, results, heartbeats,
//! Python job management, and worker capabilities.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use somatize_compiler::ExecutionPlan;
use somatize_core::event::Event;
use somatize_core::store::{DataRef, DataStore};
use somatize_core::value::Value;

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

/// How input data is provided to a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source")]
#[non_exhaustive]
pub enum InputSource {
    /// Data embedded directly in the message (small payloads).
    Inline { value: Value },
    /// Data referenced in a remote store (large payloads).
    Reference { data_ref: DataRef },
}

impl InputSource {
    /// Resolve the input to a concrete Value.
    /// Tries persistent DataStore first, then temp store for HTTP uploads.
    pub fn resolve(
        &self,
        data_store: Option<&dyn somatize_core::store::DataStore>,
        temp_store: &somatize_core::store::LocalDataStore,
    ) -> Value {
        match self {
            InputSource::Inline { value } => value.clone(),
            InputSource::Reference { data_ref } => {
                if let Some(store) = data_store
                    && let Ok(val) = store.get(data_ref)
                {
                    return val;
                }
                temp_store.get(data_ref).unwrap_or_else(|e| {
                    tracing::warn!("Failed to resolve DataRef: {e}");
                    Value::Empty
                })
            }
        }
    }
}

/// A serialized filter: cloudpickle bytes to reconstruct on the worker.
///
/// Uses cloudpickle (like Spark/Dask/Ray) to serialize the full Python object
/// including bytecode, closures, and cross-module dependencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedFilter {
    /// Node ID this filter is registered under.
    pub node_id: String,
    /// cloudpickle.dumps() bytes (base64-encoded for JSON transport).
    #[serde(with = "base64_bytes")]
    pub pickled_filter: Vec<u8>,
    /// Trained state (if fitted).
    pub state: Option<Value>,
    /// Pip requirements detected from the filter's imports (e.g. ["torch", "transformers"]).
    #[serde(default)]
    pub requirements: Vec<String>,
    /// Whether the filter is trainable (has meaningful fit()) or stateless.
    #[serde(default)]
    pub trainable: bool,
    /// The filter's real config hash from the coordinator, so cache keys
    /// computed on the worker match those computed locally. `None` for
    /// payloads from older coordinators — the worker then falls back to
    /// hashing the pickled filter bytes (config changes still invalidate).
    #[serde(default)]
    pub config_hash: Option<somatize_core::cache::CacheKey>,
}

/// Serde helper: `Vec<u8>` ↔ base64 string for JSON-safe binary transport.
mod base64_bytes {
    use base64::engine::{Engine, general_purpose::STANDARD};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        STANDARD.encode(bytes).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        STANDARD.decode(s).map_err(serde::de::Error::custom)
    }
}

/// Execution mode: fit (training) or forward (inference).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum ExecutionMode {
    /// Training: fit each filter, then forward to propagate outputs.
    Fit {
        /// Supervised labels (optional).
        y: Option<Value>,
        /// If set, the worker splits the input into batches internally.
        /// Model is loaded once, batches processed in a loop.
        #[serde(default)]
        batch_size: Option<usize>,
    },
    /// Inference: forward only (default).
    #[default]
    Forward,
}

/// A serialized plan ready for remote execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedPlan {
    pub plan_id: PlanId,
    pub plan: ExecutionPlan,
    /// Input data — inline for small values, DataRef for large ones.
    pub input: Option<InputSource>,
    /// Filter definitions for the worker to reconstruct.
    #[serde(default)]
    pub filters: Vec<SerializedFilter>,
    /// Fit or Forward.
    #[serde(default)]
    pub mode: ExecutionMode,
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

    /// Python job progress update.
    JobProgress {
        worker_id: WorkerId,
        job_id: String,
        phase: String,
        step: u32,
        total: u32,
        metrics: serde_json::Value,
    },

    /// Python job result.
    JobResult {
        worker_id: WorkerId,
        job_id: String,
        success: bool,
        metrics: serde_json::Value,
        output: String,
        duration_ms: u64,
    },

    // ── Distributed training responses ──
    /// Response to GetState: trained filter states.
    StateResult {
        worker_id: WorkerId,
        plan_id: PlanId,
        states: std::collections::HashMap<String, Value>,
    },

    /// Response to GetGradients: gradient data.
    GradientsResult {
        worker_id: WorkerId,
        plan_id: PlanId,
        gradients: std::collections::HashMap<String, Value>,
    },
}

/// A Python pipeline job: source files + requirements for isolated execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonPipelineJob {
    pub job_id: String,
    pub pipeline_id: String,
    pub investigation_id: String,
    /// Source files: path → content
    pub files: Vec<PipelineFile>,
    /// pip requirements (content of requirements.txt)
    pub requirements: String,
    /// Entry point: which file/function to execute
    pub entry_point: String,
    /// Input data (JSON-serialized)
    pub input_data: Option<serde_json::Value>,
    /// Extra parameters
    pub params: serde_json::Value,
}

/// A source file in a pipeline job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineFile {
    pub path: String,
    pub content: String,
}

/// Messages from Coordinator → Worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CoordinatorToWorker {
    /// Accept worker registration.
    Registered { worker_id: WorkerId },

    /// Assign a native Soma plan for execution.
    AssignPlan { plan: SerializedPlan },

    /// Assign a Python pipeline job (with environment isolation).
    AssignPythonJob { job: PythonPipelineJob },

    /// Cancel a running plan/job.
    CancelPlan { plan_id: PlanId },

    /// Request current status.
    StatusRequest,

    /// Ping for keepalive.
    Ping,

    /// Graceful shutdown: worker should finish running plans and exit.
    Shutdown { reason: String },

    // ── Distributed training messages ──
    /// Request trained states from specific filters.
    GetState {
        plan_id: PlanId,
        node_ids: Vec<String>,
    },

    /// Load states into filters (e.g. after FedAvg aggregation).
    SetState {
        plan_id: PlanId,
        states: std::collections::HashMap<String, Value>,
    },

    /// Request gradients from filters (for AllReduce in DataParallel).
    GetGradients {
        plan_id: PlanId,
        node_ids: Vec<String>,
    },

    /// Apply aggregated gradients (after AllReduce).
    ApplyGradients {
        plan_id: PlanId,
        gradients: std::collections::HashMap<String, Value>,
    },
}

/// How output is delivered in PlanResult.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "delivery")]
#[non_exhaustive]
pub enum OutputDelivery {
    /// Small output — embedded directly in the WS message.
    Inline { value: Value },
    /// Large output — stored on worker, download via HTTP GET /download?key=...
    Reference {
        data_ref: somatize_core::store::DataRef,
    },
}

// `OutputDelivery::resolve` lived here and had no callers. It downloaded a
// referenced output over HTTP and mapped *every* failure — connection
// refused, auth rejected, malformed body — to `Value::Empty`, so a failed
// download was indistinguishable from a plan that legitimately produced
// nothing. The working implementation is `WsTransport::resolve_output`,
// which does the same download and returns `Result`; keeping a lenient
// duplicate beside it only invited a caller to pick the wrong one.

/// Result of a plan execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum PlanResult {
    Success {
        output: OutputDelivery,
        duration_ms: u64,
        /// Trained states returned after Fit mode (node_id → state).
        /// Empty for Forward mode.
        #[serde(default)]
        states: std::collections::HashMap<String, Value>,
    },
    Failed {
        error: String,
        duration_ms: u64,
    },
}

/// Streaming protocol: chunked data transfer over WebSocket Binary frames.
///
/// Wire format: msgpack-encoded StreamMessage (efficient binary, no JSON overhead).
/// Client sends StreamBegin + N × ChunkData + StreamEnd.
/// Worker responds with ChunkResult per chunk + StreamComplete at the end.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum StreamMessage {
    /// Begin a streaming session.
    StreamBegin {
        stream_id: String,
        plan_id: PlanId,
        /// Number of chunks (None if unknown ahead of time).
        total_chunks: Option<usize>,
        /// The plan to execute — input comes via chunks, not inline.
        plan: Box<SerializedPlan>,
    },
    /// A single chunk of input data.
    ChunkData {
        stream_id: String,
        chunk_index: usize,
        value: Value,
    },
    /// All chunks have been sent.
    StreamEnd { stream_id: String },
    /// Result for a processed chunk (streamed back to client).
    ChunkResult {
        stream_id: String,
        chunk_index: usize,
        value: Value,
    },
    /// Final result after all chunks processed.
    StreamComplete {
        stream_id: String,
        result: PlanResult,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::event::PlanSummary;

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
                input: Some(InputSource::Inline {
                    value: Value::tensor(vec![1.0, 2.0], vec![2]),
                }),
                filters: vec![],
                mode: ExecutionMode::default(),
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
            output: OutputDelivery::Inline {
                value: Value::tensor(vec![0.95], vec![1]),
            },
            duration_ms: 1234,
            states: std::collections::HashMap::new(),
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
