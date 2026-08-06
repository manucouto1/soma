//! Wire protocol for coordinator ↔ worker communication.
//!
//! Defines message types for plan assignment, results, heartbeats,
//! Python job management, and worker capabilities.

use crate::error::{Result, WorkerError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use somatize_compiler::ExecutionPlan;
use somatize_core::data::store::{DataRef, DataStore};
use somatize_core::data::value::Value;
use somatize_core::tracking::event::Event;

/// Unique worker identifier.
pub type WorkerId = String;

/// Unique plan execution identifier.
pub type PlanId = String;

/// What this build speaks.
///
/// The wire had no version at all, while the two other formats this
/// workspace persists — tracking records and experiment records — both
/// carry one. A driver and a worker from different builds simply
/// exchanged JSON and hoped: a field the receiver did not know was
/// dropped by `#[serde(default)]`, so a plan compiled by a newer
/// coordinator ran with pieces of it silently missing, and the failure
/// surfaced as a wrong result rather than as a refusal.
///
/// Bump it whenever a change alters what a peer must understand to
/// execute a plan correctly — not for a purely additive field that an
/// older peer can safely ignore.
pub const PROTOCOL_VERSION: u32 = 1;

/// Version carried by a payload written before the field existed.
///
/// Distinct from 1 so a peer can tell "did not say" from "said 1".
fn unversioned() -> u32 {
    0
}

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
    /// Device name as the driver reports it (e.g. "A100").
    pub name: String,
    /// Total device memory in bytes.
    pub memory_bytes: u64,
}

/// Current load metrics reported by a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMetrics {
    /// CPU utilization across all cores, 0.0–1.0.
    pub cpu_usage: f32,
    /// RAM utilization as a fraction of total, 0.0–1.0.
    pub memory_usage: f32,
    /// Per-GPU utilization, in the same order as [`Capabilities::gpus`].
    pub gpu_usage: Vec<f32>,
    /// Plans currently executing.
    pub active_plans: usize,
    /// Plans accepted but not yet started.
    pub queue_depth: usize,
    /// When this snapshot was taken; heartbeats carry it so the
    /// coordinator can tell a fresh reading from a stale one.
    pub timestamp: DateTime<Utc>,
}

/// How input data is provided to a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source")]
#[non_exhaustive]
pub enum InputSource {
    /// Data embedded directly in the message (small payloads).
    Inline {
        /// The value itself, carried in the message.
        value: Value,
    },
    /// Data referenced in a remote store (large payloads).
    Reference {
        /// Where to fetch the value from; see [`InputSource::resolve`].
        data_ref: DataRef,
    },
}

impl InputSource {
    /// Resolve the input to a concrete Value.
    ///
    /// Tries the persistent [`DataStore`] first, then the temp store that
    /// HTTP uploads land in.
    ///
    /// [`DataStore`]: somatize_core::data::store::DataStore
    ///
    /// A reference that resolves nowhere is an **error**, and it did not
    /// used to be: this logged a warning and returned [`Value::Empty`].
    /// That value went on to the filter, so the failure surfaced as a
    /// `TypeError` inside somebody's own `fit` — hundreds of thousands of
    /// rows after the actual problem, and pointing at their code. The
    /// usual cause is the asymmetry this error names: the client uploaded
    /// to a store the worker was never given.
    pub fn resolve(
        &self,
        data_store: Option<&dyn somatize_core::data::store::DataStore>,
        temp_store: &somatize_core::data::store::LocalDataStore,
    ) -> Result<Value> {
        match self {
            InputSource::Inline { value } => Ok(value.clone()),
            InputSource::Reference { data_ref } => {
                if let Some(store) = data_store
                    && let Ok(val) = store.get(data_ref)
                {
                    return Ok(val);
                }
                temp_store.get(data_ref).map_err(|e| {
                    let where_it_looked = if data_store.is_some() {
                        "neither this worker's DataStore nor its temp store"
                    } else {
                        "this worker's temp store, and it has no DataStore \
                         configured — a client that uploads to one must be \
                         talking to a worker given the same one"
                    };
                    WorkerError::Transport(format!(
                        "cannot resolve the input reference {data_ref:?}: \
                         looked in {where_it_looked} ({e})"
                    ))
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
    /// What the sender speaks. See [`PROTOCOL_VERSION`].
    #[serde(default = "unversioned")]
    pub protocol_version: u32,
    /// Identifies this execution end to end — results, events and
    /// cancellations all refer back to it.
    pub plan_id: PlanId,
    /// The compiled plan the worker will execute.
    pub plan: ExecutionPlan,
    /// Input data — inline for small values, DataRef for large ones.
    pub input: Option<InputSource>,
    /// Filter definitions for the worker to reconstruct.
    #[serde(default)]
    pub filters: Vec<SerializedFilter>,
    /// Fit or Forward.
    #[serde(default)]
    pub mode: ExecutionMode,
    /// The run's experiment seed, folded into every cache key on the
    /// worker exactly as it is locally. Absent (the pre-seed wire
    /// format) means unseeded — which shares cache lines across a
    /// sweep's seeds, the bug this field exists to close.
    #[serde(default)]
    pub seed: Option<i64>,
    /// Free-form annotations that travel with the plan (experiment name,
    /// submitter, ...). The worker carries them; it never interprets them.
    pub metadata: serde_json::Value,
}

/// Encode a streaming frame.
///
/// `to_vec_named`, not `to_vec`. msgpack can write a struct either as a map
/// of named fields or as a bare array of values, and `rmp_serde::to_vec`
/// chooses the array. `Value` is an adjacently-tagged enum, which can only
/// be *read back* from named fields — so every frame carrying a tensor
/// encoded fine and then failed to decode with "invalid type: sequence,
/// expected struct variant Value::Tensor".
///
/// Nobody saw it because both receivers dropped the error: one behind
/// `if let Ok(..)`, the other behind `unwrap_or_default()`, which sent an
/// empty frame. The chunk simply never arrived.
pub fn encode_frame(msg: &StreamMessage) -> somatize_core::error::Result<Vec<u8>> {
    rmp_serde::to_vec_named(msg).map_err(|e| {
        somatize_core::error::SomaError::Other(format!("encoding a stream frame: {e}"))
    })
}

/// Decode a streaming frame.
pub fn decode_frame(bytes: &[u8]) -> somatize_core::error::Result<StreamMessage> {
    rmp_serde::from_slice(bytes).map_err(|e| {
        somatize_core::error::SomaError::Other(format!("decoding a stream frame: {e}"))
    })
}

impl SerializedPlan {
    /// A plan tagged with the version this build speaks.
    ///
    /// Callers build plans through this rather than the literal, so the
    /// version cannot be forgotten at one of the six construction sites.
    pub fn new(plan_id: impl Into<PlanId>, plan: ExecutionPlan) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            plan_id: plan_id.into(),
            plan,
            input: None,
            filters: Vec::new(),
            mode: ExecutionMode::Forward,
            seed: None,
            metadata: serde_json::json!({}),
        }
    }

    /// Attach the plan's input data — inline or by reference.
    pub fn with_input(mut self, input: InputSource) -> Self {
        self.input = Some(input);
        self
    }

    /// Attach the filter payloads the worker must reconstruct before
    /// the plan can run.
    pub fn with_filters(mut self, filters: Vec<SerializedFilter>) -> Self {
        self.filters = filters;
        self
    }

    /// Choose fit or forward execution (the default is
    /// [`ExecutionMode::Forward`]).
    pub fn with_mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Replace the free-form metadata (defaults to `{}`).
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Can this build execute the plan as its sender meant it?
    ///
    /// Refusing is the point. Executing a plan you only partly understand
    /// produces a number, and nothing downstream can tell it apart from a
    /// correct one.
    pub fn check_version(&self) -> std::result::Result<(), String> {
        if self.protocol_version == PROTOCOL_VERSION {
            return Ok(());
        }
        Err(format!(
            "protocol mismatch: this worker speaks version {PROTOCOL_VERSION}, \
             the plan was sent as version {} ({}). Upgrade whichever side is older",
            self.protocol_version,
            if self.protocol_version == 0 {
                "a build from before the wire was versioned"
            } else if self.protocol_version < PROTOCOL_VERSION {
                "older"
            } else {
                "newer"
            }
        ))
    }
}

/// Messages from Worker → Coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerToCoordinator {
    /// Worker announces itself.
    Register {
        /// The identity this worker will report in every later message.
        worker_id: WorkerId,
        /// What the worker can run — the coordinator places plans by these.
        capabilities: Capabilities,
    },

    /// Periodic health check.
    Heartbeat {
        /// Sender.
        worker_id: WorkerId,
        /// A load snapshot the coordinator reads for placement decisions.
        load: LoadMetrics,
    },

    /// Execution event streamed back in real-time.
    Event {
        /// Sender.
        worker_id: WorkerId,
        /// Which execution the event belongs to.
        plan_id: PlanId,
        /// The runtime event, forwarded verbatim.
        event: Event,
    },

    /// Plan execution completed.
    PlanResult {
        /// Sender.
        worker_id: WorkerId,
        /// Which execution finished.
        plan_id: PlanId,
        /// Success with its output, or failure with the error.
        result: PlanResult,
    },

    /// Python job progress update.
    JobProgress {
        /// Sender.
        worker_id: WorkerId,
        /// Which Python job is reporting.
        job_id: String,
        /// Coarse stage label ("environment", "execute", ...).
        phase: String,
        /// Which phase the job is in, 1-based.
        step: u32,
        /// How many phases there are in total.
        total: u32,
        /// Job-defined metrics at this point; `{}` when it has none yet.
        metrics: serde_json::Value,
    },

    /// Python job result.
    JobResult {
        /// Sender.
        worker_id: WorkerId,
        /// Which Python job finished.
        job_id: String,
        /// Whether the job's process exited cleanly.
        success: bool,
        /// The last JSON line the job printed to stdout — the job's way
        /// of reporting final metrics; `{}` when it printed none.
        metrics: serde_json::Value,
        /// Captured stdout on success; stderr followed by stdout on
        /// failure, so the traceback comes first.
        output: String,
        /// Wall-clock execution time in milliseconds.
        duration_ms: u64,
    },

    // ── Distributed training responses ──
    /// Response to GetState: trained filter states.
    StateResult {
        /// Sender.
        worker_id: WorkerId,
        /// Which execution the states came from.
        plan_id: PlanId,
        /// Trained state per requested node id.
        states: std::collections::HashMap<String, Value>,
    },

    /// A command failed, and the client can read why.
    ///
    /// Without this variant an error was sent as a bare `{"error": …}`,
    /// which is not a `WorkerToCoordinator` at all — and the client skips
    /// what it cannot parse, so it waited for a reply that had already
    /// been sent, until the socket closed. Every failure the worker
    /// reported over WebSocket hung its caller.
    Error {
        /// What went wrong, as the worker saw it.
        message: String,
    },

    /// A command that produces no data succeeded.
    ///
    /// `SetState` and `ApplyGradients` need *a* reply: the client blocks
    /// until it can parse one, so an unknown `{"type":"Ack"}` would leave
    /// it waiting until the socket closed.
    Ack {
        /// Sender.
        worker_id: WorkerId,
    },

    /// Response to GetGradients: gradient data.
    GradientsResult {
        /// Sender.
        worker_id: WorkerId,
        /// Which execution produced the gradients.
        plan_id: PlanId,
        /// Gradient payload per requested node id, opaque bytes for the
        /// aggregator to combine.
        gradients: std::collections::HashMap<String, Value>,
    },
}

/// A Python pipeline job: source files + requirements for isolated execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonPipelineJob {
    /// Identifies this job in progress updates and its result.
    pub job_id: String,
    /// Which pipeline the files define. Also names the isolated
    /// environment, so re-running the same pipeline reuses its venv.
    pub pipeline_id: String,
    /// The investigation this job belongs to — grouping across jobs,
    /// carried for the record.
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
    /// Destination path, relative to the job's working directory.
    pub path: String,
    /// Full file content, written verbatim.
    pub content: String,
}

/// Messages from Coordinator → Worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CoordinatorToWorker {
    /// Accept worker registration.
    Registered {
        /// Echoes the id the worker registered under.
        worker_id: WorkerId,
    },

    /// Assign a native Soma plan for execution.
    AssignPlan {
        /// The plan, its input, and the filters to reconstruct.
        plan: SerializedPlan,
    },

    /// Assign a Python pipeline job (with environment isolation).
    AssignPythonJob {
        /// Sources, requirements and entry point to run in isolation.
        job: PythonPipelineJob,
    },

    /// Cancel a running plan/job.
    CancelPlan {
        /// Which execution to stop.
        plan_id: PlanId,
    },

    /// Request current status.
    StatusRequest,

    /// Ping for keepalive.
    Ping,

    /// Graceful shutdown: worker should finish running plans and exit.
    Shutdown {
        /// Why the coordinator asked; for the worker's log, not logic.
        reason: String,
    },

    // ── Distributed training messages ──
    /// Request trained states from specific filters.
    GetState {
        /// Which execution holds the filters.
        plan_id: PlanId,
        /// Nodes whose trained state is wanted.
        node_ids: Vec<String>,
    },

    /// Load states into filters (e.g. after FedAvg aggregation).
    SetState {
        /// Which execution holds the filters.
        plan_id: PlanId,
        /// Replacement state per node id, loaded into each filter.
        states: std::collections::HashMap<String, Value>,
    },

    /// Request gradients from filters (for AllReduce in DataParallel).
    GetGradients {
        /// Which execution holds the filters.
        plan_id: PlanId,
        /// Nodes whose gradients are wanted.
        node_ids: Vec<String>,
    },

    /// Apply aggregated gradients (after AllReduce).
    ApplyGradients {
        /// Which execution holds the filters.
        plan_id: PlanId,
        /// Aggregated gradients per node id; each filter's optimizer
        /// steps with them.
        gradients: std::collections::HashMap<String, Value>,
    },
}

/// How output is delivered in PlanResult.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "delivery")]
#[non_exhaustive]
pub enum OutputDelivery {
    /// Small output — embedded directly in the WS message.
    Inline {
        /// The output itself.
        value: Value,
    },
    /// Large output — stored on worker, download via HTTP GET /download?key=...
    Reference {
        /// The download key; `WsTransport::resolve_output` turns it back
        /// into a value.
        data_ref: somatize_core::data::store::DataRef,
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
    /// The plan ran to completion.
    Success {
        /// The final output — inline, or a reference to download.
        output: OutputDelivery,
        /// Wall-clock execution time in milliseconds.
        duration_ms: u64,
        /// Trained states returned after Fit mode (node_id → state).
        /// Empty for Forward mode.
        #[serde(default)]
        states: std::collections::HashMap<String, Value>,
    },
    /// The plan did not complete.
    Failed {
        /// What went wrong, as the worker reported it.
        error: String,
        /// Wall-clock time until the failure, in milliseconds.
        duration_ms: u64,
    },
}

/// Streaming protocol: chunked data transfer over WebSocket Binary frames.
///
/// Wire format: msgpack-encoded StreamMessage (efficient binary, no JSON overhead).
/// Client sends StreamBegin + N × ChunkData + StreamEnd.
/// Worker responds with ChunkResult per chunk — except while a
/// Barrier-mode node is accumulating, which yields nothing until the
/// flush — and StreamComplete at the end.
///
/// The worker drives each session with the runtime's `StreamRun`, the
/// same stream executor a local `Graph.stream()` uses, so chunk caching
/// and StreamMode semantics do not fork between local and remote.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum StreamMessage {
    /// Begin a streaming session.
    StreamBegin {
        /// Names the session; every later frame quotes it.
        stream_id: String,
        /// The execution id the session runs under.
        plan_id: PlanId,
        /// Number of chunks (None if unknown ahead of time).
        total_chunks: Option<usize>,
        /// The plan to execute — input comes via chunks, not inline.
        plan: Box<SerializedPlan>,
    },
    /// A single chunk of input data.
    ChunkData {
        /// Which session the chunk belongs to.
        stream_id: String,
        /// Position in the stream, echoed back in the matching
        /// [`StreamMessage::ChunkResult`].
        chunk_index: usize,
        /// The chunk itself.
        value: Value,
    },
    /// All chunks have been sent.
    StreamEnd {
        /// Which session the sender has finished feeding.
        stream_id: String,
    },
    /// Result for a processed chunk (streamed back to client).
    ChunkResult {
        /// Which session produced the result.
        stream_id: String,
        /// Index of the input chunk this result answers.
        chunk_index: usize,
        /// The processed chunk.
        value: Value,
    },
    /// Final result after all chunks processed.
    StreamComplete {
        /// Which session finished.
        stream_id: String,
        /// The flush output on success — where Barrier-mode results
        /// arrive — or the failure that ended the run.
        result: PlanResult,
    },
}

#[cfg(test)]
mod tests {
    // ── Resolving an input that is not there ──

    #[test]
    fn an_unresolvable_reference_is_an_error_not_an_empty_value() {
        // It used to warn and return Value::Empty. That value travelled on
        // into the filter, so the failure surfaced as a TypeError inside
        // the user's own fit — long after the real problem and pointing at
        // their code.
        let temp = somatize_core::data::store::LocalDataStore::new(
            std::env::temp_dir().join("soma-resolve-test-empty"),
        );
        let source = InputSource::Reference {
            data_ref: somatize_core::data::store::DataRef::S3 {
                bucket: "nowhere".into(),
                key: "missing".into(),
                region: None,
            },
        };
        let err = source.resolve(None, &temp).unwrap_err().to_string();
        assert!(err.contains("cannot resolve the input reference"), "{err}");
        // And it says WHY, because the usual cause is a client configured
        // with a store the worker was never given.
        assert!(err.contains("no DataStore configured"), "{err}");
    }

    #[test]
    fn an_inline_input_resolves_to_itself() {
        let temp = somatize_core::data::store::LocalDataStore::new(
            std::env::temp_dir().join("soma-resolve-test-inline"),
        );
        let source = InputSource::Inline {
            value: Value::tensor(vec![1.0, 2.0], vec![2]),
        };
        assert_eq!(
            source.resolve(None, &temp).unwrap(),
            Value::tensor(vec![1.0, 2.0], vec![2])
        );
    }

    use super::*;
    use somatize_core::tracking::event::PlanSummary;

    fn sample_plan() -> SerializedPlan {
        SerializedPlan::new(
            "p1",
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
        )
        .with_input(InputSource::Inline {
            value: Value::tensor(vec![1.0, 2.0], vec![2]),
        })
    }

    /// Every `StreamMessage` variant survives the encoding it actually
    /// travels in.
    ///
    /// The streaming half of the protocol goes over WebSocket *binary*
    /// frames as msgpack, and had no round-trip test at all — every
    /// existing test covered the JSON path, which these messages never
    /// take. `rmp_serde` and `serde_json` disagree about enough
    /// (integer widths, `Option` in adjacently-tagged enums) that passing
    /// one proves nothing about the other.
    #[test]
    fn every_stream_message_survives_msgpack() {
        let messages = vec![
            StreamMessage::StreamBegin {
                stream_id: "s1".into(),
                plan_id: "p1".into(),
                total_chunks: Some(3),
                plan: Box::new(sample_plan()),
            },
            StreamMessage::StreamBegin {
                stream_id: "s1".into(),
                plan_id: "p1".into(),
                total_chunks: None,
                plan: Box::new(sample_plan()),
            },
            StreamMessage::ChunkData {
                stream_id: "s1".into(),
                chunk_index: 2,
                value: Value::tensor(vec![1.0, 2.0], vec![2]),
            },
            StreamMessage::StreamEnd {
                stream_id: "s1".into(),
            },
            StreamMessage::ChunkResult {
                stream_id: "s1".into(),
                chunk_index: 2,
                value: Value::text("done"),
            },
            StreamMessage::StreamComplete {
                stream_id: "s1".into(),
                result: PlanResult::Success {
                    output: OutputDelivery::Inline {
                        value: Value::text("out"),
                    },
                    duration_ms: 12,
                    states: Default::default(),
                },
            },
            StreamMessage::StreamComplete {
                stream_id: "s1".into(),
                result: PlanResult::Failed {
                    error: "boom".into(),
                    duration_ms: 3,
                },
            },
        ];

        for msg in messages {
            let bytes = encode_frame(&msg).expect("encode");
            let back = decode_frame(&bytes)
                .unwrap_or_else(|e| panic!("msgpack round-trip failed for {msg:?}: {e}"));
            assert_eq!(format!("{msg:?}"), format!("{back:?}"));
        }
    }

    /// A `SerializedFilter` carries cloudpickle bytes, which JSON cannot
    /// hold — hence the base64 helper, which nothing tested.
    #[test]
    fn pickled_filter_bytes_survive_json() {
        let filter = SerializedFilter {
            node_id: "clf".into(),
            pickled_filter: vec![0x80, 0x05, 0x00, 0xff, 0xfe],
            state: None,
            requirements: vec!["numpy".into()],
            trainable: true,
            config_hash: None,
        };
        let json = serde_json::to_string(&filter).unwrap();
        let back: SerializedFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pickled_filter, filter.pickled_filter);
        assert_eq!(back.requirements, filter.requirements);
    }

    /// A plan from a build that speaks a different version is refused.
    #[test]
    fn a_version_mismatch_is_refused_not_executed() {
        let mut plan = sample_plan();
        assert!(plan.check_version().is_ok());

        plan.protocol_version = PROTOCOL_VERSION + 1;
        let err = plan.check_version().expect_err("newer must be refused");
        assert!(err.contains("newer"), "{err}");

        plan.protocol_version = 0;
        let err = plan
            .check_version()
            .expect_err("unversioned must be refused");
        assert!(err.contains("before the wire was versioned"), "{err}");
    }

    /// A payload written before the field existed reads as version 0, not
    /// as "this build's version".
    #[test]
    fn a_plan_without_a_version_field_does_not_claim_ours() {
        let json = serde_json::json!({
            "plan_id": "old",
            "plan": {"Execute": {"node_id": "a"}},
            "input": null,
            "metadata": {}
        });
        let plan: SerializedPlan = serde_json::from_value(json).expect("decodes");
        assert_eq!(plan.protocol_version, 0);
        assert!(plan.check_version().is_err());
    }

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
                protocol_version: PROTOCOL_VERSION,
                plan_id: "plan_001".into(),
                plan: ExecutionPlan::Execute {
                    node_id: "train".into(),
                },
                input: Some(InputSource::Inline {
                    value: Value::tensor(vec![1.0, 2.0], vec![2]),
                }),
                filters: vec![],
                mode: ExecutionMode::default(),
                seed: None,
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
