//! Knowledge base for experiment tracking and research memory.
//!
//! Stores experiment records, metrics, and trajectories. Supports
//! research line detection, trend analysis, and promising direction queries.
//! Backed by in-memory storage ([`MemoryKnowledgeBase`]) or optionally
//! by ChronosVector for temporal vector search (`chronos` feature).

#[cfg(feature = "chronos")]
pub mod chronos_kb;
pub mod derivation;
pub mod file_kb;
pub mod knowledge_base;
pub mod record;

#[cfg(feature = "chronos")]
pub use chronos_kb::ChronosKnowledgeBase;
pub use derivation::{Change, DerivationMove, MetricDelta, derive};
pub use file_kb::FileKnowledgeBase;
pub use knowledge_base::{KnowledgeBase, MemoryKnowledgeBase};
pub use record::{
    ChangePoint, Embedding, ExperimentRecord, RECORD_SCHEMA_VERSION, RecordKind, ResearchLine,
    Trend, slugify,
};
