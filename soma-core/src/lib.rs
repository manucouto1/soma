pub mod cache;
pub mod error;
pub mod event;
pub mod filter;
pub mod graph;
pub mod search;
pub mod study;
pub mod value;

// Re-export core types for convenience.
pub use cache::{CacheKey, CacheStore, CacheTier, EntryMeta, Origin};
pub use error::{Result, SomaError};
pub use event::{Event, MetricRecord, PlanSummary, RunId, StudyId, TrialId};
pub use filter::{Distribution, Filter, FilterKind, FilterMeta, RemoteTarget, StreamMode};
pub use graph::{Edge, EdgeKind, Graph, Node, NodeId};
pub use search::{Scale, SearchDimension, SearchSpace, Searchable};
pub use study::{
    Direction, Objective, PruningStrategy, SearchStrategy, Study, Trial, TrialState,
};
pub use value::Value;

// Re-export derive macro
pub use soma_macros::SomaFilter;
