//! Cache implementations for the Soma runtime.
//!
//! - [`MemoryCache`] — in-memory LRU with byte-level eviction
//! - [`LocalCache`] — filesystem-backed with sharded directories
//! - [`TieredCache`] — multi-level (memory → local) with auto-promotion

pub mod local;
pub mod memory;
pub mod tiered;

pub use local::LocalCache;
pub use memory::MemoryCache;
pub use tiered::TieredCache;
