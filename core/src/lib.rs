//! The soma-next core: the graph structure, the contracts for what gets
//! executed, the shape of an execution, and the engine that walks it.
//!
//! No `#[pyclass]` here. The moment a core type carries one, it can no longer
//! be used without a Python interpreter loaded, and that does not come undone.
//!
//! The pieces and their roles, which are easy to confuse:
//!
//! | piece | role |
//! |---|---|
//! | [`Graph`] | the **structure**: which nodes exist and how they connect. Pure data |
//! | [`Catalog`] | the **store**: which implementation belongs to each node |
//! | [`Placement`] | **where** each node runs. Pure data, and separate from the plan |
//! | [`Device`] | the place inside a machine: `cpu`, `cuda:0`, `meta` |
//! | [`Host`] | the place that **is** another machine or process, by name |
//! | [`Node`] | the **contract** for what a node executes |
//! | [`Driver`] | who **serves** what a step asks for |
//! | [`Plan`] | the **decided shape** of an execution |
//! | [`compile`] | from the structure to the shape |
//! | [`Executor`] | the **engine** |
//! | [`Wire`] | declaring a graph as an expression: `a >> (b \| c) >> d` |
//!
//! One file per type, with its inherent `impl`s and the errors its operations
//! produce. See `CLAUDE.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod build;
mod catalog;
mod device;
mod driver;
mod execution;
mod graph;
mod host;
mod node;
mod placement;
mod plan;
mod value;

pub use build::{Wire, node};
pub use catalog::Catalog;
pub use device::{Device, DeviceError};
pub use driver::{Driver, DriverError};
pub use execution::{Executor, RunError};
pub use graph::{Edge, Graph, GraphError, NodeId};
pub use host::Host;
pub use node::{Ctx, Node, NodeError, Transition};
pub use placement::Placement;
pub use plan::{CompileError, Plan, compile};
pub use value::Value;
