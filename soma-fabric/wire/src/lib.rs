//! Carrying a slice of plan to another process, and bringing back what it
//! produced.
//!
//! The first implementation of [`Transport`](somatize_core::Transport), and it
//! lives **outside** the core for the same reason `python/` does: here there
//! are pipes, child processes and a byte format, which are three things a core
//! has no business knowing.
//!
//! | piece | role |
//! |---|---|
//! | [`Worker`] | this side: starts the process and sends it work |
//! | [`Serving`] | the far side: over standard input, or standing on a port |
//! | [`Provision`] | the hole: turns an artifact into a [`Provisioned`] |
//! | [`Artifact`] | what an empty worker is provisioned with |
//! | [`Request`] / [`Answer`] | what they say, in what order, and in bytes |
//!
//! Two kinds of worker:
//!
//! ```ignore
//! // A. the worker brings its own catalog — same code on both sides
//! let w = Worker::spawn(Command::new("./my-worker"))?;   // there: Serving::own(&c).over_stdin()
//!
//! // B. the worker starts empty and the client provisions it
//! let w = Worker::connect("node3:7000")?                 // there: Serving::provisioned(&p).listen(addr)
//!     .carrying(Artifact::new("pickle", "sha256:abc…", bytes),
//!               "cpython-3.13/cloudpickle-3.1");
//! ```
//!
//! A is right when you control the infrastructure. B is what removes friction
//! when you do **not**: `pip install` on a bare node, stand up a generic worker,
//! and send it everything from your machine.
//!
//! What travels: the **plan**, the **values** read there and not produced
//! there, the **placement**, and — for an empty worker — an **artifact** this
//! crate never looks inside, which is where the nodes ride.
//!
//! What does not: the **catalog as such**, since an `Arc<dyn Node>` has no way
//! of crossing a wire; the **environment**, which belongs to whoever stands the
//! worker up and cost the original soma 420 lines and a hot `pip install`; and
//! a [`Value::Opaque`](somatize_core::Value::Opaque), which carries something
//! that only exists in its own process and fails at encoding time, with the
//! host in front of you.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod artifact;
mod frame;
mod machine;
mod protocol;
mod provision;
mod serve;
mod worker;

pub use artifact::{Artifact, Label};
pub use machine::{Machine, filed};
pub use protocol::{Answer, MessageError, Request};
pub use provision::{Provision, ProvisionError, Provisioned};
pub use serve::Serving;
pub use worker::Worker;

// Re-exported because it is part of the protocol's vocabulary: whoever
// implements a `Provision` or reads an `Answer::Done` needs it.
pub use somatize_core::Outcome;
