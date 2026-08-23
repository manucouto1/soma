//! Carrying a slice of plan to another process, and bringing back what it
//! produced.
//!
//! It is the first implementation of [`Transport`](soma_next_core::Transport),
//! and it lives **outside** the core for the same reason `python/` does: the
//! core provides the hole and depends on nobody. Here there are pipes, child
//! processes and a byte format, which are three things a core has no business
//! knowing.
//!
//! # The pieces
//!
//! | piece | role |
//! |---|---|
//! | [`Worker`] | this side: starts the process and sends it work |
//! | [`Serving`] | the far side: over standard input, or standing on a port |
//! | [`Provision`] | the hole: turns an artifact into a [`Provisioned`] |
//! | [`Artifact`] | what an empty worker is provisioned with |
//! | [`Request`] / [`Answer`] | what they say, in what order, and in bytes |
//!
//! # The two kinds of worker
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
//! A is right when you control the infrastructure: the code is already there.
//! B is what removes friction when you do **not**: `pip install` on a bare node,
//! stand up a generic worker, and send it everything from your machine.
//!
//! # What travels and what does not
//!
//! The **plan**, the **values** read there and not produced there, and the
//! **placement** — all three are data. And, if the worker is empty, an
//! **artifact** this crate does not look at, which is where the **nodes and the
//! driver** ride: whoever packs one packs the other, and this crate no more
//! knows what a driver is than what a node is.
//!
//! Not the **catalog as such**: an `Arc<dyn Node>` has no way of crossing a
//! wire. Not the **environment**: that `torch` is installed on the worker is the
//! business of whoever stands it up, and putting it in here cost the original
//! soma 420 lines of environment manager and a hot `pip install`.
//!
//! And not a [`Value::Opaque`](soma_next_core::Value::Opaque), which carries
//! something that only exists in its own process: it fails at encoding time,
//! with the host in front of you. See [`protocol`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod artifact;
mod codec;
mod frame;
mod machine;
mod protocol;
mod provision;
mod serve;
mod worker;

pub use artifact::{Artifact, Label};
pub use codec::{Codec, CodecError};
pub use machine::{Machine, filed};
pub use protocol::{Answer, MessageError, Request};
pub use provision::{Provision, ProvisionError, Provisioned};
pub use serve::Serving;
pub use worker::Worker;

// Re-exported because it is part of the protocol's vocabulary: whoever
// implements a `Provision` or reads an `Answer::Done` needs it, and hunting for
// it in another crate helps nobody.
pub use soma_next_core::Outcome;
