//! The rendezvous: turning the name a graph gave a host into a way of reaching
//! it.
//!
//! A client always does the same thing — it talks to a broker — and the only
//! thing that changes between having a platform account, having a head node,
//! and having neither is **which** broker, which is a URL. There is no second
//! code path and no degraded mode.
//!
//! | deployment | what it is | what it adds |
//! |---|---|---|
//! | embedded | in the client's own process | nothing. It is what makes soma work alone |
//! | local | a process on a head node | reachable by more than one client |
//! | platform | ours | authentication, pairing, leases, metering |
//!
//! Here today: `protocol` and [`Path`] — three questions, four answers, and the
//! four ways a pair of endpoints can end up talking — and [`Embedded`], the
//! first of the three deployments. The other two arrive with their consumers,
//! and so does a `Broker` trait: today it would have one implementor.
//!
//! The one decision worth knowing before reading the code: **control and cargo
//! go by different routes.** What crosses here is a rendezvous — tens of bytes,
//! once per host per session — while what crosses the wire next door is an
//! activation. The broker is in the first and steps out of the second, which is
//! why an embedded broker can be a thread with real serialized messages and
//! still cost nothing anyone can measure.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod embedded;
mod path;
mod protocol;
mod reaching;
mod session;

pub use embedded::{Embedded, Unanswered};
pub use path::{Endpoint, Path, SessionId, SlotId};
pub use protocol::{Ask, Identity, Needs, PROTOCOL, Reply, Unreadable};
pub use reaching::Reaching;
pub use session::Session;

// Re-exported because it is this conversation's subject: a `Reach` names one,
// and whoever answers has to hold the same type the engine placed.
pub use somatize_core::Host;
