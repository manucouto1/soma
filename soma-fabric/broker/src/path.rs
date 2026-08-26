//! How a pair of endpoints ends up talking, once the broker has introduced them.
//!
//! Four variants, cheapest first, and **all four are here while only two can be
//! answered**. That is the one place this crate builds ahead of its consumer,
//! deliberately: the alternative is that adding the shared mount and the relay
//! later changes [`Reply::Met`](crate::Reply), and a message that changes is a
//! version that changes for everybody. The ladder is the design; what arrives
//! later is the **probing that chooses**, not the vocabulary.
//!
//! [`Path::InProcess`] transfers nothing at all, and the temptation is to
//! answer it with a pointer to something standing in this process. It cannot:
//! every message here has to survive a round trip through bytes, including the
//! ones an embedded broker answers without leaving the process. So it answers
//! with a [`SlotId`] the client resolves against its own registry — which cost
//! nothing and bought a consistency nobody planned, since [`Path::Relayed`] has
//! exactly the same shape.
//!
//! [`Path::Direct`] means **a duplex byte stream between the two ends, with the
//! broker out of it**, and not *a TCP address*: a worker started as a child and
//! spoken to over its pipes is the same path, which the wire next door already
//! decided by making `frame` work over `impl Read`/`impl Write`. So what varies
//! is **how the stream is obtained**, which is why the variant carries an
//! [`Endpoint`]. Getting this wrong is how the ladder grows a fifth rung that
//! is really the third one twice.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

/// How two endpoints reach each other. Cheapest first.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Path {
    /// Nothing is transferred: both ends are this process, and the value is
    /// passed. The slice still runs where it was placed — this is the `.at()`
    /// that never actually left home and pays for a trip anyway.
    ///
    /// **Never inferred.** Whoever registers a host says it is in-process; a
    /// broker that worked it out by comparing addresses would quietly undo the
    /// reason a worker is a separate process, which is the GIL.
    InProcess {
        /// Where the client finds it, in a registry only the client has.
        slot: SlotId,
    },
    /// Both ends see the same filesystem: a path is written and read. Free, and
    /// a cluster already has one.
    Mount {
        /// The directory both ends agree they can see. That they really do is
        /// what the probing establishes, and the probing is not written yet.
        dir: PathBuf,
    },
    /// The two ends reach each other and speak directly, whether over a socket
    /// or over a child's pipes. One crossing, lowest latency, broker gone.
    Direct {
        /// How to obtain the stream.
        endpoint: Endpoint,
    },
    /// Neither can reach the other, so the bytes stream through the broker.
    /// No disk, no durability, and never more than a window in flight.
    Relayed {
        /// Which stream through the relay this pair was given.
        session: SessionId,
    },
}

/// How a direct stream is obtained.
///
/// Two ways of arriving at the same thing, which is why this is not two paths.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Endpoint {
    /// A worker that was already standing: `"node3:7000"`.
    Address(String),
    /// A worker to be started here, as a child, and spoken to over its pipes:
    /// `["python", "-m", "somatize.worker"]`.
    ///
    /// A whole `argv` and not a path because whoever stands a worker up decides
    /// what it is called, what environment it needs, and whether it goes inside
    /// an `srun`.
    Command(Vec<String>),
}

/// Where the client finds something already standing in its own process.
///
/// Opaque, and meaningless anywhere else: it indexes a registry the client
/// holds. Crossing a wire it is just a number, which is correct — a broker that
/// is not this process can never answer [`Path::InProcess`] anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SlotId(pub u64);

/// Which stream through a relay a pair of endpoints was given.
///
/// A string because whoever issues it decides what it looks like, and the day
/// there is a real relay it has to be **unguessable**: holding one is what lets
/// you attach to that stream. The embedded broker never issues one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub String);

impl Path {
    /// Whether two hosts given **this same path** are one place, and so share
    /// one wire and one catalog.
    ///
    /// It matters more than it looks. A worker has *one* catalog, and half of
    /// one is a different catalog: provisioning the same process twice, once
    /// per host name, swaps what it had live and takes every activation over
    /// there with it. Getting this wrong is not an extra socket, it is a run
    /// that quietly loses its state.
    ///
    /// An **address** is an identity: the same host and port is the same
    /// process. A **command** is not — it is a thing to run, and running it
    /// twice gives two of them.
    pub fn shared(&self) -> bool {
        !matches!(
            self,
            Path::Direct {
                endpoint: Endpoint::Command(_)
            }
        )
    }
}

impl fmt::Display for SlotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slot {}", self.0)
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Display for Endpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Address(addr) => f.write_str(addr),
            Self::Command(argv) => f.write_str(&argv.join(" ")),
        }
    }
}

impl fmt::Display for Path {
    /// Named the way a reader would say it out loud, because these end up in
    /// the record and in error messages both.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InProcess { slot } => write!(f, "in this process, {slot}"),
            Self::Mount { dir } => write!(f, "over the mount at {}", dir.display()),
            Self::Direct { endpoint } => write!(f, "straight to {endpoint}"),
            Self::Relayed { session } => write!(f, "relayed, session {session}"),
        }
    }
}
