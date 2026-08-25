//! What a client and a broker say to each other, and in what order.
//!
//! Two enums, together because they are a single vocabulary — the same reason
//! the wire next door keeps its `Request` and `Answer` in one file. They are
//! **not** called `Request` and `Answer`: a client holds both conversations at
//! once, and two same-named types in scope is a rename at every use site.
//!
//! # The conversation
//!
//! ```text
//! → Hello { protocol, who }            once per session
//! ← Welcome { protocol } | Refused(why)
//!
//! → Reach { host, needs }              once per host
//! ← Met { path, good_for } | Unreachable(why)
//!
//! → Done { host }                      lets the rendezvous go
//! ```
//!
//! Six messages, and the whole point of them is the **first field of the first
//! one**.
//!
//! # Why this one is versioned when the wire is not
//!
//! The wire says, correctly, that it needs no version because *"both sides are
//! the same binary from the same `cargo build`"*. For a broker that day is the
//! first: the platform's is deployed by us, and the client is installed by
//! whoever installs it, whenever they feel like it. It is the only structural
//! difference between the two conversations, and it is cheaper written down here
//! than discovered on somebody else's cluster.
//!
//! **The rule is exact match.** A broker refuses a [`PROTOCOL`] it does not
//! speak, rather than guessing which half of a stranger's vocabulary it
//! understands, and [`Reply::Welcome`] carries its own number so the refusal can
//! say something useful instead of *"no"*.
//!
//! And a caution against a promise this does not make: MessagePack through
//! `serde` writes these positionally, so **adding a field is a version bump**,
//! not a free extension. What the version buys is that the mismatch is a
//! sentence at the greeting instead of a struct read off by one at three in the
//! morning. Adding a *variant* is the same story from the other side: an older
//! reader meets an index it has no arm for and says so.
//!
//! # The two holes with nothing in them yet
//!
//! [`Ask::Hello::who`] and [`Reply::Met::good_for`] are `Option`, and the
//! embedded broker leaves both `None`. They are here because the design says
//! the platform's broker *"adds authentication, pairing, leases and metering,
//! and the protocol is identical — what it brings is policy, not mechanism"*,
//! and that sentence is only true if the slots the policy writes into already
//! exist. Without `good_for` a lease cannot be revoked without inventing a
//! message; without `who` an identity has nowhere to go. Two fields are the
//! difference between that claim and a slogan.
//!
//! The **policy** is still not here, and will not be: what an identity is worth
//! and how long a rendezvous lasts are the platform's opinions.

use crate::Path;
use serde::{Deserialize, Serialize};
use soma_next_core::Host;
use std::fmt;
use std::time::Duration;

/// The version of this vocabulary that this binary speaks.
///
/// One number for the whole conversation and not one per message: a client that
/// understands `Reach` but not `Met` is not a client, and versioning each
/// message separately would be four negotiations to save nothing.
pub const PROTOCOL: u16 = 1;

/// What the client says.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ask {
    /// Opens the session: which vocabulary I speak, and who I claim to be.
    Hello {
        /// Always [`PROTOCOL`]. First field of the first message on purpose —
        /// it is the one thing that must be readable by a binary that disagrees
        /// with this one about everything else.
        protocol: u16,
        /// Whatever the far side's policy needs in order to know who is asking.
        /// Opaque here, and `None` everywhere there is no policy.
        ///
        /// Named `who` because `as` is a keyword; the design document calls it
        /// `as` and means this.
        who: Option<Identity>,
    },
    /// Introduce me to this host.
    ///
    /// The name is the graph's — `w1`, `gpu-a` — and resolving it is the whole
    /// job of a broker. It is a [`Host`] and not a string so that the thing the
    /// engine placed and the thing the broker resolves are the same type all the
    /// way down.
    Reach {
        /// The name the graph gave it.
        host: Host,
        /// What this slice needs of whoever runs it.
        needs: Needs,
    },
    /// I am done with this host: let the rendezvous go.
    ///
    /// Nothing is held by an embedded broker, so nothing is released. It is here
    /// because without it the platform cannot tell a session that ended from one
    /// that died, and telling those apart is metering rather than politeness.
    Done {
        /// Which one.
        host: Host,
    },
}

/// What the broker answers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reply {
    /// The session is open, and this is the vocabulary I speak.
    Welcome {
        /// Mine, so a mismatch can be reported with both numbers in it.
        protocol: u16,
    },
    /// No session, and here is why. Belongs to the session and not to a
    /// rendezvous: after this there is no conversation.
    Refused(String),
    /// You two have been introduced. Go and talk; I am out of it.
    Met {
        /// How to reach them.
        path: Path,
        /// How long this rendezvous is good for, counting from when you read
        /// it. `None` is *"nobody is taking this back"*, which is every broker
        /// that has no policy.
        ///
        /// A **duration and not an instant**, and this is not a detail of
        /// convenience: an `Instant` does not serialize and means nothing off
        /// its own process, and a wall clock was already ruled out next door —
        /// two machines on a cluster disagree by minutes as a matter of course,
        /// so an expiry stamped there and read here would be a lease that
        /// expires in the past. What crosses is *how long*; *when* is stamped by
        /// whoever reads it.
        good_for: Option<Duration>,
    },
    /// That host cannot be reached, and here is why. Belongs to the rendezvous:
    /// the session survives it, and another host may well be fine.
    Unreachable(String),
}

/// Who is asking, for whoever has an opinion about it.
///
/// A string, and this crate never looks inside: the same boundary as the wire's
/// `runtime`, which is opaque there and read by whoever implements `Provision`.
/// What a token is worth is the platform's business, and the day it is a signed
/// something rather than a string, it is still bytes with a name on them.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Identity(pub String);

/// What a slice needs of whoever runs it.
///
/// **Empty, and named.** This is where *"this wants a GPU with 40 GB"* will go,
/// and it goes nowhere until there is a queue that reads it — inventing fields
/// now would be describing a matching policy nobody has written against machines
/// nobody has described. What it buys empty is that the day it fills, `Reach`
/// does not change shape.
///
/// A struct and not a unit so that filling it is an edit here rather than a
/// change of kind at every construction site.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Needs {}

impl Ask {
    /// A greeting from this binary, claiming nothing.
    pub fn hello() -> Self {
        Self::Hello {
            protocol: PROTOCOL,
            who: None,
        }
    }

    /// A greeting from this binary, claiming to be somebody.
    pub fn hello_as(who: Identity) -> Self {
        Self::Hello {
            protocol: PROTOCOL,
            who: Some(who),
        }
    }

    /// This message in bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Unreadable> {
        write(self)
    }

    /// And back from them.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Unreadable> {
        read(bytes)
    }
}

impl Reply {
    /// This message in bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Unreadable> {
        write(self)
    }

    /// And back from them.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Unreadable> {
        read(bytes)
    }

    /// The answer to a greeting: [`Reply::Welcome`] if that is a vocabulary we
    /// speak, and a [`Reply::Refused`] naming **both** numbers if it is not.
    ///
    /// Here and not in a broker because every broker owes the same answer, and
    /// three of them writing this comparison separately is three chances to
    /// write `>=` and accept a stranger.
    pub fn to_greeting(spoken: u16) -> Self {
        match spoken == PROTOCOL {
            true => Self::Welcome { protocol: PROTOCOL },
            false => Self::Refused(format!(
                "this broker speaks version {PROTOCOL} of the protocol and the client speaks \
                 {spoken}; there is no half of a vocabulary worth guessing at, so upgrade \
                 whichever of the two is behind"
            )),
        }
    }
}

fn write<T: Serialize>(what: &T) -> Result<Vec<u8>, Unreadable> {
    rmp_serde::to_vec(what).map_err(|e| Unreadable(e.to_string()))
}

/// Reads one message, and **nothing may be left over**. Lifted from the wire
/// deliberately: leftovers are as suspicious as missing bytes, no format checks
/// it for you, and the two conversations failing the same way is one thing to
/// learn instead of two.
fn read<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, Unreadable> {
    let mut rest = bytes;
    let what: T = rmp_serde::from_read(&mut rest).map_err(|e| Unreadable(e.to_string()))?;
    match rest.len() {
        0 => Ok(what),
        left => Err(Unreadable(format!("{left} bytes left over at the end"))),
    }
}

/// These bytes are not the ones that were written: truncated, left over, or
/// never a message at all.
///
/// A struct and not an enum, unlike the wire's, and the difference is the
/// domain rather than the taste: there a message can also fail because a value
/// only exists in its own process, which is a second, entirely different thing
/// to be. Nothing in this conversation carries a value, so there is one way to
/// fail and a closed set of one is a struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unreadable(String);

impl Unreadable {
    /// What went wrong.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Unreadable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "these are not the bytes that were written: {}", self.0)
    }
}

impl std::error::Error for Unreadable {}
