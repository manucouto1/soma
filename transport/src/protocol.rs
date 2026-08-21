//! What the two sides say to each other, and in what order.
//!
//! Two enums, together because they are a single vocabulary: a message on its
//! own means nothing without the one that answers it. No `#[non_exhaustive]`,
//! like the rest of the project's enums.
//!
//! # The conversation
//!
//! ```text
//! → Hello { runtime, offering }        once per session
//! ← Ready                              "I have a catalog, go ahead"
//!   | Send                             "I do not have that artifact"
//!   | Refused(why)
//!
//! → Provision { bytes }                only if it answered Send
//! ← Ready | Refused(why)
//!
//! → Work { plan, input, known, keys, placement, memory }     n times
//! ← Done { last, produced, keys } | Failed(why)
//! ```
//!
//! # Why it is announced before being sent
//!
//! The `Hello` carries the artifact's **name**, not the artifact: a pickled
//! catalog with weights inside is megabytes, and asking "do you have
//! `sha256:abc…`?" is forty bytes.
//!
//! And a consequence worth more than the saving: **the day a store exists — a
//! MinIO, an S3, a shared folder — the worker tries it before answering `Send`,
//! and the protocol does not change a line.** The store becomes a cache in front
//! of this conversation rather than a fork in the design. It is git's
//! `have`/`want` and `docker push`'s layer exchange.
//!
//! # The same thing in bytes
//!
//! MessagePack through `serde`, and the two choices behind that are worth
//! saying out loud because an earlier version of this wrote the bytes by hand,
//! 470 lines of them, on the argument that a `#[derive(Serialize)]` would hide
//! the domain decision about [`Value::Opaque`]. That argument was wrong: an
//! `Arc<dyn Any>` cannot be derived at all, so the decision has to be written by
//! hand either way. What the 470 lines actually bought was an unversioned format
//! and an inspector we would also have had to write.
//!
//! **Why MessagePack and not `postcard` or `bincode`**: measured, not guessed.
//! `postcard` throws away the message of a custom serialization error, and here
//! that message is the one explaining why an opaque value cannot travel;
//! `bincode` writes more bytes. Being self-describing also means these bytes can
//! be read with any MessagePack tool the day something goes wrong on a machine
//! you cannot attach a debugger to.
//!
//! An `Opaque` carries something that **only exists in this process**, so it
//! fails on the way out, with the node and the host in front of you. It is asked
//! through [`Value::travels`], so the refusal is [`MessageError::Opaque`] and
//! not whatever a serializer felt like saying. Catching it at compile time
//! cannot be done: which value travels along an edge is a run-time matter.
//!
//! Mind the asymmetry: an [`Artifact`](crate::Artifact)'s bytes **do** cross
//! unlooked-at. An artifact is a pile of bytes opaque by design; an `Opaque` is
//! a pointer into this process disguised as a value.
//!
//! And still no version: both sides are the same binary from the same
//! `cargo build`. The day they stop being so, the place for one is the `Hello`,
//! which already negotiates the client's *runtime*.

use crate::{Label, Outcome};
use serde::{Deserialize, Serialize};
use soma_next_core::{Device, Keys, Memory, NodeId, Placement, Plan, Value};
use std::fmt;

/// What the client says.
///
/// The first `allow` in the project, and it is worth saying why rather than
/// leaving it to be discovered: `Work` is far bigger than `Hello`, and clippy is
/// right about the arithmetic and wrong about what to do. Boxing it would buy an
/// allocation on every message to save a few hundred bytes of stack on one that
/// is about to be serialized anyway — and `Hello` is sent **once per session**
/// while `Work` is the whole conversation.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    /// Opens the session: who I am and what I would provision you with.
    Hello {
        /// How the client identifies itself, so the worker can say no. Opaque
        /// here; the [`Provision`](crate::Provision) reads it.
        runtime: String,
        /// The name of the artifact I bring, if I bring one. `None` means "you
        /// already have your catalog", and neither kind can pretend to be the
        /// other.
        offering: Option<Label>,
    },
    /// Here is the artifact, since you asked for it.
    Provision {
        /// The bytes. Nobody here looks at them.
        bytes: Vec<u8>,
    },
    /// Execute this.
    Work {
        /// What gets executed.
        plan: Plan,
        /// The graph's input.
        input: Value,
        /// What was produced on the client that this plan reads and does not
        /// produce.
        known: Vec<(NodeId, Value)>,
        /// What each of those is called, so what runs here can name what it
        /// produces and the chain of keys does not stop at the wire.
        keys: Vec<(NodeId, Keys)>,
        /// Where each node of this slice runs.
        placement: Placement,
        /// What is remembered about the nodes of this slice: what implements
        /// each, which are settled, which are worth keeping.
        memory: Memory,
    },
}

/// What the worker answers.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Answer {
    /// Ready to work.
    Ready,
    /// I do not have that artifact: send it to me.
    Send,
    /// No, and here is why. It belongs to the session, not to a job: after this
    /// there is no conversation.
    Refused(String),
    /// Done, with what it produced.
    Done(Outcome),
    /// What you sent failed over there. Text on purpose: what is needed here is
    /// for whoever launched the run to **read** what happened.
    Failed(String),
}

impl Request {
    /// This message in bytes. Fails if some value cannot leave this process.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MessageError> {
        if let Request::Work { input, known, .. } = self {
            travelling(std::iter::once(input).chain(known.iter().map(|(_, value)| value)))?;
        }
        write(&Sending::from(self))
    }

    /// And back from them.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MessageError> {
        read::<Received>(bytes).map(Request::from)
    }
}

impl Answer {
    /// This message in bytes. Fails if the slice produced something over there
    /// that cannot come back.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MessageError> {
        if let Answer::Done(outcome) = self {
            travelling(
                std::iter::once(&outcome.last)
                    .chain(outcome.produced.iter().map(|(_, value)| value)),
            )?;
        }
        write(self)
    }

    /// And back from them.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MessageError> {
        read(bytes)
    }
}

/// That every one of these can leave the process.
fn travelling<'v>(values: impl Iterator<Item = &'v Value>) -> Result<(), MessageError> {
    match values.into_iter().all(Value::travels) {
        true => Ok(()),
        false => Err(MessageError::Opaque),
    }
}

fn write<T: Serialize>(what: &T) -> Result<Vec<u8>, MessageError> {
    rmp_serde::to_vec(what).map_err(|e| MessageError::Malformed(e.to_string()))
}

/// Reads one message, and **nothing may be left over**: leftovers are as
/// suspicious as missing bytes, and no format checks that for you.
fn read<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, MessageError> {
    let mut rest = bytes;
    let what: T =
        rmp_serde::from_read(&mut rest).map_err(|e| MessageError::Malformed(e.to_string()))?;
    match rest.len() {
        0 => Ok(what),
        left => Err(MessageError::Malformed(format!(
            "{left} bytes left over at the end"
        ))),
    }
}

/// What a request looks like on the wire.
///
/// A mirror of [`Request`], and not the type itself, because of the fields
/// `serde` cannot decide on its own: the **placement** and the **memory**. Only
/// what belongs to *this plan's* nodes travels — sending the whole thing would
/// put on the wire where nodes that do not even exist there run, and what is
/// remembered of them — and of the placement the **host** half does not travel
/// at all, having already done its job when it decided this slice would leave.
/// `serde` sees one field at a time, so those transformations live here.
///
/// The memory is the one thing here that is **built** rather than borrowed: a
/// projection has to be made somewhere, and it is a handful of short strings
/// against a plan and its values.
///
/// Two mirrors and not one so that sending copies nothing. **Their variants have
/// to stay in the same order**: what goes on the wire is the index.
#[allow(clippy::large_enum_variant)]
#[derive(Serialize)]
enum Sending<'a> {
    Hello {
        runtime: &'a str,
        offering: Option<&'a Label>,
    },
    Provision {
        bytes: &'a [u8],
    },
    Work {
        plan: &'a Plan,
        input: &'a Value,
        known: &'a [(NodeId, Value)],
        keys: &'a [(NodeId, Keys)],
        devices: Vec<(&'a NodeId, &'a Device)>,
        memory: Memory,
    },
}

#[allow(clippy::large_enum_variant)]
#[derive(Deserialize)]
enum Received {
    Hello {
        runtime: String,
        offering: Option<Label>,
    },
    Provision {
        bytes: Vec<u8>,
    },
    Work {
        plan: Plan,
        input: Value,
        known: Vec<(NodeId, Value)>,
        keys: Vec<(NodeId, Keys)>,
        devices: Vec<(NodeId, Device)>,
        memory: Memory,
    },
}

impl<'a> From<&'a Request> for Sending<'a> {
    fn from(request: &'a Request) -> Self {
        match request {
            Request::Hello { runtime, offering } => Sending::Hello {
                runtime,
                offering: offering.as_ref(),
            },
            Request::Provision { bytes } => Sending::Provision { bytes },
            Request::Work {
                plan,
                input,
                known,
                keys,
                placement,
                memory,
            } => Sending::Work {
                plan,
                input,
                known,
                keys,
                devices: devices_in(plan, placement),
                memory: memory_in(plan, memory),
            },
        }
    }
}

impl From<Received> for Request {
    fn from(received: Received) -> Self {
        match received {
            Received::Hello { runtime, offering } => Request::Hello { runtime, offering },
            Received::Provision { bytes } => Request::Provision { bytes },
            Received::Work {
                plan,
                input,
                known,
                keys,
                devices,
                memory,
            } => {
                let mut placement = Placement::new();
                for (id, device) in devices {
                    placement.place(id, device);
                }
                Request::Work {
                    plan,
                    input,
                    known,
                    keys,
                    placement,
                    memory,
                }
            }
        }
    }
}

/// The device of each node of this plan that has one.
fn devices_in<'a>(plan: &'a Plan, placement: &'a Placement) -> Vec<(&'a NodeId, &'a Device)> {
    let mut ids = Vec::new();
    nodes_in(plan, &mut ids);
    ids.into_iter()
        .filter_map(|id| placement.of(id).map(|device| (id, device)))
        .collect()
}

/// What is remembered about each node of this plan, and about no other.
///
/// **Written out one fact at a time, which is a hole with a name on it**: a new
/// thing to remember that is not added here does not fail — it simply stops
/// being true on the other side of the wire, which is the worst way for anything
/// to be wrong. Whoever adds a fifth thing to [`Memory`] adds a line here and a
/// test that crosses.
fn memory_in(plan: &Plan, memory: &Memory) -> Memory {
    let mut ids = Vec::new();
    nodes_in(plan, &mut ids);
    let mut mine = Memory::new();
    for id in ids {
        if let Some(what) = memory.identity_of(id) {
            mine.identify(id.clone(), what);
        }
        if memory.is_frozen(id) {
            mine.freeze(id.clone(), memory.state_of(id).map(str::to_string));
        }
        if memory.is_cached(id) {
            mine.cache(id.clone(), memory.salt_of(id).map(str::to_string));
        }
        if memory.is_mapped(id) {
            mine.map(id.clone());
        }
        if let Some(written) = memory.fingerprint_of(id) {
            mine.written_as(id.clone(), written);
        }
    }
    mine
}

fn nodes_in<'p>(plan: &'p Plan, out: &mut Vec<&'p NodeId>) {
    match plan {
        Plan::Empty => {}
        Plan::Execute { node, .. } => out.push(node),
        Plan::Sequence(plans) | Plan::Wave(plans) => {
            for plan in plans {
                nodes_in(plan, out);
            }
        }
        Plan::Remote { inner, .. } => nodes_in(inner, out),
    }
}

/// Why a message could not be put on the wire, or taken off it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageError {
    /// A [`Value::Opaque`] cannot leave its process.
    Opaque,
    /// These bytes are not the ones that were written: truncated, left over, or
    /// never a message at all.
    Malformed(String),
}

impl fmt::Display for MessageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Opaque => f.write_str(
                "an opaque value does not cross to another process: what it carries only \
                 exists in this one. If it has to travel, take it out of `Opaque` and send \
                 it as data",
            ),
            Self::Malformed(why) => write!(f, "these are not the bytes that were written: {why}"),
        }
    }
}

impl std::error::Error for MessageError {}
