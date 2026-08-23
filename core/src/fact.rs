//! What happened while a plan ran: the vocabulary of level 1, which is the
//! engine's.
//!
//! A **fact and not a judgement**. Whether 400 ms is slow, whether a gradient is
//! dying, whether a hit rate is bad — none of that is here, and that is the
//! split CU19 drew: the record is what happened, the diagnosis is an opinion
//! about it, and the invariant is that the opinion has to be reproducible from
//! the record without running again.
//!
//! # Why an enum, and why it may grow large
//!
//! Because the set is closed and the engine knows it: there are only so many
//! things a walk can see, and when one is added the compiler finds every
//! `match`. What the original got wrong was not the **number** of variants — it
//! was putting three vocabularies in one, so `NodeStarted` (a fact) sat beside
//! `HealthFlag` (an opinion) beside seven variants of a layer that no longer
//! exists.
//!
//! Here each level keeps its own vocabulary in its own language: this is the
//! engine's, the trainer's is Python's, and the study's has been a record on
//! disk since CU18. **They do not meet in Rust — they meet in the record.**
//!
//! # Emitted as an enum, written down as pairs
//!
//! [`Fact::flattened`] is the whole of that meeting: a name and text-to-text
//! fields, which is the shape [`Meta`] already has in the store. What is typed
//! stays typed where the compiler can help, and what crosses to another
//! vocabulary crosses as the flattest thing there is.
//!
//! [`Meta`]: https://docs.rs/soma-next-store
//!
//! # Durations and not instants
//!
//! Every measurement here is a [`Duration`], deliberately. A fact from another
//! machine is worth reading — a node took 12 ms over there — while a wall clock
//! from another machine is two clocks that disagree, which is the problem CU18
//! already solved by comparing writers with writers. **When** something was
//! written down is the store's business, and it stamps it.

use crate::{Device, Host, Key, NodeId};
use std::time::Duration;

/// One thing the engine saw.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Fact {
    /// A node was advanced, and answered.
    Ran {
        /// Which one.
        node: NodeId,
        /// How long its `forward` took, and nothing else: whatever it did in
        /// there — a retry, three rounds of something — is inside that number
        /// because the engine does not look inside a node.
        took: Duration,
        /// Where it was told to run, if it was told.
        device: Option<Device>,
    },
    /// A node was advanced and did not answer.
    ///
    /// Emitted **before** the run stops, which is the point: without it the
    /// only thing left is a `RunError` at the top, and which node it was has to
    /// be read out of a message.
    Failed {
        /// Which one.
        node: NodeId,
        /// What it said.
        why: String,
    },
    /// A node was not advanced at all: what it would have produced was already
    /// kept under that name.
    Recalled {
        /// Which one.
        node: NodeId,
        /// The name it was found under.
        key: Key,
    },
    /// A node ran and what it produced was written down.
    Kept {
        /// Which one.
        node: NodeId,
        /// The name it was written under.
        key: Key,
    },
    /// A node that maps over its items, item by item.
    ///
    /// The grain that CU16 separated: a node named once per item runs for the
    /// ones that are new and reads the rest back, so one number is not enough to
    /// say what happened.
    Items {
        /// Which one.
        node: NodeId,
        /// How many items it was given.
        of: usize,
        /// How many of them did not have to be computed.
        recalled: usize,
    },
    /// A slice of the plan crossed to another machine, and came back.
    ///
    /// `took` is the whole round trip, wire included — which is the number that
    /// answers whether sending it was worth it, and it is not the sum of what
    /// happened over there.
    Left {
        /// Whose machine.
        host: Host,
        /// How long the round trip took.
        took: Duration,
    },
    /// And this is what happened over there.
    ///
    /// Recursive so that a slice which carries on to a third host still says
    /// where each thing happened, and so that **nothing that travelled has to be
    /// rewritten**: whoever relays wraps, and the engine over there emits
    /// exactly what it would emit at home. Flattening turns the nesting into a
    /// `host` field, so the written form is flat.
    Elsewhere {
        /// Whose machine.
        host: Host,
        /// What it saw there.
        saw: Box<Fact>,
    },
    /// The whole thing is over.
    ///
    /// Emitted by [`Executor::run`](crate::Executor::run) and **not** by
    /// [`resume`](crate::Executor::resume): a `forward` is a run, and a slice
    /// executed for somebody else is not one. It is what tells whoever is
    /// writing where one record ends and the next begins.
    Finished {
        /// How long all of it took.
        took: Duration,
    },
    /// ...or it is over because of this.
    ///
    /// The other terminal fact, so a record is closed either way. What went
    /// wrong at the level of the run: a host nobody could reach, a value that
    /// could not travel. A node that failed said so as [`Fact::Failed`] first.
    Broke {
        /// What stopped it.
        why: String,
    },
}

impl Fact {
    /// This fact as a name and text-to-text fields: **how it is written down**,
    /// which is not how it is emitted.
    ///
    /// The one place the three vocabularies can meet, and the reason the core
    /// never learns what a loss is: level 2 produces this shape directly from
    /// Python and lands in the same record, beside these.
    ///
    /// [`Fact::Elsewhere`] does not survive as a name — it becomes a `host`
    /// field on whatever it wrapped, so a reader has columns and not a tree.
    pub fn flattened(&self) -> (&'static str, Vec<(String, String)>) {
        match self {
            Self::Ran { node, took, device } => {
                let mut said = vec![("node".into(), node.to_string()), took_us(took)];
                if let Some(device) = device {
                    said.push(("device".into(), device.to_string()));
                }
                ("ran", said)
            }
            Self::Failed { node, why } => (
                "failed",
                vec![
                    ("node".into(), node.to_string()),
                    ("why".into(), why.clone()),
                ],
            ),
            Self::Recalled { node, key } => (
                "recalled",
                vec![
                    ("node".into(), node.to_string()),
                    ("key".into(), key.to_string()),
                ],
            ),
            Self::Kept { node, key } => (
                "kept",
                vec![
                    ("node".into(), node.to_string()),
                    ("key".into(), key.to_string()),
                ],
            ),
            Self::Items { node, of, recalled } => (
                "items",
                vec![
                    ("node".into(), node.to_string()),
                    ("of".into(), of.to_string()),
                    ("recalled".into(), recalled.to_string()),
                ],
            ),
            Self::Left { host, took } => (
                "left",
                vec![("host".into(), host.to_string()), took_us(took)],
            ),
            Self::Elsewhere { host, saw } => {
                let (kind, mut said) = saw.flattened();
                // Last, so that a fact which crossed two machines keeps the
                // nearest host last and the reader sees the route in order.
                said.push(("host".into(), host.to_string()));
                (kind, said)
            }
            Self::Finished { took } => ("finished", vec![took_us(took)]),
            Self::Broke { why } => ("broke", vec![("why".into(), why.clone())]),
        }
    }

    /// Whether this fact ends a run, whichever way it ended.
    ///
    /// Asked by whoever writes records so it does not have to know the
    /// vocabulary: a variant added later that also ends one is added here and
    /// every writer follows.
    pub fn ends_a_run(&self) -> bool {
        matches!(self, Self::Finished { .. } | Self::Broke { .. })
    }
}

/// A duration as whole microseconds. An integer and not a float because this is
/// text that somebody reads with `cat` and something else parses, and neither
/// should have to think about how many decimals were written.
fn took_us(took: &Duration) -> (String, String) {
    ("took_us".into(), took.as_micros().to_string())
}
