//! What happened while a plan ran: the vocabulary of level 1, the engine's.
//!
//! A fact and not a judgement. Whether 400 ms is slow or a gradient is dying is
//! an opinion about the record, and the invariant is that the opinion has to be
//! reproducible from the record without running again.
//!
//! An enum because the set is closed and the engine knows it. What the original
//! got wrong was not the number of variants but putting three vocabularies in
//! one — a fact beside an opinion about facts. Here each level keeps its own:
//! this is the engine's, a training run's is Python's, a study's is a record on
//! disk. **They do not meet in Rust, they meet in the record**, and
//! [`Fact::flattened`] is that meeting: a name and text-to-text pairs.
//!
//! Every measurement is a [`Duration`] and never an instant. A duration from
//! another machine is worth reading; two wall clocks disagree. **When** it was
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
        /// How long after this run started it began. An offset into a slice
        /// is a fact about the slice, so one that ran elsewhere counts from its
        /// own start and a timeline adds the [`Fact::Left`] it arrived under.
        began: Duration,
        /// How long its `forward` took. Whatever it did in there is inside
        /// that number: the engine does not look inside a node.
        took: Duration,
        /// Where it was told to run, if it was told.
        device: Option<Device>,
    },
    /// A node was advanced and did not answer. Emitted **before** the run
    /// stops, so a watcher learns which node while it is happening.
    Failed {
        /// Which one.
        node: NodeId,
        /// What it said.
        why: String,
    },
    /// A node was not run because nobody needed what it makes.
    ///
    /// A fact and not an absence: a node missing from a record cannot be told
    /// from one that was never in the graph.
    Spared {
        /// Which one.
        node: NodeId,
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
    /// A node that maps over its items, item by item — it runs the new ones
    /// and reads the rest back, so one number would not say what happened.
    Items {
        /// Which one.
        node: NodeId,
        /// How many items it was given.
        of: usize,
        /// How many of them did not have to be computed.
        recalled: usize,
    },
    /// A slice of the plan crossed to another machine, and came back. `took`
    /// is the whole round trip, which is not the sum of what happened there.
    Left {
        /// Whose machine.
        host: Host,
        /// How long after this run started it left.
        began: Duration,
        /// How long the round trip took.
        took: Duration,
    },
    /// And this is what happened over there. Recursive, so a slice that
    /// carried on to a third host still says where each thing happened and
    /// nothing that travelled is rewritten; flattening turns it into a `host`.
    Elsewhere {
        /// Whose machine.
        host: Host,
        /// What it saw there.
        saw: Box<Fact>,
    },
    /// A level that is **not** the engine had something to say, already flat.
    ///
    /// The carrier and not the vocabulary: the core does not learn what a load
    /// average is, only that other levels exist and one may be speaking from
    /// another machine. Not for level 2, whose loss is computed where the
    /// notebook is and goes straight into the record.
    Said {
        /// What kind of thing it is, which is what it will be written down as.
        kind: String,
        /// And its fields, text to text, already in the written form.
        pairs: Vec<(String, String)>,
    },
    /// The whole thing is over. Emitted by [`Executor::run`](crate::Executor::run)
    /// and not by [`resume`](crate::Executor::resume): a slice is not a
    /// `forward`. It is what tells a writer where one record ends.
    Finished {
        /// How long all of it took.
        took: Duration,
    },
    /// ...or it is over because of this: the other terminal fact, so a record
    /// is closed either way. A node that failed said so as [`Fact::Failed`].
    Broke {
        /// What stopped it.
        why: String,
    },
}

impl Fact {
    /// This fact as a name and text-to-text fields: **how it is written down**,
    /// which is not how it is emitted.
    ///
    /// [`Fact::Elsewhere`] does not survive as a name — it becomes a `host`
    /// field on whatever it wrapped, so a reader gets columns and not a tree.
    pub fn flattened(&self) -> (&str, Vec<(String, String)>) {
        match self {
            Self::Ran {
                node,
                began,
                took,
                device,
            } => {
                let mut said = vec![
                    ("node".into(), node.to_string()),
                    began_us(began),
                    took_us(took),
                ];
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
            Self::Spared { node } => ("spared", vec![("node".into(), node.to_string())]),
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
            Self::Left { host, began, took } => (
                "left",
                vec![
                    ("host".into(), host.to_string()),
                    began_us(began),
                    took_us(took),
                ],
            ),
            // Already flat, and reshaping it here would be this crate deciding
            // something about a vocabulary it does not know.
            Self::Said { kind, pairs } => (kind.as_str(), pairs.clone()),
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

    /// Whether this fact ends a run, whichever way. Asked by whoever writes
    /// records so it does not have to know the vocabulary.
    pub fn ends_a_run(&self) -> bool {
        matches!(self, Self::Finished { .. } | Self::Broke { .. })
    }
}

/// A duration as whole microseconds — an integer, because this is text somebody
/// reads with `cat` and something else parses.
fn took_us(took: &Duration) -> (String, String) {
    ("took_us".into(), took.as_micros().to_string())
}

/// And where it sat on the run's own timeline, which is what makes a picture of
/// *what ran when* possible at all.
fn began_us(began: &Duration) -> (String, String) {
    ("began_us".into(), began.as_micros().to_string())
}
