//! What is remembered about each node.
//!
//! The fifth fact, and like the other four it is a type of its own because it
//! answers a question of its own:
//!
//! | piece | answers |
//! |---|---|
//! | [`Graph`] | **what** exists and how it connects |
//! | [`Catalog`](crate::Catalog) | **who** executes it |
//! | [`Placement`](crate::Placement) | **where** |
//! | [`Plan`](crate::Plan) | **when**, and with what concurrency |
//! | `Memory` | **what is remembered** |
//!
//! Four maps, independent of each other for the same reason `Placement` has two:
//! a node can be frozen without being cached, named without being frozen, and
//! any combination of the rest.
//!
//! # What goes in a key and what does not
//!
//! ```text
//! key(root) = H(content)                          ← the only place data is hashed
//! key(node) = H(identity, state, keys of its predecessors)
//! ```
//!
//! The **identity** is the name of what implements the node — the class, in
//! Python — and it is in the key because without it two different nodes called
//! `embed` collide in a shared store. The **fingerprint of the code** is *not*:
//! a cosmetic refactor would invalidate half the store in silence. It is kept
//! beside the value and compared on a hit, which turns the same event into a
//! line on `stderr` instead of a cache that quietly went cold.
//!
//! # Frozen, which the core defines without knowing what a gradient is
//!
//! **A frozen node's state does not change while the graph runs.** That is a
//! statement about cache validity, and the core can hold it the same way it
//! holds a [`Device`](crate::Device): as inert information it reasons over —
//! here, [`cacheable`] — and somebody else obeys. `soma_next.torch` is what
//! turns it true, with `requires_grad_(False)`, exactly as the node and not the
//! core is what moves a tensor to a GPU.
//!
//! And it is why the digest of the state is given **to** `freeze`: settling is
//! what makes both things true at once, so it is the one moment worth paying to
//! hash the weights at.

use crate::{Graph, NodeId};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// What is remembered about each node. The ones not listed have nothing said
/// about them, which is the same as nothing being kept.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Memory {
    /// The ones whose state does not change while the graph runs, each with the
    /// digest of the state it is settled at — `None` for one with no state to
    /// settle, like a tokenizer.
    frozen: HashMap<NodeId, Option<String>>,
    /// The ones whose output is worth keeping, each with the salt its declarer
    /// added, if any.
    cached: HashMap<NodeId, Option<String>>,
    /// What implements each one, by name.
    identities: HashMap<NodeId, String>,
    /// Which version of that code the graph was written against. Metadata.
    fingerprints: HashMap<NodeId, String>,
}

impl Memory {
    /// Nothing remembered about anything.
    pub fn new() -> Self {
        Self::default()
    }

    /// Says what implements this node, returning what it was called before.
    pub fn identify(&mut self, id: impl Into<NodeId>, what: impl Into<String>) -> Option<String> {
        self.identities.insert(id.into(), what.into())
    }

    /// What implements it, if it was said.
    pub fn identity_of(&self, id: &NodeId) -> Option<&str> {
        self.identities.get(id).map(String::as_str)
    }

    /// Says this node's state does not change from here on, and the digest of
    /// the state it is settled at — `None` when there is no state to settle.
    ///
    /// Called twice on purpose: declaring `.frozen()` says it with no digest,
    /// and whoever knows how to hash the weights says it again with one.
    pub fn freeze(&mut self, id: impl Into<NodeId>, state: Option<String>) {
        self.frozen.insert(id.into(), state);
    }

    /// Whether this node's state was said not to change.
    pub fn is_frozen(&self, id: &NodeId) -> bool {
        self.frozen.contains_key(id)
    }

    /// The digest of the state it is frozen at, if it is frozen and has one.
    pub fn state_of(&self, id: &NodeId) -> Option<&str> {
        self.frozen.get(id)?.as_deref()
    }

    /// Says this node's output is worth keeping, with the caller's salt if they
    /// gave one — `.cached(salt="a100-fp16")` is how you tell apart two runs the
    /// key cannot tell apart on its own.
    pub fn cache(&mut self, id: impl Into<NodeId>, salt: Option<String>) {
        self.cached.insert(id.into(), salt);
    }

    /// Whether this node's output is kept. A node that is not is **not** a break
    /// in the chain: its key is still computed and passed on, it is just not
    /// stored.
    pub fn is_cached(&self, id: &NodeId) -> bool {
        self.cached.contains_key(id)
    }

    /// The salt it is cached under, if it is cached and has one.
    pub fn salt_of(&self, id: &NodeId) -> Option<&str> {
        self.cached.get(id)?.as_deref()
    }

    /// Notes which version of the code this graph was written against.
    /// **Metadata**: never in a key, only compared on a hit.
    pub fn written_as(
        &mut self,
        id: impl Into<NodeId>,
        fingerprint: impl Into<String>,
    ) -> Option<String> {
        self.fingerprints.insert(id.into(), fingerprint.into())
    }

    /// Which version of the code it was written against, if it was noted.
    pub fn fingerprint_of(&self, id: &NodeId) -> Option<&str> {
        self.fingerprints.get(id).map(String::as_str)
    }

    /// How many nodes have anything said about them at all.
    pub fn len(&self) -> usize {
        self.frozen
            .keys()
            .chain(self.cached.keys())
            .chain(self.identities.keys())
            .chain(self.fingerprints.keys())
            .collect::<HashSet<_>>()
            .len()
    }

    /// Whether nothing has been said about any node, which is what lets the
    /// engine skip all of this.
    pub fn is_empty(&self) -> bool {
        self.frozen.is_empty()
            && self.cached.is_empty()
            && self.identities.is_empty()
            && self.fingerprints.is_empty()
    }
}

/// Whether what this graph says to keep can honestly be kept.
///
/// A free function and not a method for the same reason
/// [`compile`](crate::compile) is one: it needs the graph **and** the table, so
/// it was never a method of either.
///
/// The rule is one line and it is a rule about **prefixes**:
///
/// > a node's output can be kept if nothing upstream of it can change — itself
/// > included.
///
/// Freezing the node alone is not enough. In torch's terms, freezing layer 3 of
/// 5 does not stop the gradient crossing it towards layers 1 and 2, and the
/// value that would be restored from the store is a **leaf**: the backward pass
/// would stop there and everything above it would quietly stop training. The
/// same rule falls out a second way, without mentioning gradients at all — the
/// digest of the state is in the key, so a node that keeps changing gets a new
/// key every run and never hits, only fills the store.
///
/// Being named is checked in the same walk, because a node with no identity has
/// no key, and a chain with a hole in it delivers no key to what is below.
pub fn cacheable(graph: &Graph, memory: &Memory) -> Result<(), MemoryError> {
    for id in graph.nodes() {
        if !memory.is_cached(id) {
            continue;
        }
        for above in upstream(graph, id) {
            if !memory.is_frozen(&above) {
                return Err(MemoryError::Unsettled {
                    cached: id.clone(),
                    moving: above,
                });
            }
            if memory.identity_of(&above).is_none() {
                return Err(MemoryError::Nameless {
                    cached: id.clone(),
                    nameless: above,
                });
            }
        }
    }
    Ok(())
}

/// This node and everything it reads, transitively, **nearest first** — so what
/// an error names is the closest thing to the problem and not the furthest.
fn upstream(graph: &Graph, id: &NodeId) -> Vec<NodeId> {
    let mut seen: HashSet<&NodeId> = HashSet::from([id]);
    let mut out = vec![id.clone()];
    let mut next = 0;
    while next < out.len() {
        for above in graph.predecessors(&out[next]) {
            if seen.insert(above) {
                out.push(above.clone());
            }
        }
        next += 1;
    }
    out
}

/// Why what this graph says to keep could not honestly be kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    /// Something a cached node depends on can still change.
    Unsettled {
        /// The one that says to keep its output.
        cached: NodeId,
        /// The one that can still change. The same node when it is itself.
        moving: NodeId,
    },
    /// Something a cached node depends on has no identity, so no key.
    Nameless {
        /// The one that says to keep its output.
        cached: NodeId,
        /// The one nobody said what it is.
        nameless: NodeId,
    },
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsettled { cached, moving } if cached == moving => write!(
                f,
                "`{cached}` keeps its output and is not frozen: what is worth keeping \
                 is what does not change, and a node that still trains gets a new key \
                 every run"
            ),
            Self::Unsettled { cached, moving } => write!(
                f,
                "`{cached}` keeps its output and `{moving}`, which it reads, is not \
                 frozen: an output is only reusable if nothing above it can change"
            ),
            Self::Nameless { cached, nameless } if cached == nameless => write!(
                f,
                "`{cached}` keeps its output and nobody said what implements it, so \
                 there is nothing to build its key out of"
            ),
            Self::Nameless { cached, nameless } => write!(
                f,
                "`{cached}` keeps its output and nobody said what implements `{nameless}`, \
                 which it reads: a chain of keys with a hole in it reaches nothing below"
            ),
        }
    }
}

impl std::error::Error for MemoryError {}
