//! Where each node runs.
//!
//! A type of its own, and not a field of [`Graph`](crate::Graph) or
//! [`Catalog`](crate::Catalog), because placing is a fourth fact:
//!
//! | piece | answers |
//! |---|---|
//! | [`Graph`](crate::Graph) | **what** exists and how it connects |
//! | [`Catalog`](crate::Catalog) | **who** executes it |
//! | `Placement` | **where** |
//! | [`Plan`](crate::Plan) | **when**, and with what concurrency |
//!
//! It does not fit in the graph — a `Graph` is topology only, and the engine
//! does not look at it — nor in the catalog, which is the half that is **not**
//! data: the day a subgraph travels to another machine, the placement travels
//! with it and the implementations do not.
//!
//! # The two halves, and who obeys each
//!
//! | half | who reads it | when |
//! |---|---|---|
//! | [`Host`] | [`distribute`](crate::distribute) | when deciding the shape |
//! | [`Device`] | the node, via `ctx.device` | when executing it |
//!
//! The node obeys the [`Device`] because the core does not know how to move
//! anything to a GPU; it cannot obey the [`Host`], because its code does not run
//! here. Hence two maps and not a pair: a node can have a host without a device,
//! a device without a host, both, or neither.
//!
//! And hence [`compile`](crate::compile) sees neither. A device is **inert** for
//! the traversal, so placing cannot alter the plan. A host is not, but crossing
//! a wire is decided in a separate, named step.
//!
//! A bare map is what it is, without checking that the ids exist: that is
//! checked where there is a graph in front of you — [`Wire::on`](crate::Wire::on)
//! and [`Wire::at`](crate::Wire::at) can only name their own nodes, and the
//! bindings' `place()` validates against the graph.

use crate::{Device, Host, NodeId};
use std::collections::{HashMap, HashSet};

/// Where each node runs. The ones not listed run wherever they land.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Placement {
    devices: HashMap<NodeId, Device>,
    hosts: HashMap<NodeId, Host>,
}

impl Placement {
    /// Nothing placed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Places a node on a device, returning where it was before.
    pub fn place(&mut self, id: impl Into<NodeId>, device: Device) -> Option<Device> {
        self.devices.insert(id.into(), device)
    }

    /// Sends a node to a host, returning which one it was on before.
    /// Independent of [`place`](Self::place).
    pub fn place_at(&mut self, id: impl Into<NodeId>, host: Host) -> Option<Host> {
        self.hosts.insert(id.into(), host)
    }

    /// Which device this node runs on, if it was said. `None` is "wherever it
    /// already is", not `cpu`.
    pub fn of(&self, id: &NodeId) -> Option<&Device> {
        self.devices.get(id)
    }

    /// Which host this node runs on, if it was said. `None` is here.
    pub fn host_of(&self, id: &NodeId) -> Option<&Host> {
        self.hosts.get(id)
    }

    /// Every host this placement names, once each, in a fixed order.
    ///
    /// The half of [`host_of`](Self::host_of) that reads the other way, and it
    /// exists because of who asks. A client that was handed a dictionary of
    /// workers already knew the names — they were its keys. A client that talks
    /// to a broker does not: it has to ask *which hosts does this graph name*,
    /// and then ask the broker for each. The names live here and nowhere else.
    ///
    /// **Once each**, because a host named by ten nodes is one rendezvous, not
    /// ten. And **sorted**, which is not tidiness: these come out of a `HashMap`,
    /// and iterating one gives a different order every run. That would make the
    /// order rendezvous are asked for — and so the order failures happen in —
    /// irreproducible, and this project has already paid for a nondeterministic
    /// order once, when an artifact's id changed because the caller reordered a
    /// dictionary.
    ///
    /// A `Vec` and not an `impl Iterator`: deduplicating means collecting, so
    /// the work is already done and pretending it is lazy would only be a
    /// costume.
    pub fn hosts(&self) -> Vec<&Host> {
        let mut named: Vec<&Host> = self.hosts.values().collect();
        named.sort();
        named.dedup();
        named
    }

    /// How many nodes have something said about them: device, host or both.
    pub fn len(&self) -> usize {
        self.devices
            .keys()
            .chain(self.hosts.keys())
            .collect::<HashSet<_>>()
            .len()
    }

    /// Whether nothing has been said about any node.
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty() && self.hosts.is_empty()
    }

    /// Whether no node has been sent to any host, which is what allows skipping
    /// [`distribute`](crate::distribute).
    pub fn is_local(&self) -> bool {
        self.hosts.is_empty()
    }
}
