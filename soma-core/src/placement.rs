//! Where each node runs: the fourth fact, beside [`Graph`](crate::Graph) (what
//! exists), [`Catalog`](crate::Catalog) (who executes it) and
//! [`Plan`](crate::Plan) (when).
//!
//! It does not fit in the graph — topology only — nor in the catalog, which is
//! the half that is **not** data: when a subgraph travels, the placement travels
//! with it and the implementations do not.
//!
//! Two maps and not a pair, because the two halves are obeyed by different
//! people: [`distribute`](crate::distribute) reads the [`Host`] when deciding
//! the shape, and the node reads the [`Device`] through `ctx.device` when
//! executing. A node can have either, both or neither. Hence
//! [`compile`](crate::compile) sees neither: a device is inert for the traversal,
//! and crossing a wire is a separate, named step.
//!
//! A bare map, without checking that the ids exist: that is checked where there
//! is a graph in front of you.

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
    /// The half of [`host_of`](Self::host_of) that reads the other way, for the
    /// client that talks to a broker and so does not already know the names.
    /// Once each, because a host named by ten nodes is one rendezvous. **Sorted**,
    /// because these come out of a `HashMap` and an irreproducible order would
    /// make the order failures happen in irreproducible too.
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
