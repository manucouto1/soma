//! Declaring a graph as an expression, instead of by calls.
//!
//! ```ignore
//! let (graph, catalog, placement, memory) = (node("source", Add(1.0))
//!     >> (node("left", Add(10.0)).on(Device::Cuda(0)) | node("right", Add(100.0)))
//!     >> node("join", Mean))
//! .somatize()?;
//! ```
//!
//! `>>` chains and `|` opens branches, the same syntax as the Python DSL. In
//! Rust it falls out of implementing [`std::ops::Shr`] and
//! [`std::ops::BitOr`] on a type of our own; no macro needed.
//!
//! A [`Wire`] is a half-declared graph: where you enter (`heads`) and where you
//! leave (`terminals`), which is all it takes to glue another one on. Nothing is
//! materialized until [`Wire::somatize`], so joining two pieces concatenates
//! lists rather than merging graphs.

use crate::{Catalog, Device, Graph, GraphError, Host, Memory, Node, NodeId, Placement};
use std::ops::{BitOr, Shr};
use std::sync::Arc;

/// A half-declared graph.
pub struct Wire {
    parts: Result<Parts, GraphError>,
}

struct Parts {
    nodes: Vec<(NodeId, Arc<dyn Node>)>,
    edges: Vec<(NodeId, NodeId)>,
    heads: Vec<NodeId>,
    terminals: Vec<NodeId>,
    /// The ones that already have a device. An id appears at most once.
    devices: Vec<(NodeId, Device)>,
    /// The ones that already have a host. Separate from the devices so that
    /// `.on(...)` does not shadow an inner `.at(...)`, or the other way round.
    hosts: Vec<(NodeId, Host)>,
    /// What implements each one. Filled where the concrete type is still known,
    /// which is [`node`] and nowhere else: from there on it is an `Arc<dyn
    /// Node>` and the name is gone.
    identities: Vec<(NodeId, String)>,
    /// The ones settled, each with the digest of the state they are settled at —
    /// never one here, because hashing weights is torch's job and this is the
    /// core.
    frozen: Vec<(NodeId, Option<String>)>,
    /// The ones worth keeping, each with its salt — likewise never one here:
    /// telling apart two runs the key cannot is a knob for whoever runs them,
    /// and it is [`Memory::cache`] for anyone who wants it.
    cached: Vec<(NodeId, Option<String>)>,
}

/// A lone node.
///
/// Named after its type, because this is the last place that knows it: what a
/// node is called is half of the key its output is kept under, and one line
/// later there is only an `Arc<dyn Node>`. Python does the same thing with the
/// class name.
pub fn node<N: Node + 'static>(id: impl Into<NodeId>, implementation: N) -> Wire {
    single(
        id.into(),
        std::any::type_name::<N>(),
        Arc::new(implementation),
    )
}

fn single(id: NodeId, identity: &str, implementation: Arc<dyn Node>) -> Wire {
    Wire {
        parts: Ok(Parts {
            nodes: vec![(id.clone(), implementation)],
            edges: Vec::new(),
            heads: vec![id.clone()],
            terminals: vec![id.clone()],
            devices: Vec::new(),
            hosts: Vec::new(),
            identities: vec![(id, identity.to_string())],
            frozen: Vec::new(),
            cached: Vec::new(),
        }),
    }
}

impl Shr for Wire {
    type Output = Wire;

    /// `a >> b`: everything leaving `a` enters everything starting `b`.
    fn shr(self, next: Wire) -> Wire {
        combine(self, next, |left, right| Parts {
            edges: left
                .terminals
                .iter()
                .flat_map(|from| right.heads.iter().map(|to| (from.clone(), to.clone())))
                .chain(left.edges)
                .chain(right.edges)
                .collect(),
            nodes: left.nodes.into_iter().chain(right.nodes).collect(),
            heads: left.heads,
            terminals: right.terminals,
            devices: left.devices.into_iter().chain(right.devices).collect(),
            hosts: left.hosts.into_iter().chain(right.hosts).collect(),
            identities: left
                .identities
                .into_iter()
                .chain(right.identities)
                .collect(),
            frozen: left.frozen.into_iter().chain(right.frozen).collect(),
            cached: left.cached.into_iter().chain(right.cached).collect(),
        })
    }
}

impl BitOr for Wire {
    type Output = Wire;

    /// `a | b`: two branches that do not touch. Whatever comes in reaches both,
    /// and whatever leaves either one leaves here.
    fn bitor(self, other: Wire) -> Wire {
        combine(self, other, |left, right| Parts {
            nodes: left.nodes.into_iter().chain(right.nodes).collect(),
            edges: left.edges.into_iter().chain(right.edges).collect(),
            heads: left.heads.into_iter().chain(right.heads).collect(),
            terminals: left.terminals.into_iter().chain(right.terminals).collect(),
            devices: left.devices.into_iter().chain(right.devices).collect(),
            hosts: left.hosts.into_iter().chain(right.hosts).collect(),
            identities: left
                .identities
                .into_iter()
                .chain(right.identities)
                .collect(),
            frozen: left.frozen.into_iter().chain(right.frozen).collect(),
            cached: left.cached.into_iter().chain(right.cached).collect(),
        })
    }
}

/// Joins two pieces, keeping the first failure if either carries one.
fn combine(left: Wire, right: Wire, join: impl FnOnce(Parts, Parts) -> Parts) -> Wire {
    Wire {
        parts: match (left.parts, right.parts) {
            (Ok(left), Ok(right)) => Ok(join(left, right)),
            (Err(e), _) | (_, Err(e)) => Err(e),
        },
    }
}

/// Gives `what` to the nodes that did not already have something of that half:
/// the "innermost one wins" rule, written once for both.
fn fill<T: Clone>(nodes: &[(NodeId, Arc<dyn Node>)], placed: &mut Vec<(NodeId, T)>, what: T) {
    let unplaced: Vec<NodeId> = nodes
        .iter()
        .map(|(id, _)| id)
        .filter(|id| !placed.iter().any(|(already, _)| already == *id))
        .cloned()
        .collect();
    placed.extend(unplaced.into_iter().map(|id| (id, what.clone())));
}

impl Wire {
    /// This whole piece on one device. The innermost one wins:
    /// `(a.on(Cuda(0)) >> b).on(Cuda(1))` leaves `a` on 0 and `b` on 1.
    pub fn on(self, device: Device) -> Wire {
        Wire {
            parts: self.parts.map(|mut parts| {
                fill(&parts.nodes, &mut parts.devices, device);
                parts
            }),
        }
    }

    /// This whole piece on one host, likewise, and **independent** of the
    /// device: the two can be written in any order.
    pub fn at(self, host: impl Into<Host>) -> Wire {
        let host = host.into();
        Wire {
            parts: self.parts.map(|mut parts| {
                fill(&parts.nodes, &mut parts.hosts, host);
                parts
            }),
        }
    }

    /// This whole piece settled: its state does not change while the graph
    /// runs. The innermost one wins, like the rest.
    ///
    /// Only half of it, and the half the core can hold: whoever knows what a
    /// gradient is has to make it true, and says it again with the digest of the
    /// state it settled at. See [`Memory::freeze`].
    pub fn frozen(self) -> Wire {
        Wire {
            parts: self.parts.map(|mut parts| {
                fill(&parts.nodes, &mut parts.frozen, None);
                parts
            }),
        }
    }

    /// This whole piece worth keeping: what each of its nodes produces is looked
    /// up before being computed, and kept after.
    ///
    /// Declaring it does not make it honest — that is
    /// [`cacheable`](crate::cacheable)'s question, and it is asked before
    /// running, not here.
    pub fn cached(self) -> Wire {
        Wire {
            parts: self.parts.map(|mut parts| {
                fill(&parts.nodes, &mut parts.cached, None);
                parts
            }),
        }
    }

    /// Materializes what was declared: the structure, the store, the placement
    /// and what is remembered, none containing the others. Fails on a repeated
    /// id, above all.
    pub fn somatize(self) -> Result<(Graph, Catalog, Placement, Memory), GraphError> {
        let parts = self.parts?;
        let mut graph = Graph::new();
        let mut catalog = Catalog::new();
        let mut placement = Placement::new();
        let mut memory = Memory::new();

        for (id, implementation) in parts.nodes {
            graph.add_node(id.clone())?;
            catalog.insert(id, implementation);
        }
        for (from, to) in parts.edges {
            graph.add_edge(from, to)?;
        }
        for (id, device) in parts.devices {
            placement.place(id, device);
        }
        for (id, host) in parts.hosts {
            placement.place_at(id, host);
        }
        for (id, what) in parts.identities {
            memory.identify(id, what);
        }
        for (id, state) in parts.frozen {
            memory.freeze(id, state);
        }
        for (id, salt) in parts.cached {
            memory.cache(id, salt);
        }
        Ok((graph, catalog, placement, memory))
    }
}
