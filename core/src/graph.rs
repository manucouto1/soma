//! What is connected to what.
//!
//! A core `Graph` is **topology only**: identities and edges. What a node does
//! is none of its business, because creating a graph does not need to know. That
//! map (id → implementation) lives with whoever has implementations to store.
//! It is the reason the core depends on nothing.

use std::collections::HashSet;
use std::fmt;

/// A node's name inside a graph. Its own type so no other kind of id gets
/// through.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
pub struct NodeId(String);

impl NodeId {
    /// The id as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for NodeId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for NodeId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A directed connection between two nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    /// Where it leaves from.
    pub source: NodeId,
    /// Where it arrives.
    pub target: NodeId,
}

/// A directed acyclic graph of named nodes.
///
/// The invariant — unique ids, edges between nodes that exist, no cycles — is
/// upheld by the constructors, so `topological_sort` cannot fail. Adjacency is
/// computed on the fly: O(n) where it could be O(1), deliberately.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Graph {
    nodes: Vec<NodeId>,
    edges: Vec<Edge>,
}

impl Graph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node, unless the id is taken.
    pub fn add_node(&mut self, id: impl Into<NodeId>) -> Result<&NodeId, GraphError> {
        let id = id.into();
        if self.contains(&id) {
            return Err(GraphError::DuplicateNode(id));
        }
        self.nodes.push(id);
        Ok(self.nodes.last().expect("just inserted it"))
    }

    /// Connects two nodes that exist, unless the edge is already there or would
    /// close a cycle.
    pub fn add_edge(
        &mut self,
        source: impl Into<NodeId>,
        target: impl Into<NodeId>,
    ) -> Result<&Edge, GraphError> {
        let (source, target) = (source.into(), target.into());
        for end in [&source, &target] {
            if !self.contains(end) {
                return Err(GraphError::UnknownNode(end.clone()));
            }
        }
        if self
            .edges
            .iter()
            .any(|e| e.source == source && e.target == target)
        {
            return Err(GraphError::DuplicateEdge {
                from: source,
                to: target,
            });
        }
        if source == target || self.reaches(&target, &source) {
            return Err(GraphError::WouldCycle {
                from: source,
                to: target,
            });
        }
        self.edges.push(Edge { source, target });
        Ok(self.edges.last().expect("just inserted it"))
    }

    /// A free id starting from the one you want, suffixing `_2`, `_3`, … if needed.
    pub fn free_id(&self, wanted: &str) -> NodeId {
        let mut candidate = NodeId::from(wanted);
        let mut n = 1;
        while self.contains(&candidate) {
            n += 1;
            candidate = NodeId::from(format!("{wanted}_{n}"));
        }
        candidate
    }

    /// The nodes, in insertion order.
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// The edges, in insertion order.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// How many nodes there are.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// `true` while the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Whether the id names a node of this graph.
    pub fn contains(&self, id: &NodeId) -> bool {
        self.nodes.contains(id)
    }

    /// The nodes feeding into `id`, in their edges' insertion order.
    pub fn predecessors(&self, id: &NodeId) -> Vec<&NodeId> {
        self.edges
            .iter()
            .filter(|e| &e.target == id)
            .map(|e| &e.source)
            .collect()
    }

    /// The nodes `id` feeds into, in their edges' insertion order.
    pub fn successors(&self, id: &NodeId) -> Vec<&NodeId> {
        self.edges
            .iter()
            .filter(|e| &e.source == id)
            .map(|e| &e.target)
            .collect()
    }

    /// The nodes without predecessors: where execution enters.
    pub fn roots(&self) -> Vec<&NodeId> {
        self.nodes
            .iter()
            .filter(|id| !self.edges.iter().any(|e| e.target == **id))
            .collect()
    }

    /// The nodes without successors: where it leaves.
    pub fn leaves(&self) -> Vec<&NodeId> {
        self.nodes
            .iter()
            .filter(|id| !self.edges.iter().any(|e| e.source == **id))
            .collect()
    }

    /// The nodes in an order where each comes after its predecessors. Ties
    /// break by insertion order, so it is deterministic.
    pub fn topological_sort(&self) -> Vec<&NodeId> {
        let mut pending: Vec<usize> = self
            .nodes
            .iter()
            .map(|id| self.predecessors(id).len())
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut placed = HashSet::new();

        while order.len() < self.nodes.len() {
            let next = pending
                .iter()
                .enumerate()
                .find(|(i, n)| **n == 0 && !placed.contains(i))
                .map(|(i, _)| i)
                .expect("a non-empty DAG always has a node with nothing pending");

            placed.insert(next);
            let id = &self.nodes[next];
            for succ in self.successors(id) {
                let i = self
                    .nodes
                    .iter()
                    .position(|n| n == succ)
                    .expect("an edge only points at nodes of the graph");
                pending[i] -= 1;
            }
            order.push(id);
        }
        order
    }

    /// Whether `from` reaches `to` by following edges. This is the cycle check.
    fn reaches(&self, from: &NodeId, to: &NodeId) -> bool {
        let mut frontier = vec![from];
        let mut seen = HashSet::new();
        while let Some(current) = frontier.pop() {
            if current == to {
                return true;
            }
            if seen.insert(current) {
                frontier.extend(self.successors(current));
            }
        }
        false
    }
}

/// An attempt to build a graph that cannot exist.
///
/// The four ways to break the invariant, returned at insertion time: there is
/// no `validate()` afterwards, because there is no instant at which the graph
/// is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// There is already a node with that id.
    DuplicateNode(NodeId),
    /// The id names no node of this graph.
    UnknownNode(NodeId),
    /// That edge is already there.
    DuplicateEdge {
        /// Source.
        from: NodeId,
        /// Target.
        to: NodeId,
    },
    /// Adding that edge would close a cycle.
    WouldCycle {
        /// Source.
        from: NodeId,
        /// Target, which already reaches the source.
        to: NodeId,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNode(id) => write!(f, "there is already a node called `{id}`"),
            Self::UnknownNode(id) => write!(f, "`{id}` names no node of this graph"),
            Self::DuplicateEdge { from, to } => {
                write!(f, "the edge `{from}` → `{to}` already exists")
            }
            Self::WouldCycle { from, to } => write!(
                f,
                "the edge `{from}` → `{to}` would close a cycle: `{to}` already reaches `{from}`"
            ),
        }
    }
}

impl std::error::Error for GraphError {}
