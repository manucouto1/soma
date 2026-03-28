use crate::error::{Result, SomaError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Unique identifier for a node in a graph.
///
/// Currently a type alias. Will be promoted to a newtype in a future version
/// for stronger type safety (tracked in architecture-review.md).
pub type NodeId = String;

/// Unique identifier for an edge in a graph.
pub type EdgeId = String;

/// A node in the computational graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub label: String,
    pub filter_name: String,
}

impl Node {
    pub fn new(id: impl Into<String>, label: impl Into<String>, filter_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            filter_name: filter_name.into(),
        }
    }
}

/// Type of connection between nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Normal data flow: output of source becomes input of target.
    Data,
    /// Control flow edge (for conditional/loop logic).
    Control,
}

/// A directed edge connecting two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub kind: EdgeKind,
    pub label: Option<String>,
}

impl Edge {
    pub fn data(id: impl Into<String>, source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            source: source.into(),
            target: target.into(),
            kind: EdgeKind::Data,
            label: None,
        }
    }

    pub fn control(id: impl Into<String>, source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            source: source.into(),
            target: target.into(),
            kind: EdgeKind::Control,
            label: None,
        }
    }
}

/// A directed graph of computational nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    /// Get a node by its ID.
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Get all node IDs.
    pub fn node_ids(&self) -> Vec<&str> {
        self.nodes.iter().map(|n| n.id.as_str()).collect()
    }

    /// Get predecessors of a node (nodes with edges pointing to it).
    pub fn predecessors(&self, node_id: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.target == node_id)
            .map(|e| e.source.as_str())
            .collect()
    }

    /// Get successors of a node (nodes it points to).
    pub fn successors(&self, node_id: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.source == node_id)
            .map(|e| e.target.as_str())
            .collect()
    }

    /// Find root nodes (no incoming edges).
    pub fn roots(&self) -> Vec<&str> {
        let has_incoming: HashSet<&str> = self.edges.iter().map(|e| e.target.as_str()).collect();
        self.nodes
            .iter()
            .filter(|n| !has_incoming.contains(n.id.as_str()))
            .map(|n| n.id.as_str())
            .collect()
    }

    /// Find leaf nodes (no outgoing edges).
    pub fn leaves(&self) -> Vec<&str> {
        let has_outgoing: HashSet<&str> = self.edges.iter().map(|e| e.source.as_str()).collect();
        self.nodes
            .iter()
            .filter(|n| !has_outgoing.contains(n.id.as_str()))
            .map(|n| n.id.as_str())
            .collect()
    }

    /// Compute in-degree for each node.
    fn in_degrees(&self) -> HashMap<&str, usize> {
        let mut degrees: HashMap<&str, usize> =
            self.nodes.iter().map(|n| (n.id.as_str(), 0)).collect();
        for edge in &self.edges {
            *degrees.entry(edge.target.as_str()).or_insert(0) += 1;
        }
        degrees
    }

    /// Topological sort using Kahn's algorithm.
    /// Returns Err if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<&str>> {
        let mut in_deg = self.in_degrees();
        let mut queue: Vec<&str> = in_deg
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort(); // deterministic order

        let mut sorted = Vec::with_capacity(self.nodes.len());

        while let Some(node) = queue.pop() {
            sorted.push(node);
            let mut next = Vec::new();
            for succ in self.successors(node) {
                if let Some(deg) = in_deg.get_mut(succ) {
                    *deg -= 1;
                    if *deg == 0 {
                        next.push(succ);
                    }
                }
            }
            next.sort();
            // Insert at beginning so we process in deterministic order
            for n in next.into_iter().rev() {
                queue.push(n);
            }
        }

        if sorted.len() != self.nodes.len() {
            return Err(SomaError::CycleDetected);
        }

        Ok(sorted)
    }

    /// Validate the graph structure.
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate node IDs
        let mut seen = HashSet::new();
        for node in &self.nodes {
            if !seen.insert(&node.id) {
                return Err(SomaError::Compilation(format!(
                    "duplicate node id: `{}`",
                    node.id
                )));
            }
        }

        // Check that all edge endpoints reference existing nodes
        let node_ids: HashSet<&str> = self.nodes.iter().map(|n| n.id.as_str()).collect();
        for edge in &self.edges {
            if !node_ids.contains(edge.source.as_str()) {
                return Err(SomaError::NodeNotFound(edge.source.clone()));
            }
            if !node_ids.contains(edge.target.as_str()) {
                return Err(SomaError::NodeNotFound(edge.target.clone()));
            }
        }

        // Check for cycles
        self.topological_sort()?;

        Ok(())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing linear pipelines easily.
pub fn linear_pipeline(nodes: Vec<Node>) -> Graph {
    let mut graph = Graph::new();
    for (i, node) in nodes.iter().enumerate() {
        graph.add_node(node.clone());
        if i > 0 {
            graph.add_edge(Edge::data(
                format!("e_{}", i),
                &nodes[i - 1].id,
                &node.id,
            ));
        }
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_linear_graph() -> Graph {
        linear_pipeline(vec![
            Node::new("a", "Scaler", "StandardScaler"),
            Node::new("b", "PCA", "PCA"),
            Node::new("c", "SVM", "SVM"),
        ])
    }

    #[test]
    fn linear_pipeline_structure() {
        let g = sample_linear_graph();
        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.edges.len(), 2);
    }

    #[test]
    fn roots_and_leaves() {
        let g = sample_linear_graph();
        assert_eq!(g.roots(), vec!["a"]);
        assert_eq!(g.leaves(), vec!["c"]);
    }

    #[test]
    fn predecessors_and_successors() {
        let g = sample_linear_graph();
        assert!(g.predecessors("a").is_empty());
        assert_eq!(g.predecessors("b"), vec!["a"]);
        assert_eq!(g.successors("a"), vec!["b"]);
        assert_eq!(g.successors("b"), vec!["c"]);
        assert!(g.successors("c").is_empty());
    }

    #[test]
    fn topological_sort_linear() {
        let g = sample_linear_graph();
        let sorted = g.topological_sort().unwrap();
        assert_eq!(sorted, vec!["a", "b", "c"]);
    }

    #[test]
    fn topological_sort_parallel() {
        let mut g = Graph::new();
        g.add_node(Node::new("root", "Root", "Input"));
        g.add_node(Node::new("b1", "Branch1", "F1"));
        g.add_node(Node::new("b2", "Branch2", "F2"));
        g.add_node(Node::new("merge", "Merge", "Merge"));
        g.add_edge(Edge::data("e1", "root", "b1"));
        g.add_edge(Edge::data("e2", "root", "b2"));
        g.add_edge(Edge::data("e3", "b1", "merge"));
        g.add_edge(Edge::data("e4", "b2", "merge"));

        let sorted = g.topological_sort().unwrap();
        // root must be first, merge must be last
        assert_eq!(sorted[0], "root");
        assert_eq!(sorted[3], "merge");
        // b1 and b2 can be in any order between root and merge
        let middle: HashSet<&str> = sorted[1..3].iter().copied().collect();
        assert!(middle.contains("b1"));
        assert!(middle.contains("b2"));
    }

    #[test]
    fn topological_sort_detects_cycle() {
        let mut g = Graph::new();
        g.add_node(Node::new("a", "A", "F"));
        g.add_node(Node::new("b", "B", "F"));
        g.add_edge(Edge::data("e1", "a", "b"));
        g.add_edge(Edge::data("e2", "b", "a")); // cycle!

        let result = g.topological_sort();
        assert!(matches!(result, Err(SomaError::CycleDetected)));
    }

    #[test]
    fn validate_accepts_valid_graph() {
        let g = sample_linear_graph();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn validate_rejects_duplicate_ids() {
        let mut g = Graph::new();
        g.add_node(Node::new("a", "A", "F"));
        g.add_node(Node::new("a", "A2", "F"));
        assert!(matches!(g.validate(), Err(SomaError::Compilation(_))));
    }

    #[test]
    fn validate_rejects_missing_edge_target() {
        let mut g = Graph::new();
        g.add_node(Node::new("a", "A", "F"));
        g.add_edge(Edge::data("e1", "a", "nonexistent"));
        assert!(matches!(g.validate(), Err(SomaError::NodeNotFound(_))));
    }

    #[test]
    fn graph_serde_roundtrip() {
        let g = sample_linear_graph();
        let json = serde_json::to_string(&g).unwrap();
        let deserialized: Graph = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.nodes.len(), 3);
        assert_eq!(deserialized.edges.len(), 2);
    }

    #[test]
    fn empty_graph_is_valid() {
        let g = Graph::new();
        assert!(g.validate().is_ok());
        assert!(g.topological_sort().unwrap().is_empty());
    }

    #[test]
    fn single_node_graph() {
        let mut g = Graph::new();
        g.add_node(Node::new("solo", "Solo", "F"));
        assert_eq!(g.roots(), vec!["solo"]);
        assert_eq!(g.leaves(), vec!["solo"]);
        assert_eq!(g.topological_sort().unwrap(), vec!["solo"]);
    }
}
